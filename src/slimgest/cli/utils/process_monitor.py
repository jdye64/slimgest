"""
Process monitoring utility for tracking CPU, memory, disk I/O, and GPU usage across all spawned processes.

This module provides a background monitor that tracks resource usage of a parent process
and all its children, saving time-series metrics and generating graphs.
"""

from __future__ import annotations

import psutil
import threading
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# GPU monitoring support (optional)
try:
    import pynvml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    pynvml = None


class ProcessMonitor:
    """Monitor CPU, memory, disk I/O, and GPU usage for a process and all its children."""
    
    def __init__(self, pid: Optional[int] = None, sample_interval: float = 1.0):
        """
        Initialize the process monitor.
        
        Args:
            pid: Process ID to monitor. If None, monitors current process.
            sample_interval: Time in seconds between samples (default: 1.0)
        """
        self.pid = pid or psutil.Process().pid
        self.sample_interval = sample_interval
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Time series data
        self.timestamps: List[float] = []
        self.cpu_percent: List[float] = []
        self.memory_mb: List[float] = []
        self.disk_read_mb: List[float] = []
        self.disk_write_mb: List[float] = []
        self.num_processes: List[int] = []
        
        # GPU metrics (per-GPU lists)
        self.gpu_utilization: List[List[float]] = []  # [gpu_id][timestamp]
        self.gpu_memory_used_mb: List[List[float]] = []  # [gpu_id][timestamp]
        self.gpu_memory_total_mb: List[float] = []  # Total memory per GPU (constant)
        self.gpu_count = 0
        self.gpu_names: List[str] = []
        self.gpu_enabled = False
        
        # Stage/phase annotations for the X-axis
        self.stage_annotations: List[Dict[str, Any]] = []  # List of {timestamp, stage_name}
        
        # Initial disk I/O counters for calculating deltas
        self._last_disk_read = 0
        self._last_disk_write = 0
        self._start_time = None
        
        # Cache Process objects to maintain CPU tracking state
        self._process_cache: Dict[int, psutil.Process] = {}
        
        # Flag to track if per-process I/O is available
        self._use_system_io = False
        self._last_system_io_read = 0
        self._last_system_io_write = 0
        
        # Initialize GPU monitoring
        self._init_gpu_monitoring()
    
    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring if NVIDIA GPUs are available."""
        if not GPU_AVAILABLE:
            return
        
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            
            if self.gpu_count > 0:
                self.gpu_enabled = True
                print(f"\n{'='*60}")
                print(f"GPU Monitoring Enabled - {self.gpu_count} GPU(s) detected")
                print(f"{'='*60}")
                
                for i in range(self.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_memory_mb = memory_info.total / (1024 * 1024)
                    
                    self.gpu_names.append(name)
                    self.gpu_memory_total_mb.append(total_memory_mb)
                    self.gpu_utilization.append([])
                    self.gpu_memory_used_mb.append([])
                    
                    print(f"GPU {i}: {name}")
                    print(f"  Total Memory: {total_memory_mb:.0f} MB ({memory_info.total / (1024**3):.2f} GB)")
                
                print(f"{'='*60}\n")
        except Exception as e:
            print(f"Warning: Could not initialize GPU monitoring: {e}")
            self.gpu_enabled = False
    
    def start(self):
        """Start monitoring in a background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._start_time = time.time()
        
        # Try to get initial disk I/O counters
        try:
            process = psutil.Process(self.pid)
            io_counters = process.io_counters()
            self._last_disk_read = io_counters.read_bytes
            self._last_disk_write = io_counters.write_bytes
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError, NotImplementedError):
            # Fall back to system-wide I/O monitoring
            self._use_system_io = True
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self._last_system_io_read = disk_io.read_bytes
                    self._last_system_io_write = disk_io.write_bytes
            except (AttributeError, NotImplementedError):
                pass
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def annotate_stage(self, stage_name: str):
        """
        Annotate the current time with a processing stage name.
        This will be displayed on graphs as vertical lines or markers.
        
        Args:
            stage_name: Name of the processing stage (e.g., "PDF Splitting", "Page Elements")
        """
        if self._start_time is not None:
            elapsed_time = time.time() - self._start_time
            self.stage_annotations.append({
                'timestamp': elapsed_time,
                'stage_name': stage_name
            })
    
    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        # Cleanup GPU monitoring
        if self.gpu_enabled and GPU_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass  # Ignore errors during cleanup
            
    def _get_all_processes(self, parent_pid: int) -> List[psutil.Process]:
        """Get parent process and all children recursively."""
        processes = []
        try:
            parent = psutil.Process(parent_pid)
            processes.append(parent)
            
            # Get all children recursively
            children = parent.children(recursive=True)
            processes.extend(children)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        return processes
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                self._collect_sample()
            except Exception as e:
                print(f"Error collecting monitoring sample: {e}")
            
            time.sleep(self.sample_interval)
    
    def _collect_sample(self):
        """Collect a single sample of resource usage."""
        timestamp = time.time() - self._start_time
        
        # Get all processes (parent + children) - these are fresh Process objects
        discovered_processes = self._get_all_processes(self.pid)
        
        if not discovered_processes:
            return
        
        # Update our process cache with new processes
        # Key insight: we need to cache Process objects to maintain CPU tracking state
        current_pids = set()
        for proc in discovered_processes:
            try:
                proc_pid = proc.pid
                current_pids.add(proc_pid)
                
                if proc_pid not in self._process_cache:
                    # New process - cache it and initialize CPU tracking
                    self._process_cache[proc_pid] = proc
                    # First call establishes baseline (will return 0.0 but that's OK)
                    proc.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Clean up cache for processes that no longer exist
        dead_pids = set(self._process_cache.keys()) - current_pids
        for pid in dead_pids:
            del self._process_cache[pid]
        
        # Aggregate metrics across all CACHED processes
        # This ensures CPU tracking state is maintained
        total_cpu = 0.0
        total_memory_bytes = 0
        total_disk_read = 0
        total_disk_write = 0
        
        for proc_pid, proc in list(self._process_cache.items()):
            try:
                # CPU percentage - non-blocking call on cached Process object
                # This maintains internal state and gives us accurate readings
                cpu = proc.cpu_percent(interval=None)
                total_cpu += cpu
                
                # Memory usage
                mem_info = proc.memory_info()
                total_memory_bytes += mem_info.rss
                
                # Disk I/O - try to get cumulative counters
                try:
                    io_counters = proc.io_counters()
                    total_disk_read += io_counters.read_bytes
                    total_disk_write += io_counters.write_bytes
                except (AttributeError, psutil.AccessDenied, NotImplementedError):
                    # Some platforms don't support I/O counters or access denied
                    pass
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process died, remove from cache
                if proc_pid in self._process_cache:
                    del self._process_cache[proc_pid]
                continue
        
        # Calculate disk I/O rates (bytes per sample interval)
        disk_read_delta = 0
        disk_write_delta = 0
        
        # Try per-process I/O first
        if total_disk_read > 0 or total_disk_write > 0:
            disk_read_delta = max(0, total_disk_read - self._last_disk_read)
            disk_write_delta = max(0, total_disk_write - self._last_disk_write)
            
            self._last_disk_read = total_disk_read
            self._last_disk_write = total_disk_write
        elif self._use_system_io:
            # Fall back to system-wide I/O
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    current_read = disk_io.read_bytes
                    current_write = disk_io.write_bytes
                    
                    disk_read_delta = max(0, current_read - self._last_system_io_read)
                    disk_write_delta = max(0, current_write - self._last_system_io_write)
                    
                    self._last_system_io_read = current_read
                    self._last_system_io_write = current_write
            except (AttributeError, NotImplementedError):
                pass
        
        # Collect GPU metrics
        if self.gpu_enabled:
            self._collect_gpu_sample()
        
        # Store the sample
        self.timestamps.append(timestamp)
        self.cpu_percent.append(total_cpu)
        self.memory_mb.append(total_memory_bytes / (1024 * 1024))
        self.disk_read_mb.append(disk_read_delta / (1024 * 1024))
        self.disk_write_mb.append(disk_write_delta / (1024 * 1024))
        self.num_processes.append(len(self._process_cache))
    
    def _collect_gpu_sample(self):
        """Collect GPU utilization and memory usage for all GPUs."""
        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                
                # Get memory usage
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used_mb = memory_info.used / (1024 * 1024)
                
                # Store metrics
                self.gpu_utilization[i].append(gpu_util)
                self.gpu_memory_used_mb[i].append(memory_used_mb)
                
        except Exception as e:
            # If we can't read GPU metrics, append zeros
            for i in range(self.gpu_count):
                self.gpu_utilization[i].append(0)
                self.gpu_memory_used_mb[i].append(0)
    
    def save_metrics(self, output_path: Path):
        """
        Save raw time-series metrics to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
        """
        # Build GPU metadata
        gpu_metadata = {}
        if self.gpu_enabled:
            gpu_metadata = {
                "gpu_count": self.gpu_count,
                "gpu_names": self.gpu_names,
                "gpu_memory_total_mb": self.gpu_memory_total_mb
            }
        
        # Build time series data
        time_series_data = {
            "timestamps": self.timestamps,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "disk_read_mb": self.disk_read_mb,
            "disk_write_mb": self.disk_write_mb,
            "num_processes": self.num_processes
        }
        
        # Add GPU time series if enabled
        if self.gpu_enabled:
            time_series_data["gpu_utilization"] = self.gpu_utilization
            time_series_data["gpu_memory_used_mb"] = self.gpu_memory_used_mb
        
        # Build summary statistics
        summary_stats = {
            "avg_cpu_percent": sum(self.cpu_percent) / len(self.cpu_percent) if self.cpu_percent else 0,
            "max_cpu_percent": max(self.cpu_percent) if self.cpu_percent else 0,
            "avg_memory_mb": sum(self.memory_mb) / len(self.memory_mb) if self.memory_mb else 0,
            "max_memory_mb": max(self.memory_mb) if self.memory_mb else 0,
            "total_disk_read_mb": sum(self.disk_read_mb) if self.disk_read_mb else 0,
            "total_disk_write_mb": sum(self.disk_write_mb) if self.disk_write_mb else 0,
            "avg_num_processes": sum(self.num_processes) / len(self.num_processes) if self.num_processes else 0,
            "max_num_processes": max(self.num_processes) if self.num_processes else 0,
        }
        
        # Add GPU summary statistics
        if self.gpu_enabled:
            gpu_summary = {}
            for i in range(self.gpu_count):
                gpu_util = self.gpu_utilization[i]
                gpu_mem = self.gpu_memory_used_mb[i]
                gpu_summary[f"gpu_{i}"] = {
                    "name": self.gpu_names[i],
                    "avg_utilization": sum(gpu_util) / len(gpu_util) if gpu_util else 0,
                    "max_utilization": max(gpu_util) if gpu_util else 0,
                    "avg_memory_used_mb": sum(gpu_mem) / len(gpu_mem) if gpu_mem else 0,
                    "max_memory_used_mb": max(gpu_mem) if gpu_mem else 0,
                    "total_memory_mb": self.gpu_memory_total_mb[i]
                }
            summary_stats["gpus"] = gpu_summary
        
        metrics = {
            "metadata": {
                "pid": self.pid,
                "sample_interval": self.sample_interval,
                "start_time": datetime.fromtimestamp(self._start_time).isoformat() if self._start_time else None,
                "duration_seconds": self.timestamps[-1] if self.timestamps else 0,
                "num_samples": len(self.timestamps),
                **gpu_metadata
            },
            "time_series": time_series_data,
            "summary": summary_stats,
            "stage_annotations": self.stage_annotations  # Add stage annotations
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def generate_graphs(self, output_path: Path):
        """
        Generate graphs of resource usage and save as PNG.
        
        Args:
            output_path: Path to save the PNG file
        """
        if not self.timestamps:
            print("No data to plot")
            return
        
        # Convert timestamps to minutes for better readability
        time_minutes = [t / 60 for t in self.timestamps]
        
        # Determine layout based on GPU availability
        if self.gpu_enabled and self.gpu_count > 0:
            # Create a figure with 6 subplots (3 rows x 2 cols)
            fig, axes = plt.subplots(3, 2, figsize=(16, 18))
            fig.suptitle('Process Resource Monitoring (CPU, Memory, Disk, GPU)', fontsize=16, fontweight='bold')
        else:
            # Create a figure with 4 subplots (2 rows x 2 cols)
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Process Resource Monitoring', fontsize=16, fontweight='bold')
        
        # Plot 1: CPU Usage
        ax1 = axes[0, 0]
        ax1.plot(time_minutes, self.cpu_percent, color='#2E86AB', linewidth=1.5)
        ax1.fill_between(time_minutes, self.cpu_percent, alpha=0.3, color='#2E86AB')
        ax1.set_xlabel('Time (minutes)', fontsize=11)
        ax1.set_ylabel('CPU Usage (%)', fontsize=11)
        ax1.set_title('CPU Usage Over Time', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add average line
        if self.cpu_percent:
            avg_cpu = sum(self.cpu_percent) / len(self.cpu_percent)
            ax1.axhline(y=avg_cpu, color='r', linestyle='--', linewidth=1, 
                       label=f'Average: {avg_cpu:.1f}%')
            ax1.legend(loc='upper right', fontsize=9)
        
        # Plot 2: Memory Usage
        ax2 = axes[0, 1]
        ax2.plot(time_minutes, self.memory_mb, color='#A23B72', linewidth=1.5)
        ax2.fill_between(time_minutes, self.memory_mb, alpha=0.3, color='#A23B72')
        ax2.set_xlabel('Time (minutes)', fontsize=11)
        ax2.set_ylabel('Memory Usage (MB)', fontsize=11)
        ax2.set_title('Memory Usage Over Time', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add average line
        if self.memory_mb:
            avg_mem = sum(self.memory_mb) / len(self.memory_mb)
            ax2.axhline(y=avg_mem, color='r', linestyle='--', linewidth=1,
                       label=f'Average: {avg_mem:.0f} MB')
            ax2.legend(loc='upper right', fontsize=9)
        
        # Plot 3: Disk I/O
        ax3 = axes[1, 0]
        ax3.plot(time_minutes, self.disk_read_mb, color='#18A558', linewidth=1.5, 
                label='Read', marker='o', markersize=2, markevery=max(1, len(time_minutes)//50))
        ax3.plot(time_minutes, self.disk_write_mb, color='#F18F01', linewidth=1.5,
                label='Write', marker='s', markersize=2, markevery=max(1, len(time_minutes)//50))
        ax3.set_xlabel('Time (minutes)', fontsize=11)
        ax3.set_ylabel('Disk I/O Rate (MB/sample)', fontsize=11)
        ax3.set_title('Disk I/O Over Time', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Number of Processes
        ax4 = axes[1, 1]
        ax4.plot(time_minutes, self.num_processes, color='#C73E1D', linewidth=1.5)
        ax4.fill_between(time_minutes, self.num_processes, alpha=0.3, color='#C73E1D')
        ax4.set_xlabel('Time (minutes)', fontsize=11)
        ax4.set_ylabel('Number of Processes', fontsize=11)
        ax4.set_title('Active Processes Over Time', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add average line
        if self.num_processes:
            avg_procs = sum(self.num_processes) / len(self.num_processes)
            ax4.axhline(y=avg_procs, color='r', linestyle='--', linewidth=1,
                       label=f'Average: {avg_procs:.1f}')
            ax4.legend(loc='upper right', fontsize=9)
        
        # Plot 5 & 6: GPU metrics (if available)
        if self.gpu_enabled and self.gpu_count > 0:
            # Colors for different GPUs
            gpu_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
            
            # Plot 5: GPU Utilization
            ax5 = axes[2, 0]
            for i in range(self.gpu_count):
                color = gpu_colors[i % len(gpu_colors)]
                gpu_name = self.gpu_names[i] if i < len(self.gpu_names) else f"GPU {i}"
                # Truncate long GPU names
                display_name = gpu_name if len(gpu_name) <= 30 else gpu_name[:27] + "..."
                ax5.plot(time_minutes, self.gpu_utilization[i], color=color, linewidth=1.5,
                        label=f'{display_name}', alpha=0.8)
            
            ax5.set_xlabel('Time (minutes)', fontsize=11)
            ax5.set_ylabel('GPU Utilization (%)', fontsize=11)
            ax5.set_title('GPU Utilization Over Time', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            ax5.set_ylim([0, 105])
            if self.gpu_count <= 4:  # Only show legend if not too many GPUs
                ax5.legend(loc='upper right', fontsize=8)
            
            # Plot 6: GPU Memory Usage
            ax6 = axes[2, 1]
            for i in range(self.gpu_count):
                color = gpu_colors[i % len(gpu_colors)]
                gpu_name = self.gpu_names[i] if i < len(self.gpu_names) else f"GPU {i}"
                display_name = gpu_name if len(gpu_name) <= 30 else gpu_name[:27] + "..."
                gpu_mem_gb = [mb / 1024 for mb in self.gpu_memory_used_mb[i]]
                total_gb = self.gpu_memory_total_mb[i] / 1024 if i < len(self.gpu_memory_total_mb) else 0
                
                ax6.plot(time_minutes, gpu_mem_gb, color=color, linewidth=1.5,
                        label=f'{display_name} ({total_gb:.1f} GB total)', alpha=0.8)
            
            ax6.set_xlabel('Time (minutes)', fontsize=11)
            ax6.set_ylabel('GPU Memory Usage (GB)', fontsize=11)
            ax6.set_title('GPU Memory Usage Over Time', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            if self.gpu_count <= 4:  # Only show legend if not too many GPUs
                ax6.legend(loc='upper right', fontsize=8)
        
        # Add stage annotations to all subplots
        if self.stage_annotations:
            # Convert stage timestamps to minutes
            stage_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
            
            # Flatten axes array for easy iteration
            all_axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
            
            for annotation_idx, annotation in enumerate(self.stage_annotations):
                stage_time_min = annotation['timestamp'] / 60
                stage_name = annotation['stage_name']
                color = stage_colors[annotation_idx % len(stage_colors)]
                
                # Add vertical line to each subplot
                for ax in all_axes:
                    ax.axvline(x=stage_time_min, color=color, linestyle='--', 
                              linewidth=1.5, alpha=0.7, zorder=5)
                
                # Add stage label on the top subplot only (to avoid clutter)
                top_ax = all_axes[0]
                # Position label slightly above the plot
                y_pos = top_ax.get_ylim()[1]
                top_ax.text(stage_time_min, y_pos, f' {stage_name}', 
                          rotation=90, verticalalignment='bottom', 
                          horizontalalignment='right',
                          fontsize=8, color=color, weight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor=color, alpha=0.8))
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save the figure
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved resource monitoring graph to: {output_path}")


def monitor_process(func):
    """
    Decorator to monitor a function's process resource usage.
    
    Usage:
        @monitor_process
        def my_function():
            # Your code here
            pass
    """
    def wrapper(*args, **kwargs):
        import os
        monitor = ProcessMonitor(pid=os.getpid())
        monitor.start()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            monitor.stop()
            
            # Save results to current directory
            monitor.save_metrics(Path("process_metrics.json"))
            monitor.generate_graphs(Path("process_monitoring.png"))
    
    return wrapper

