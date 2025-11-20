#!/usr/bin/env python3
"""
Simple test script to demonstrate the process monitoring functionality.

This script simulates CPU, memory, and disk I/O activity to verify the monitoring works.
"""

import time
import os
import multiprocessing as mp
from pathlib import Path

# Import the monitoring module
from src.slimgest.cli.utils.process_monitor import ProcessMonitor


def worker_task(duration: int):
    """Simulate work by doing some CPU and memory intensive operations."""
    print(f"Worker {os.getpid()} started")
    
    # Allocate some memory
    data = []
    for i in range(10):
        # Create a 10MB list
        data.append([0] * (1024 * 1024))
        time.sleep(duration / 10)
        
        # Do some CPU work
        result = sum(range(1000000))
    
    # Write some data to disk
    temp_file = Path(f"/tmp/test_worker_{os.getpid()}.dat")
    with open(temp_file, 'wb') as f:
        f.write(b'0' * (10 * 1024 * 1024))  # 10 MB
    
    # Clean up
    temp_file.unlink()
    
    print(f"Worker {os.getpid()} finished")


def main():
    """Main test function."""
    print("Starting process monitoring test...")
    print(f"Main process PID: {os.getpid()}")
    
    # Create scratch directory if it doesn't exist
    scratch_dir = Path("scratch")
    scratch_dir.mkdir(exist_ok=True)
    
    # Initialize monitor
    monitor = ProcessMonitor(sample_interval=0.5)  # Sample every 0.5 seconds
    monitor.start()
    
    print("\nSpawning worker processes to simulate activity...")
    
    # Spawn some worker processes to simulate the real workload
    num_workers = 4
    duration = 10  # seconds per worker
    
    with mp.Pool(processes=num_workers) as pool:
        # Run workers in parallel
        pool.map(worker_task, [duration] * num_workers)
    
    print("\nAll workers finished. Stopping monitoring...")
    
    # Stop monitoring
    monitor.stop()
    
    # Save results
    metrics_path = scratch_dir / "test_monitoring_metrics.json"
    graph_path = scratch_dir / "test_monitoring_graph.png"
    
    monitor.save_metrics(metrics_path)
    monitor.generate_graphs(graph_path)
    
    print(f"\n✓ Saved metrics to: {metrics_path}")
    print(f"✓ Saved graph to: {graph_path}")
    
    # Print summary
    if monitor.timestamps:
        print(f"\nMonitoring Summary:")
        print(f"  Duration: {monitor.timestamps[-1]:.1f} seconds")
        print(f"  Samples collected: {len(monitor.timestamps)}")
        print(f"  Average CPU: {sum(monitor.cpu_percent) / len(monitor.cpu_percent):.1f}%")
        print(f"  Peak memory: {max(monitor.memory_mb):.0f} MB")
        print(f"  Max processes: {max(monitor.num_processes)}")


if __name__ == "__main__":
    main()

