# Process Monitoring Documentation

This document describes the process monitoring capabilities added to the slimgest PDF processing pipeline.

## Overview

The monitoring system tracks CPU usage, memory consumption, and disk I/O across all processes spawned during PDF processing. It collects time-series data and generates visualizations to help understand resource utilization.

## Features

- **Real-time Monitoring**: Tracks resource usage while the PDF processing is running
- **Multi-process Tracking**: Monitors the main process and all child worker processes
- **Time-series Data**: Collects samples at regular intervals (default: 1 second)
- **JSON Export**: Saves raw metrics in JSON format for further analysis
- **Visualization**: Generates comprehensive graphs showing resource usage over time

## Metrics Tracked

1. **CPU Usage**: Total CPU percentage across all processes
2. **Memory Usage**: Total memory consumption (RSS) in MB
3. **Disk I/O**: Read and write rates in MB per sample
4. **Process Count**: Number of active processes at each sample point

## Files Generated

When you run the PDF processing pipeline, two additional files will be created in the scratch directory:

1. **`process_monitoring_metrics.json`**: Raw time-series data including:
   - Timestamps (seconds from start)
   - CPU percentages
   - Memory usage (MB)
   - Disk read/write rates (MB)
   - Number of active processes
   - Summary statistics (averages, peaks, totals)

2. **`process_monitoring_graph.png`**: A 4-panel visualization showing:
   - CPU usage over time
   - Memory usage over time
   - Disk I/O (read and write) over time
   - Number of active processes over time

## Usage

### Automatic Monitoring (Integrated)

The monitoring is automatically enabled when you run the PDF processing pipeline:

```bash
# The monitoring will automatically start and save results to the scratch directory
python -m slimgest.cli.local process <input_dir> <scratch_dir>
```

After processing completes, check the scratch directory for:
- `process_monitoring_metrics.json`
- `process_monitoring_graph.png`

### Standalone Testing

To test the monitoring functionality independently:

```bash
python test_monitoring.py
```

This will simulate some CPU, memory, and disk activity and save test results to the `scratch/` directory.

### Programmatic Usage

You can also use the monitoring module in your own scripts:

```python
from slimgest.cli.utils.process_monitor import ProcessMonitor
from pathlib import Path

# Initialize monitor
monitor = ProcessMonitor(sample_interval=1.0)  # Sample every 1 second

# Start monitoring
monitor.start()

# Your code here...
# All child processes will be automatically tracked

# Stop monitoring
monitor.stop()

# Save results
monitor.save_metrics(Path("metrics.json"))
monitor.generate_graphs(Path("monitoring.png"))
```

### Using the Decorator

For simple function monitoring:

```python
from slimgest.cli.utils.process_monitor import monitor_process

@monitor_process
def my_processing_function():
    # Your code here
    pass

# Monitoring results will be saved automatically
my_processing_function()
```

## JSON Metrics Format

The JSON file contains three main sections:

### Metadata
```json
{
  "metadata": {
    "pid": 12345,
    "sample_interval": 1.0,
    "start_time": "2025-11-20T10:30:00",
    "duration_seconds": 120.5,
    "num_samples": 120
  }
}
```

### Time Series
```json
{
  "time_series": {
    "timestamps": [0.0, 1.0, 2.0, ...],
    "cpu_percent": [45.2, 48.1, 52.3, ...],
    "memory_mb": [1024.5, 1150.2, 1245.8, ...],
    "disk_read_mb": [10.5, 8.2, 12.1, ...],
    "disk_write_mb": [5.2, 6.8, 4.5, ...],
    "num_processes": [1, 5, 8, 10, ...]
  }
}
```

### Summary Statistics
```json
{
  "summary": {
    "avg_cpu_percent": 47.5,
    "max_cpu_percent": 95.2,
    "avg_memory_mb": 1200.5,
    "max_memory_mb": 2048.7,
    "total_disk_read_mb": 1024.5,
    "total_disk_write_mb": 512.3,
    "avg_num_processes": 8.5,
    "max_num_processes": 12
  }
}
```

## Graph Interpretation

### CPU Usage Panel (Top Left)
- Shows total CPU usage across all processes as percentage
- Dashed red line indicates average CPU usage
- Useful for identifying CPU bottlenecks

### Memory Usage Panel (Top Right)
- Shows total memory consumption in MB
- Dashed red line indicates average memory usage
- Helps identify memory leaks or high memory requirements

### Disk I/O Panel (Bottom Left)
- Green line: disk read rate (MB per sample interval)
- Orange line: disk write rate (MB per sample interval)
- Useful for understanding I/O patterns

### Active Processes Panel (Bottom Right)
- Shows how many processes were active at each time point
- Reflects the parallelization level
- Corresponds to the `parallel_workers` configuration

## Configuration

The monitoring system uses these defaults:

- **Sample Interval**: 1.0 seconds (configurable in code)
- **Auto-start**: Yes (when using integrated mode)
- **Output Location**: Scratch directory specified in command line

## Dependencies

The monitoring feature requires:
- `psutil>=5.9.0` - For process information and resource usage
- `matplotlib>=3.8.0` - For generating graphs

These are automatically included in `pyproject.toml`.

## Performance Impact

The monitoring system is designed to have minimal performance impact:
- Runs in a separate background thread
- Uses efficient sampling (default 1 second intervals)
- Typical overhead: < 1% CPU, < 10 MB memory

## Troubleshooting

### No graphs generated
- Ensure matplotlib is installed: `pip install matplotlib`
- Check that the scratch directory has write permissions

### Missing disk I/O data
- Some platforms (e.g., macOS) may not support I/O counters
- Windows may require administrator privileges for some I/O metrics

### Process not tracked
- Ensure the monitoring starts before spawning child processes
- Check that processes aren't being created with `daemon=False` and `start_new_session=True`

## Examples

### Analyzing Processing Performance

After running the pipeline, you can analyze the JSON metrics:

```python
import json
from pathlib import Path

# Load metrics
with open("scratch/process_monitoring_metrics.json") as f:
    metrics = json.load(f)

# Print summary
summary = metrics["summary"]
print(f"Average CPU: {summary['avg_cpu_percent']:.1f}%")
print(f"Peak Memory: {summary['max_memory_mb']:.0f} MB")
print(f"Total Disk Read: {summary['total_disk_read_mb']:.0f} MB")
print(f"Total Disk Write: {summary['total_disk_write_mb']:.0f} MB")

# Analyze time series
ts = metrics["time_series"]
duration = ts["timestamps"][-1]
print(f"Total duration: {duration:.1f} seconds")
```

### Comparing Multiple Runs

```python
import json
import matplotlib.pyplot as plt

# Load multiple metric files
runs = ["run1", "run2", "run3"]
metrics = {}

for run in runs:
    with open(f"scratch/{run}/process_monitoring_metrics.json") as f:
        metrics[run] = json.load(f)

# Compare CPU usage
fig, ax = plt.subplots(figsize=(12, 6))
for run, data in metrics.items():
    ts = data["time_series"]
    ax.plot(ts["timestamps"], ts["cpu_percent"], label=run)

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("CPU Usage (%)")
ax.set_title("CPU Usage Comparison Across Runs")
ax.legend()
ax.grid(True)
plt.savefig("cpu_comparison.png")
```

## License

Same as the main project (MIT).

