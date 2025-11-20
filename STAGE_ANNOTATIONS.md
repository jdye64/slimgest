# Stage Annotations in Process Monitoring

## Overview
The process monitoring graphs now include **stage annotations** that show which part of the processing pipeline is running at any given time. This makes it easy to correlate resource usage (CPU, memory, GPU) with specific processing stages.

## Visual Features

### On the Graphs
- **Vertical dashed lines** appear at each stage transition
- **Color-coded labels** at the top of the graph identify each stage
- **Lines appear on all subplots** for easy correlation across metrics

### Stages Annotated in local.py

1. **Initialization** - Setting up directories and preparing
2. **Loading Models** - Loading AI models into memory (major memory spike expected)
3. **PDF Processing** - Active processing of PDFs (high CPU/GPU utilization)
4. **Saving Results** - Writing output files (disk I/O intensive)

## How to Read the Graphs

### Example Interpretation:
```
CPU Usage Graph:
├─ Initialization (low CPU)
├─ Loading Models (moderate CPU, high memory)
├─ PDF Processing (HIGH CPU/GPU - your main bottleneck)
└─ Saving Results (low CPU, high disk I/O)
```

### GPU Utilization:
- Check GPU graphs during "PDF Processing" stage
- Zero GPU usage outside this stage = models not using GPU
- Non-zero GPU = successful GPU acceleration

### Memory Usage:
- Sharp increase at "Loading Models" = model weights loaded
- Sustained during "PDF Processing" = models in memory
- May not drop immediately (Python garbage collection)

## In Your Output Files

### process_monitoring_metrics.json
Stage annotations are saved as:
```json
{
  "stage_annotations": [
    {"timestamp": 0.0, "stage_name": "Initialization"},
    {"timestamp": 5.2, "stage_name": "Loading Models"},
    {"timestamp": 12.8, "stage_name": "PDF Processing"},
    {"timestamp": 125.3, "stage_name": "Saving Results"}
  ]
}
```

### process_monitoring_graph.png
- Open the PNG file to see the visual stage markers
- Each stage shows as a vertical dashed line with label
- Colors rotate through a palette for easy distinction

## Benefits

✅ **Identify bottlenecks**: See which stage consumes most resources
✅ **Optimize timing**: Know exactly how long each phase takes
✅ **Debug issues**: Correlate errors with specific processing stages
✅ **GPU verification**: Confirm GPU is actually being used during inference
✅ **Memory leaks**: Spot if memory doesn't release after stages

## Custom Annotations

You can add your own stage annotations in any Python code:

```python
from slimgest.cli.utils.process_monitor import ProcessMonitor

monitor = ProcessMonitor()
monitor.start()

# Your processing stages
monitor.annotate_stage("Stage 1: Data Loading")
# ... do work ...

monitor.annotate_stage("Stage 2: Model Inference")
# ... do work ...

monitor.stop()
monitor.save_metrics(Path("metrics.json"))
monitor.generate_graphs(Path("graph.png"))
```

## Next Steps

Run your normal processing command:
```bash
slimgest-local /path/to/pdfs ./scratch
```

Then check `scratch/process_monitoring_graph.png` to see your processing stages with resource usage!
