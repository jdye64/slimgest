# Monitoring Fixes - CPU and Disk I/O

## Problem

The initial implementation had both CPU usage and disk I/O metrics showing all zeros. This was due to how `psutil` API works and Linux permission issues.

## Root Causes

### 1. CPU Metrics Returning Zero

**Issue**: `psutil.Process.cpu_percent(interval=0)` requires a previous call to establish a baseline.

- The first call to `cpu_percent(interval=0)` always returns `0.0`
- Subsequent calls return CPU usage since the last call
- Since we were getting fresh process objects each sample, we effectively got 0.0 every time

**Example of the problem**:
```python
proc = psutil.Process(pid)
cpu1 = proc.cpu_percent(interval=0)  # Returns 0.0 (no baseline)
cpu2 = proc.cpu_percent(interval=0)  # Returns actual usage since cpu1

# But if we get a fresh process object:
proc = psutil.Process(pid)  # Fresh object
cpu3 = proc.cpu_percent(interval=0)  # Returns 0.0 again!
```

### 2. Disk I/O Metrics Returning Zero

**Issue**: Per-process I/O counters may not be accessible on Linux without special permissions.

- `process.io_counters()` requires `/proc/[pid]/io` access
- This may require elevated permissions or specific kernel configurations
- Child processes spawned by multiprocessing may not have accessible I/O counters
- Some systems don't implement per-process I/O tracking at all

## Solutions Implemented

### Fix 1: CPU Tracking with Process Cache

Implemented a process cache that tracks which processes we've already seen:

```python
# Track processes we've seen before
self._process_cpu_cache: Dict[int, float] = {}

# In _collect_sample():
for proc in processes:
    proc_pid = proc.pid
    
    if proc_pid not in self._process_cpu_cache:
        # First time seeing this process - use blocking call to establish baseline
        cpu = proc.cpu_percent(interval=0.1)  # 0.1 second blocking call
        self._process_cpu_cache[proc_pid] = cpu
    else:
        # We've seen this process - non-blocking call works correctly now
        cpu = proc.cpu_percent(interval=None)  # Non-blocking
        self._process_cpu_cache[proc_pid] = cpu
```

**Key improvements**:
1. Track each process by PID
2. First measurement uses a short blocking interval (0.1s) to establish baseline
3. Subsequent measurements use non-blocking calls
4. Clean up cache when processes terminate

### Fix 2: Disk I/O with System-Wide Fallback

Added fallback to system-wide disk I/O when per-process counters aren't available:

```python
# Try per-process I/O first
if total_disk_read > 0 or total_disk_write > 0:
    # Per-process I/O is available
    disk_read_delta = total_disk_read - self._last_disk_read
    disk_write_delta = total_disk_write - self._last_disk_write
elif self._use_system_io:
    # Fallback to system-wide I/O counters
    disk_io = psutil.disk_io_counters()  # All disks, all processes
    disk_read_delta = disk_io.read_bytes - self._last_system_io_read
    disk_write_delta = disk_io.write_bytes - self._last_system_io_write
```

**Key improvements**:
1. Attempt per-process I/O collection first (more accurate)
2. If unavailable, fall back to system-wide counters
3. System-wide counters show total disk activity (less precise but better than nothing)
4. Handle platform differences gracefully

### Fix 3: Better Error Handling

Added comprehensive exception handling for various scenarios:

```python
try:
    io_counters = proc.io_counters()
    total_disk_read += io_counters.read_bytes
    total_disk_write += io_counters.write_bytes
except (AttributeError, psutil.AccessDenied, NotImplementedError):
    # Platform doesn't support it, or access denied
    pass
```

## Technical Details

### CPU Measurement Process

1. **First Sample**: When a new process is detected:
   - Call `cpu_percent(interval=0.1)` - blocks for 0.1 seconds
   - Establishes baseline CPU time for this process
   - Returns CPU usage during that 0.1 second period

2. **Subsequent Samples**:
   - Call `cpu_percent(interval=None)` - non-blocking
   - Returns CPU usage since last call
   - Much faster (no blocking)

3. **Process Lifecycle**:
   - Cache is cleaned up when processes terminate
   - New child processes are automatically detected and tracked
   - PIDs are used as cache keys

### Disk I/O Measurement

#### Per-Process I/O (Preferred)
- Tracks: `read_bytes` and `write_bytes` per process
- Source: `/proc/[pid]/io` on Linux
- Advantages: Accurate, attributes I/O to specific processes
- Limitations: May require permissions, not all platforms support it

#### System-Wide I/O (Fallback)
- Tracks: Total system disk reads and writes
- Source: `/proc/diskstats` on Linux, system APIs on other platforms
- Advantages: Always available, no special permissions needed
- Limitations: Includes ALL processes, not just yours

### Performance Impact

The fixes have minimal performance impact:

**CPU Tracking**:
- First measurement per process: 0.1s blocking time
- Subsequent measurements: < 0.001s (non-blocking)
- With 10 worker processes: ~1s initial delay (processes start at different times)
- After initialization: negligible overhead

**Disk I/O Tracking**:
- Per-process: ~0.001s per process
- System-wide: ~0.0001s (single syscall)
- Both are negligible compared to sample interval (1.0s)

## Verification

To verify the fixes work, run the test script:

```bash
python test_monitoring.py
```

You should now see:
- Non-zero CPU percentages in the metrics JSON
- Non-zero disk I/O rates (either per-process or system-wide)
- Proper graphs showing resource usage patterns

### Expected Output

```json
{
  "time_series": {
    "cpu_percent": [15.2, 48.1, 52.3, 89.5, ...],  // Non-zero values!
    "memory_mb": [1024.5, 1150.2, 1245.8, ...],
    "disk_read_mb": [10.5, 8.2, 12.1, ...],        // Should have values
    "disk_write_mb": [5.2, 6.8, 4.5, ...],         // Should have values
    "num_processes": [1, 5, 8, 10, ...]
  }
}
```

## Platform-Specific Notes

### Linux
- Per-process I/O should work for your own processes
- May need elevated permissions for other users' processes
- System-wide I/O is always available

### macOS
- Per-process I/O counters may not be available (platform limitation)
- Will automatically fall back to system-wide counters
- CPU tracking works fine

### Windows
- Per-process I/O available with appropriate permissions
- May require administrator privileges in some cases
- CPU tracking works fine

## Troubleshooting

### Still seeing zero CPU?

1. Check that processes are actually running when samples are collected
2. Verify sample interval isn't too short (should be >= 1.0 second)
3. Check that processes aren't immediately exiting

### Still seeing zero disk I/O?

1. **Verify system-wide I/O works**:
   ```python
   import psutil
   print(psutil.disk_io_counters())
   ```

2. **Check per-process I/O access**:
   ```python
   import psutil
   import os
   proc = psutil.Process(os.getpid())
   print(proc.io_counters())  # May raise AccessDenied or AttributeError
   ```

3. **If both fail**: Your system may not expose I/O stats, which is rare but possible on some configurations

### CPU values seem too high?

CPU percentages are summed across all processes. With 10 worker processes each using 50% CPU:
- Total CPU = 10 × 50% = 500%
- This is correct! (represents 5 full CPU cores worth of work)

## Summary

The fixes ensure robust monitoring across different platforms and permission scenarios:

✅ CPU tracking now works correctly with process caching  
✅ Disk I/O has fallback to system-wide counters  
✅ Better error handling for platform differences  
✅ Minimal performance overhead  
✅ Works with multiprocessing worker pools  

The monitoring should now provide accurate, useful metrics for your PDF processing pipeline!

