"""
Test script to validate the multiprocessing worker setup.
"""
import os
import time
from multiprocessing import Queue, Process

from slimgest.web.worker import start_worker


def test_worker_startup():
    """Test that workers can start up and load models."""
    print("Testing worker startup...")
    
    request_queue = Queue(maxsize=10)
    result_queue = Queue()
    
    # Use a test OCR model directory (or skip if not available)
    ocr_model_dir = os.environ.get(
        "NEMOTRON_OCR_MODEL_DIR",
        "/home/jdyer/Development/slim-gest/models/nemotron-ocr-v1/checkpoints"
    )
    
    # Start a single worker
    worker_id = 0
    p = Process(
        target=start_worker,
        args=(worker_id, request_queue, result_queue, ocr_model_dir),
        daemon=True,
    )
    
    print(f"Starting worker {worker_id}...")
    p.start()
    
    # Give it some time to load models
    print("Waiting for worker to load models (this may take a minute)...")
    time.sleep(5)
    
    if p.is_alive():
        print("✓ Worker is alive and running")
        
        # Send shutdown signal
        print("Sending shutdown signal...")
        request_queue.put({"type": "shutdown"})
        
        # Wait for graceful shutdown
        p.join(timeout=5)
        
        if not p.is_alive():
            print("✓ Worker shut down gracefully")
            return True
        else:
            print("✗ Worker did not shut down, terminating...")
            p.terminate()
            p.join()
            return False
    else:
        print("✗ Worker died during startup")
        return False


def test_ipc_queues():
    """Test basic IPC queue functionality."""
    print("\nTesting IPC queues...")
    
    request_queue = Queue(maxsize=10)
    result_queue = Queue()
    
    # Test putting and getting
    test_data = {"job_id": "test-123", "type": "test", "data": "hello"}
    request_queue.put(test_data)
    
    retrieved = request_queue.get()
    
    if retrieved == test_data:
        print("✓ IPC queues working correctly")
        return True
    else:
        print("✗ IPC queue data mismatch")
        return False


if __name__ == "__main__":
    print("="*60)
    print("Multiprocessing Worker Test Suite")
    print("="*60)
    
    results = []
    
    # Test IPC queues first (fast)
    results.append(("IPC Queues", test_ipc_queues()))
    
    # Test worker startup (slow - loads models)
    print("\nNote: Worker startup test will take a while as it loads ML models...")
    results.append(("Worker Startup", test_worker_startup()))
    
    # Print summary
    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
    
    exit(0 if all_passed else 1)
