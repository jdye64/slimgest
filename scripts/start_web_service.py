#!/usr/bin/env python3
"""
Simple script to start the Slim-Gest web service with recommended settings.
"""
import os
import sys


def main():
    """Start the Slim-Gest web service."""
    # Set recommended defaults
    if "SLIMGEST_NUM_WORKERS" not in os.environ:
        # Default to 2 workers
        os.environ["SLIMGEST_NUM_WORKERS"] = "2"
    
    if "NEMOTRON_OCR_MODEL_DIR" not in os.environ:
        # Try to find the OCR model directory
        possible_paths = [
            "/home/jdyer/Development/slim-gest/models/nemotron-ocr-v1/checkpoints",
            "./models/nemotron-ocr-v1/checkpoints",
            "../models/nemotron-ocr-v1/checkpoints",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                os.environ["NEMOTRON_OCR_MODEL_DIR"] = path
                print(f"Using OCR model directory: {path}")
                break
        else:
            print("Warning: NEMOTRON_OCR_MODEL_DIR not set and could not find model directory")
            print("Set NEMOTRON_OCR_MODEL_DIR environment variable to the OCR model path")
    
    # Print configuration
    print("="*60)
    print("Slim-Gest Web Service")
    print("="*60)
    print(f"Workers: {os.environ.get('SLIMGEST_NUM_WORKERS', 'not set')}")
    print(f"OCR Model: {os.environ.get('NEMOTRON_OCR_MODEL_DIR', 'not set')}")
    print("="*60)
    print()
    
    # Import and run
    try:
        from slimgest.web.__main__ import main as web_main
        web_main()
    except ImportError as e:
        print(f"Error: Could not import slimgest.web: {e}")
        print("Make sure you have installed the package:")
        print("  pip install -e .")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
