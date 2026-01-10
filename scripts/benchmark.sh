#!/bin/bash
# Quick setup script for building and running the benchmark

set -e

echo "=========================================="
echo "Slim-Gest Benchmark Setup"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "Error: docker-compose is not available"
    exit 1
fi

# Parse command line arguments
ACTION=${1:-help}

case $ACTION in
    build)
        echo "Building Docker image..."
        docker build -t slimgest .
        echo "✓ Build complete"
        ;;
    
    up)
        echo "Starting servers with docker-compose..."
        $COMPOSE_CMD up -d
        echo "✓ Servers started"
        echo ""
        echo "Python server: http://localhost:7670"
        echo "Rust server:   http://localhost:7671"
        echo ""
        echo "Check logs with: $COMPOSE_CMD logs -f"
        ;;
    
    down)
        echo "Stopping servers..."
        $COMPOSE_CMD down
        echo "✓ Servers stopped"
        ;;
    
    logs)
        $COMPOSE_CMD logs -f
        ;;
    
    test)
        if [ -z "$2" ]; then
            echo "Error: Please provide a test PDF file"
            echo "Usage: ./scripts/benchmark.sh test <path_to_pdf>"
            exit 1
        fi
        echo "Testing servers with $2..."
        python examples/test_servers.py "$2"
        ;;
    
    benchmark)
        if [ -z "$2" ]; then
            echo "Error: Please provide a directory of PDFs"
            echo "Usage: ./scripts/benchmark.sh benchmark <path_to_pdfs>"
            exit 1
        fi
        echo "Running benchmark on $2..."
        python examples/benchmark_servers.py "$2" \
            --python-url http://localhost:7670 \
            --rust-url http://localhost:7671 \
            --dpi 150 \
            --concurrent 1 \
            --output benchmark_results.json
        ;;
    
    health)
        echo "Checking server health..."
        echo ""
        echo "Python server:"
        curl -s http://localhost:7670/ | python -m json.tool || echo "✗ Not responding"
        echo ""
        echo "Rust server:"
        curl -s http://localhost:7671/ | python -m json.tool || echo "✗ Not responding"
        ;;
    
    help|*)
        echo "Usage: ./scripts/benchmark.sh <command> [args]"
        echo ""
        echo "Commands:"
        echo "  build              Build the Docker image"
        echo "  up                 Start both servers with docker-compose"
        echo "  down               Stop the servers"
        echo "  logs               Show server logs (follow mode)"
        echo "  health             Check if servers are running"
        echo "  test <pdf>         Test both servers with a single PDF"
        echo "  benchmark <dir>    Run full benchmark on directory of PDFs"
        echo ""
        echo "Example workflow:"
        echo "  ./scripts/benchmark.sh build"
        echo "  ./scripts/benchmark.sh up"
        echo "  ./scripts/benchmark.sh health"
        echo "  ./scripts/benchmark.sh test test.pdf"
        echo "  ./scripts/benchmark.sh benchmark test_pdfs/"
        echo "  ./scripts/benchmark.sh down"
        ;;
esac
