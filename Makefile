.PHONY: help build up down logs health test benchmark clean

help:
	@echo "Slim-Gest Benchmark Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make build       - Build Docker image with both servers"
	@echo "  make up          - Start both servers with docker-compose"
	@echo "  make down        - Stop the servers"
	@echo "  make logs        - Show server logs (follow mode)"
	@echo "  make health      - Check if servers are responding"
	@echo "  make clean       - Clean up Docker resources"
	@echo ""
	@echo "Testing and benchmarking:"
	@echo "  make test PDF=<path>     - Test both servers with a single PDF"
	@echo "  make benchmark DIR=<dir> - Run benchmark on directory of PDFs"
	@echo ""
	@echo "Example workflow:"
	@echo "  make build"
	@echo "  make up"
	@echo "  make health"
	@echo "  make test PDF=test.pdf"
	@echo "  make benchmark DIR=test_pdfs/"
	@echo "  make down"

build:
	docker build -t slimgest .

up:
	docker compose up -d
	@echo ""
	@echo "Servers started:"
	@echo "  Python FastAPI: http://localhost:7670"
	@echo "  Rust Axum:      http://localhost:7671"

down:
	docker compose down

logs:
	docker compose logs -f

health:
	@echo "Checking Python server..."
	@curl -s http://localhost:7670/ | python -m json.tool || echo "Not responding"
	@echo ""
	@echo "Checking Rust server..."
	@curl -s http://localhost:7671/ | python -m json.tool || echo "Not responding"

test:
	@if [ -z "$(PDF)" ]; then \
		echo "Error: PDF variable is required"; \
		echo "Usage: make test PDF=path/to/test.pdf"; \
		exit 1; \
	fi
	python examples/test_servers.py $(PDF)

benchmark:
	@if [ -z "$(DIR)" ]; then \
		echo "Error: DIR variable is required"; \
		echo "Usage: make benchmark DIR=path/to/pdfs/"; \
		exit 1; \
	fi
	python examples/benchmark_servers.py $(DIR) \
		--python-url http://localhost:7670 \
		--rust-url http://localhost:7671 \
		--dpi 150 \
		--concurrent 1 \
		--output benchmark_results.json

clean:
	docker compose down -v
	docker rmi slimgest || true
	rm -f benchmark_results.json
