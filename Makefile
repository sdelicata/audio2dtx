# Audio2DTX - Infrastructure Commands
# Simplified Makefile for docker-compose only

# Build all services
build:
	docker-compose build

# Run with arguments (batch mode)
run:
	docker-compose run --rm audio2dtx-main $(ARGS)

# Run in interactive mode
run-interactive:
	docker-compose run --rm audio2dtx-main

# Run tests (batch mode with default song)
test:
	docker-compose run --rm audio2dtx-main song.mp3 --batch

# Start all services in background
start:
	docker-compose up -d

# Stop all services
stop:
	docker-compose down

# View logs from all services
logs:
	docker-compose logs -f

# View logs from main service only
logs-main:
	docker-compose logs -f audio2dtx-main

# View logs from magenta service only
logs-magenta:
	docker-compose logs -f magenta-service

# Health check for magenta service
health:
	docker-compose exec magenta-service curl -f http://localhost:5000/health

# Clean up containers and images
clean:
	docker-compose down --rmi all --volumes --remove-orphans
	rm -rf output/*

# Help - show available commands
help:
	@echo "Available commands:"
	@echo "  build          - Build all services"
	@echo "  run ARGS=...   - Run with custom arguments"
	@echo "  run-interactive - Run in interactive mode"
	@echo "  test           - Run tests with default song"
	@echo "  start          - Start services in background"
	@echo "  stop           - Stop all services"
	@echo "  logs           - View logs (all services)"
	@echo "  logs-main      - View logs (main service only)"
	@echo "  logs-magenta   - View logs (magenta service only)"
	@echo "  health         - Check magenta service health"
	@echo "  clean          - Clean up everything"
	@echo "  help           - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  make run ARGS='song.mp3 --use-rock-ultimate --batch'"
	@echo "  make run ARGS='song.mp3 --use-ensemble --title MyChart'"

.PHONY: build run run-interactive test start stop logs logs-main logs-magenta health clean help