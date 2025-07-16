# Single container commands (legacy)
build:
	docker build -t audio2dtx .

run: build
	docker run --rm -v $(PWD)/input:/app/input -v $(PWD)/output:/app/output audio2dtx song.mp3 --batch

run-interactive: build
	docker run --rm -it -v $(PWD)/input:/app/input -v $(PWD)/output:/app/output audio2dtx song.mp3

test: build
	docker run --rm -v $(PWD)/input:/app/input -v $(PWD)/output:/app/output audio2dtx song.mp3 --batch

# Docker Compose commands (with Magenta service)
build-all:
	docker-compose build

run-magenta: build-all
	docker-compose run --rm audio2dtx-main song.mp3 --batch

run-magenta-interactive: build-all
	docker-compose run --rm audio2dtx-main song.mp3

test-magenta: build-all
	docker-compose run --rm audio2dtx-main song.mp3 --batch

# Service management
start-services:
	docker-compose up -d

stop-services:
	docker-compose down

logs-magenta:
	docker-compose logs magenta-service

# Health checks
health-check:
	docker-compose exec magenta-service curl -f http://localhost:5000/health

service-info:
	docker-compose exec magenta-service curl -f http://localhost:5000/info


# Cleanup
clean:
	docker rmi audio2dtx || true
	docker rmi audio2dtx_magenta-service || true
	docker rmi audio2dtx_audio2dtx-main || true
	rm -rf output/*

clean-all: stop-services
	docker-compose down --rmi all --volumes
	rm -rf output/*

.PHONY: build run run-interactive test build-all run-magenta run-magenta-interactive test-magenta start-services stop-services logs-magenta health-check service-info clean clean-all

