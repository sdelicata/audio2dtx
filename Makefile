build:
	docker build -t audio2dtx .

run: build
	docker run --rm -v (pwd)/input:/app/input -v (pwd)/output:/app/output audio2dtx

