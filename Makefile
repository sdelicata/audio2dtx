build:
	docker build -t audio2dtx .

run: build
	docker run --rm -v $(PWD)/input:/app/input -v $(PWD)/output:/app/output audio2dtx song.mp3 --batch

run-interactive: build
	docker run --rm -it -v $(PWD)/input:/app/input -v $(PWD)/output:/app/output audio2dtx song.mp3

test: build
	docker run --rm -v $(PWD)/input:/app/input -v $(PWD)/output:/app/output audio2dtx song.mp3 --batch

clean:
	docker rmi audio2dtx || true
	rm -rf output/*

.PHONY: build run run-interactive test clean

