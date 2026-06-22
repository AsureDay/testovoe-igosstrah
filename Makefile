.PHONY: build up logs

build:
	docker-compose -f deploy/docker-compose.yml build

up:
	docker-compose -f deploy/docker-compose.yml up -d

logs:
	docker-compose -f deploy/docker-compose.yml logs -f
