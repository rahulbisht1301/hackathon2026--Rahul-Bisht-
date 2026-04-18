.PHONY: run stop logs test lint clean setup run-local

setup:
	cp -n .env.example .env || true
	docker compose build

run:
	docker compose up --abort-on-container-exit

run-local:
	PYTHONPATH=src poetry run python -m agent.main

stop:
	docker compose down

logs:
	docker compose logs -f agent

test:
	poetry run pytest tests/ -v --asyncio-mode=auto

lint:
	poetry run ruff check src/ tests/

clean:
	docker compose down -v
	rm -f audit_log.json
	rm -rf chroma_db/

