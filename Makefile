.PHONY: dev-backend dev-frontend test sync install

dev-backend:
	cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm run dev

test:
	cd backend && uv run pytest

sync:
	cd backend && uv sync

install: sync
	cd frontend && npm install
