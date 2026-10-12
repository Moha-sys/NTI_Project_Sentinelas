.PHONY: help setup test lint run dev-backend dev-dashboard k8s-deploy chaos-inject clean

PYTHON_VENV = backend/.venv/bin
PYTHON = $(PYTHON_VENV)/python
PIP = $(PYTHON_VENV)/pip
PYTEST = $(PYTHON_VENV)/pytest

help:
	@echo "AIOps Platform - Autonomous Incident Detection & Self-Healing"
	@echo "============================================================"
	@echo "Available targets:"
	@echo "  make setup          - Initialize Python venv and install dependencies"
	@echo "  make test           - Run automated backend test suite (unit & RAG)"
	@echo "  make dev-backend    - Run FastAPI backend in reload mode"
	@echo "  make dev-dashboard  - Run React dashboard development server"
	@echo "  make index-runbooks - Index DevOps runbooks into ChromaDB vector store"
	@echo "  make train-ml       - Train Isolation Forest + LightGBM RCA models"
	@echo "  make k8s-deploy     - Deploy target microservices to Kubernetes"
	@echo "  make clean          - Remove temporary files and cached data"

setup:
	python3 -m venv backend/.venv
	$(PIP) install --upgrade pip
	$(PIP) install -r backend/requirements.txt

test:
	$(PYTEST) -v backend/tests/

index-runbooks:
	$(PYTHON) -m app.rag.indexer

train-ml:
	cd backend && python -m app.ml.train

dev-backend:
	cd backend && ../$(PYTHON_VENV)/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

