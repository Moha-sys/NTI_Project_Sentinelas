import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Chaos-Grounded Autonomous AIOps Platform"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    RUNBOOKS_DIR: Path = BASE_DIR / "runbooks"
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "data" / "chroma"

    # Telemetry backends
    PROMETHEUS_URL: str = os.getenv("PROMETHEUS_URL", "http://prometheus-k8s.monitoring.svc.cluster.local:9090")
    LOKI_URL: str = os.getenv("LOKI_URL", "http://loki-gateway.monitoring.svc.cluster.local:3100")
    TEMPO_URL: str = os.getenv("TEMPO_URL", "http://tempo.monitoring.svc.cluster.local:3200")
    
    # Caching & Database
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    POSTGRES_URL: str = os.getenv("POSTGRES_URL", "postgresql://aiops:aiops_password@localhost:5432/aiops_ground_truth")

    # Local LLM (Ollama) for RCA narrative synthesis
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    OLLAMA_TIMEOUT_SECONDS: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))

    # Multi-Tenant SLA response limits (in seconds)
    SLA_CLIENT_A_SECONDS: int = 300   # 5 minutes
    SLA_CLIENT_B_SECONDS: int = 900   # 15 minutes
    SLA_CLIENT_C_SECONDS: int = 1800  # 30 minutes

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
