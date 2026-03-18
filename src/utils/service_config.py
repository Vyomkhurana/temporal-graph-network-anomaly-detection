"""External service configuration helpers with opt-in validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import yaml


class ServiceConfigError(Exception):
    """Raised when enabled service configuration is incomplete."""


def load_services_config(path: str = "configs/services.yaml") -> Dict:
    """Load service toggle/configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise ServiceConfigError(f"Services config not found: {path}")

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _get_missing_env_vars(required_env: List[str]) -> List[str]:
    """Return missing environment variables from required list."""
    return [var for var in required_env if not os.getenv(var)]


def validate_enabled_services(services_config: Dict) -> None:
    """Validate env vars only for services explicitly enabled in YAML."""
    checks = {
        "mlflow": ["MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME"],
        "kafka": [
            "KAFKA_BOOTSTRAP_SERVERS",
            "KAFKA_TOPIC_EVENTS",
            "KAFKA_SECURITY_PROTOCOL",
        ],
    }

    for service_name, required_env in checks.items():
        service_cfg = services_config.get(service_name, {})
        if not service_cfg.get("enabled", False):
            continue

        missing = _get_missing_env_vars(required_env)
        if missing:
            missing_joined = ", ".join(missing)
            raise ServiceConfigError(
                f"Service '{service_name}' is enabled but missing env vars: {missing_joined}"
            )
