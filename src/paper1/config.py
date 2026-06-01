"""Configuration loader.

Reads `config/models.yaml` and the `OPENROUTER_API_KEY` env var (via .env).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.yaml"
DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "config" / "prompts"


@dataclass(frozen=True)
class AgentModelConfig:
    """Model and sampling settings for a single agent."""

    model_id: str
    price_in_per_m: float
    price_out_per_m: float
    max_tokens: int
    top_p: float
    temperatures: tuple[float, ...] | None  # only set for the Extractor (voting)
    temperature: float | None  # only set for Critic / Consolidator


@dataclass(frozen=True)
class Defaults:
    request_timeout_s: float
    max_retries: int
    retry_backoff_s: float


@dataclass(frozen=True)
class Concurrency:
    papers_in_flight: int
    per_provider_max: int


@dataclass(frozen=True)
class Config:
    """Resolved configuration for one pipeline run."""

    extractor: AgentModelConfig
    critic: AgentModelConfig
    consolidator: AgentModelConfig
    defaults: Defaults
    concurrency: Concurrency
    api_key: str
    base_url: str
    referer: str | None
    title: str | None
    prompts_dir: Path


def _agent_from_dict(d: dict) -> AgentModelConfig:
    temps = d.get("temperatures")
    temp = d.get("temperature")
    return AgentModelConfig(
        model_id=d["model_id"],
        price_in_per_m=float(d["price_in_per_m"]),
        price_out_per_m=float(d["price_out_per_m"]),
        max_tokens=int(d.get("max_tokens", 1500)),
        top_p=float(d.get("top_p", 0.95)),
        temperatures=tuple(temps) if temps is not None else None,
        temperature=float(temp) if temp is not None else None,
    )


def load_config(
    config_path: Path | None = None,
    prompts_dir: Path | None = None,
    *,
    require_api_key: bool = True,
) -> Config:
    """Load configuration from YAML + .env."""

    config_path = config_path or DEFAULT_CONFIG_PATH
    prompts_dir = prompts_dir or DEFAULT_PROMPTS_DIR

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if require_api_key and not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    raw = yaml.safe_load(config_path.read_text())

    return Config(
        extractor=_agent_from_dict(raw["extractor"]),
        critic=_agent_from_dict(raw["critic"]),
        consolidator=_agent_from_dict(raw["consolidator"]),
        defaults=Defaults(
            request_timeout_s=float(raw["defaults"]["request_timeout_s"]),
            max_retries=int(raw["defaults"]["max_retries"]),
            retry_backoff_s=float(raw["defaults"]["retry_backoff_s"]),
        ),
        concurrency=Concurrency(
            papers_in_flight=int(raw["concurrency"]["papers_in_flight"]),
            per_provider_max=int(raw["concurrency"]["per_provider_max"]),
        ),
        api_key=api_key,
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        referer=os.environ.get("OPENROUTER_REFERER"),
        title=os.environ.get("OPENROUTER_TITLE"),
        prompts_dir=prompts_dir,
    )


def load_prompt(name: str, prompts_dir: Path | None = None) -> str:
    """Load a prompt template by name (e.g., 'extractor', 'critic', 'consolidator')."""

    prompts_dir = prompts_dir or DEFAULT_PROMPTS_DIR
    path = prompts_dir / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text()
