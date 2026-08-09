"""Настройки мини-ассистента: источник данных и параметры LLM.

Порядок приоритета для параметров LLM: аргументы CLI > переменные окружения > .env > значения по умолчанию.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv не обязателен
    pass

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", ""))
    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    max_tool_iters: int = 25
    max_pages_per_call: int = 10
    max_toc_nodes: int = 60


def get_settings() -> Settings:
    return Settings()
