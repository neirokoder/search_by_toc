"""CLI-чат мини-ассистента по архиву документов.

Пример запуска:
    python cli.py                          # ключ/адрес/модель из env или .env
    python cli.py --api-key ... --model gpt-4o-mini
    python cli.py --base-url http://localhost:8000/v1 --api-key sk-...
"""

import argparse
import sys
from pathlib import Path

from archive import Archive
from assistant import MiniAssistant
from config import Settings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Мини-ассистент по архиву документов")
    p.add_argument("--data-dir", type=Path, default=None, help="Каталог с zip-архивами документов")
    p.add_argument("--api-key", default=None, help="Ключ API (приоритет над env/OPENAI_API_KEY)")
    p.add_argument("--base-url", default=None, help="Адрес OpenAI-совместимого сервера (приоритет над env)")
    p.add_argument("--model", default=None, help="Название модели (приоритет над env)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings()
    if args.data_dir is not None:
        settings.data_dir = args.data_dir
    if args.api_key is not None:
        settings.api_key = args.api_key
    if args.base_url is not None:
        settings.base_url = args.base_url
    if args.model is not None:
        settings.model = args.model

    if not settings.api_key and not settings.base_url:
        print("Не задан ключ API и адрес сервера. Укажите --api-key или переменную окружения OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    archive = Archive(settings.data_dir)
    docs = archive.documents()
    if not docs:
        print(f"В каталоге {settings.data_dir} не найдено zip-архивов документов.", file=sys.stderr)
        sys.exit(1)

    print(f"Архив: {len(docs)} документов.")
    print(archive.list_text())
    print()
    print("Задайте вопрос по документам (exit — выход):")

    assistant = MiniAssistant(archive, settings)
    while True:
        try:
            query = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in ("exit", "выход", "quit"):
            break
        print("Ассистент:", flush=True)
        try:
            answer = assistant.run(query)
        except Exception as exc:
            print(f"Ошибка при обработке запроса: {exc}")
            continue
        print(answer)


if __name__ == "__main__":
    main()
