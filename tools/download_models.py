"""Скачивание моделей MediaPipe (вызывается при установке)."""
from __future__ import annotations

import os
import sys

# чтобы работал запуск из любой папки: tools/download_models.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mocap_studio.core.detect import ensure_models, model_dir  # noqa: E402


def main() -> int:
    print("Каталог моделей:", model_dir())
    pose, hands = ensure_models(progress=lambda m: print(" ", m))
    print("Поза:", os.path.basename(pose))
    print("Кисти:", os.path.basename(hands))
    print("Готово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
