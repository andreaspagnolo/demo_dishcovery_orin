from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def utcish_local_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class ResultHistory:
    def __init__(self, root: Path = Path("demo_runs")) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self) -> Path:
        base = self.root / utcish_local_timestamp()
        path = base
        suffix = 1
        while path.exists():
            suffix += 1
            path = Path(f"{base}_{suffix:02d}")
        path.mkdir(parents=True)
        return path
    def save(
        self,
        run_dir: Path,
        command_text: str,
        parsed_command: Any,
        image_path: Path,
        result: dict[str, Any],
        spoken_answer: str,
    ) -> dict[str, Path]:
        run_dir.mkdir(parents=True, exist_ok=True)
        saved_frame = run_dir / "captured_frame.jpg"
        if image_path.resolve() != saved_frame.resolve():
            shutil.copy2(image_path, saved_frame)

        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "voice_command_text": command_text,
            "parsed_command": parsed_command,
            "captured_frame": str(saved_frame),
            "result": result,
            "spoken_answer": spoken_answer,
        }
        metadata_path = run_dir / "interaction.json"
        metadata_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=json_default),
            encoding="utf-8",
        )
        (run_dir / "spoken_answer.txt").write_text(spoken_answer + "\n", encoding="utf-8")
        (run_dir / "voice_command.txt").write_text(command_text + "\n", encoding="utf-8")
        return {"metadata": metadata_path, "frame": saved_frame}

    def save_bookmark(
        self,
        command_text: str,
        parsed_command: Any,
        result: dict[str, Any],
        spoken_answer: str,
    ) -> Path:
        saved_dir = self.root / "saved_results"
        saved_dir.mkdir(parents=True, exist_ok=True)
        path = saved_dir / f"{utcish_local_timestamp()}.json"
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "voice_command_text": command_text,
            "parsed_command": parsed_command,
            "result": result,
            "spoken_answer": spoken_answer,
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=json_default),
            encoding="utf-8",
        )
        return path
