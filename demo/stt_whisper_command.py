#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.command_parser import normalize_command_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a short command and transcribe it with a Whisper backend.")
    parser.add_argument(
        "--backend",
        choices=("auto", "whisper-trt", "whisper.cpp", "whisper-cpp", "faster-whisper", "openai-whisper"),
        default="auto",
    )
    parser.add_argument("--model", default="base.en")
    parser.add_argument("--audio", type=Path, default=None, help="Existing mono 16 kHz WAV to transcribe. If omitted, this script records one chunk.")
    parser.add_argument("--seconds", type=float, default=3.5)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--input-device", default="")
    parser.add_argument("--recorder", choices=("auto", "sounddevice", "arecord", "parec"), default="auto")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="faster-whisper compute type. auto uses float16 on CUDA and int8 otherwise.",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--vad-filter", action="store_true")
    parser.add_argument("--whisper-trt-command", default=os.environ.get("WHISPER_TRT_COMMAND", "whisper_trt"))
    parser.add_argument(
        "--whisper-trt-args",
        default=os.environ.get("WHISPER_TRT_ARGS", "--model {model} --audio {audio} --language {language}"),
        help="Argument template appended to the whisper_trt command.",
    )
    parser.add_argument("--whisper-cpp-command", default=os.environ.get("WHISPER_CPP_COMMAND", "whisper-cli"))
    parser.add_argument(
        "--whisper-cpp-args",
        default=os.environ.get("WHISPER_CPP_ARGS", "-m {model} -f {audio} -l {language} -nt -np"),
        help="Argument template appended to the whisper.cpp command.",
    )
    return parser.parse_args()


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def normalize_backend_name(value: str) -> str:
    return "whisper.cpp" if value == "whisper-cpp" else value


def command_available(command: str) -> bool:
    if shutil.which(command):
        return True
    path = Path(command).expanduser()
    return path.exists() and os.access(path, os.X_OK)


def backend_available(
    backend: str,
    whisper_trt_command: str | None = None,
    whisper_cpp_command: str | None = None,
) -> bool:
    backend = normalize_backend_name(backend)
    if backend == "whisper-trt":
        return command_available(whisper_trt_command or os.environ.get("WHISPER_TRT_COMMAND", "whisper_trt")) or package_available("whisper_trt")
    if backend == "whisper.cpp":
        command = whisper_cpp_command or os.environ.get("WHISPER_CPP_COMMAND", "whisper-cli")
        return command_available(command) or command_available("main")
    if backend == "faster-whisper":
        return package_available("faster_whisper")
    if backend == "openai-whisper":
        return package_available("whisper")
    if backend == "auto":
        return any_backend_available()
    return False


def any_backend_available() -> bool:
    return any(backend_available(name) for name in ("whisper-trt", "whisper.cpp", "faster-whisper", "openai-whisper"))


def ctranslate2_cuda_available() -> bool:
    if not package_available("ctranslate2"):
        return False
    try:
        import ctranslate2

        return bool(ctranslate2.get_cuda_device_count() > 0)
    except Exception:
        return False


def choose_backend(args: argparse.Namespace) -> str:
    requested = normalize_backend_name(args.backend)
    if requested != "auto":
        if not backend_available(requested, args.whisper_trt_command, args.whisper_cpp_command):
            raise RuntimeError(f"Requested Whisper backend is not available: {requested}")
        return requested
    if backend_available("whisper-trt", args.whisper_trt_command, args.whisper_cpp_command):
        return "whisper-trt"
    if backend_available("whisper.cpp", args.whisper_trt_command, args.whisper_cpp_command):
        return "whisper.cpp"
    if package_available("faster_whisper"):
        return "faster-whisper"
    if package_available("whisper"):
        return "openai-whisper"
    raise RuntimeError("Install whisper_trt, whisper.cpp, faster-whisper, or openai-whisper to use this STT command.")


def resolve_device(requested: str, backend: str) -> str:
    if requested != "auto":
        return requested
    if backend == "faster-whisper":
        return "cuda" if ctranslate2_cuda_available() else "cpu"
    if backend == "openai-whisper" and package_available("torch"):
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if backend == "whisper-trt":
        return "cuda"
    return "cpu"


def resolve_compute_type(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    return "float16" if device == "cuda" else "int8"


def command_prefix(command: str, module_name: str | None = None) -> list[str]:
    if shutil.which(command):
        return [command]
    path = Path(command).expanduser()
    if path.exists() and os.access(path, os.X_OK):
        return [str(path.resolve())]
    if module_name and package_available(module_name):
        return [sys.executable, "-m", module_name]
    return [command]


def sounddevice_usable() -> bool:
    if not package_available("sounddevice") or not package_available("numpy"):
        return False
    try:
        import sounddevice  # noqa: F401
        import numpy  # noqa: F401
    except Exception:
        return False
    return True


def sounddevice_id(value: str) -> int | str | None:
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.lstrip("-").isdigit():
        return int(stripped)
    return stripped


def make_temp_wav_path() -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="dishcovery_whisper_stt_", suffix=".wav", delete=False)
    handle.close()
    return Path(handle.name)


def write_wav(samples: "Any", sample_rate: int) -> Path:
    import numpy as np

    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    wav_path = make_temp_wav_path()
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return wav_path


def record_with_sounddevice(seconds: float, sample_rate: int, input_device: str) -> Path:
    import numpy as np
    import sounddevice as sd

    sample_count = int(max(0.1, seconds) * sample_rate)
    audio = sd.rec(
        sample_count,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=sounddevice_id(input_device),
    )
    sd.wait()
    samples = np.asarray(audio).reshape(-1)
    return write_wav(samples, sample_rate)


def run_recording_process(argv: list[str], stdout: Any = subprocess.PIPE) -> tuple[int, str]:
    process = subprocess.run(argv, stdout=stdout, stderr=subprocess.PIPE, text=True)
    return int(process.returncode or 0), process.stderr or ""


def record_with_arecord(seconds: float, sample_rate: int, input_device: str) -> Path:
    wav_path = make_temp_wav_path()
    argv = [
        "arecord",
        "-q",
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate),
        "-c",
        "1",
        "-d",
        str(max(1, int(math.ceil(seconds)))),
    ]
    if input_device.strip():
        argv.extend(["-D", input_device.strip()])
    argv.append(str(wav_path))
    rc, stderr = run_recording_process(argv)
    if rc != 0:
        try:
            wav_path.unlink()
        except OSError:
            pass
        raise RuntimeError(f"arecord failed with exit code {rc}: {stderr.strip()}")
    return wav_path


def record_with_parec(seconds: float, sample_rate: int, input_device: str) -> Path:
    wav_path = make_temp_wav_path()
    with wav_path.open("wb") as handle:
        argv = [
            "timeout",
            str(max(1, int(math.ceil(seconds)))),
            "parec",
            "--record",
            "--file-format=wav",
            "--format=s16le",
            "--rate",
            str(sample_rate),
            "--channels",
            "1",
            "--device",
            input_device.strip() or "@DEFAULT_SOURCE@",
        ]
        rc, stderr = run_recording_process(argv, stdout=handle)
    if rc not in {0, 124}:
        try:
            wav_path.unlink()
        except OSError:
            pass
        raise RuntimeError(f"parec failed with exit code {rc}: {stderr.strip()}")
    return wav_path


def record_audio(args: argparse.Namespace) -> Path:
    failures: list[str] = []
    if args.recorder in {"auto", "sounddevice"} and sounddevice_usable():
        try:
            return record_with_sounddevice(args.seconds, args.sample_rate, args.input_device)
        except Exception as exc:
            failures.append(f"sounddevice: {type(exc).__name__}: {exc}")
            if args.recorder == "sounddevice":
                raise
    elif args.recorder in {"auto", "sounddevice"}:
        failures.append("sounddevice: unavailable")
    if args.recorder in {"auto", "arecord"} and shutil.which("arecord") is not None:
        try:
            return record_with_arecord(args.seconds, args.sample_rate, args.input_device)
        except Exception as exc:
            failures.append(f"arecord: {type(exc).__name__}: {exc}")
            if args.recorder == "arecord" or shutil.which("parec") is None:
                raise
    elif args.recorder in {"auto", "arecord"}:
        failures.append("arecord: executable not found")
    if args.recorder in {"auto", "parec"} and shutil.which("parec") is not None:
        try:
            return record_with_parec(args.seconds, args.sample_rate, args.input_device)
        except Exception as exc:
            failures.append(f"parec: {type(exc).__name__}: {exc}")
            raise RuntimeError("Could not record audio. " + " | ".join(failures)) from exc
    if args.recorder in {"auto", "parec"}:
        failures.append("parec: executable not found")
    raise RuntimeError("Could not record audio. " + " | ".join(failures))


def external_mapping(wav_path: Path, args: argparse.Namespace, device: str) -> dict[str, str]:
    language = args.language or "en"
    return {
        "audio": str(wav_path),
        "audio_path": str(wav_path),
        "wav": str(wav_path),
        "model": args.model,
        "sample_rate": str(args.sample_rate),
        "seconds": str(args.seconds),
        "language": language,
        "device": device,
    }


def extract_transcript(stdout: str) -> str:
    candidates: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith(("whisper_", "main:", "ggml", "system_info", "load_", "sampling", "error:")):
            continue
        line = re.sub(r"^\[[^\]]+\]\s*", "", line)
        line = re.sub(r"^\d{2}:\d{2}:\d{2}(?:[,.]\d+)?\s*-->\s*\d{2}:\d{2}:\d{2}(?:[,.]\d+)?\s*", "", line)
        line = line.strip()
        if line:
            candidates.append(line)
    return normalize_command_text(" ".join(candidates[-3:]) if candidates else "")


def run_external_backend(command: str, arg_template: str, mapping: dict[str, str], module_name: str | None = None) -> str:
    argv = command_prefix(command, module_name) + shlex.split(arg_template.format(**mapping))
    completed = subprocess.run(argv, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip() or "no stderr output"
        raise RuntimeError(f"{argv[0]} failed with exit code {completed.returncode}: {message}")
    text = extract_transcript(completed.stdout)
    if not text and completed.stderr:
        text = extract_transcript(completed.stderr)
    return text


def transcribe_with_whisper_trt(wav_path: Path, args: argparse.Namespace, device: str) -> str:
    return run_external_backend(
        args.whisper_trt_command,
        args.whisper_trt_args,
        external_mapping(wav_path, args, device),
        module_name="whisper_trt",
    )


def transcribe_with_whisper_cpp(wav_path: Path, args: argparse.Namespace, device: str) -> str:
    command = args.whisper_cpp_command
    if not command_available(command) and command_available("main"):
        command = "main"
    return run_external_backend(
        command,
        args.whisper_cpp_args,
        external_mapping(wav_path, args, device),
    )


def transcribe_with_faster_whisper(wav_path: Path, args: argparse.Namespace, device: str) -> str:
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device=device, compute_type=resolve_compute_type(args.compute_type, device))
    segments, _info = model.transcribe(
        str(wav_path),
        beam_size=args.beam_size,
        language=args.language or None,
        condition_on_previous_text=False,
        vad_filter=args.vad_filter,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def transcribe_with_openai_whisper(wav_path: Path, args: argparse.Namespace, device: str) -> str:
    import whisper

    model_name = "turbo" if args.model == "large-v3" else args.model
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(
        str(wav_path),
        language=args.language or None,
        fp16=device == "cuda",
        condition_on_previous_text=False,
        temperature=0.0,
    )
    return str(result.get("text") or "").strip()


def main() -> None:
    args = parse_args()
    wav_path = args.audio.expanduser() if args.audio is not None else record_audio(args)
    created_audio = args.audio is None
    try:
        if not wav_path.exists():
            raise RuntimeError(f"Audio file does not exist: {wav_path}")
        backend = choose_backend(args)
        device = resolve_device(args.device, backend)
        if backend == "whisper-trt":
            text = transcribe_with_whisper_trt(wav_path, args, device)
        elif backend == "whisper.cpp":
            text = transcribe_with_whisper_cpp(wav_path, args, device)
        elif backend == "faster-whisper":
            text = transcribe_with_faster_whisper(wav_path, args, device)
        else:
            text = transcribe_with_openai_whisper(wav_path, args, device)
        print(normalize_command_text(text), flush=True)
    finally:
        if created_audio:
            try:
                wav_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STT error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
