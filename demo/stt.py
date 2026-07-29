from __future__ import annotations

import importlib.util
import math
import os
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .command_parser import normalize_command_text


DEFAULT_WHISPER_MODEL = "base.en"
DEFAULT_MOONSHINE_MODEL = "moonshine/tiny"


@dataclass(frozen=True)
class SpeechGate:
    has_speech: bool
    rms: float
    peak: float
    speech_seconds: float
    start_sample: int
    end_sample: int


class SpeechToText:
    def __init__(
        self,
        backend: str = "whisper",
        model: str | Path | None = None,
        command_template: str = "",
        listen_seconds: float = 2.0,
        sample_rate: int = 16_000,
        input_device: str = "",
        recorder: str = "auto",
        device: str = "auto",
        vad_enabled: bool = True,
        vad_rms_threshold: float = 0.008,
        vad_peak_threshold: float = 0.03,
        min_speech_seconds: float = 0.18,
        recording_started_callback: Any | None = None,
    ) -> None:
        self.backend = (backend or "whisper").strip().lower()
        self.model = str(model).strip() if model is not None else ""
        self.command_template = command_template.strip()
        self.listen_seconds = listen_seconds
        self.sample_rate = sample_rate
        self.input_device = input_device.strip()
        self.recorder = (recorder or "auto").strip().lower()
        self.device = (device or "auto").strip()
        self.vad_enabled = vad_enabled
        self.vad_rms_threshold = vad_rms_threshold
        self.vad_peak_threshold = vad_peak_threshold
        self.min_speech_seconds = min_speech_seconds
        self.recording_started_callback = recording_started_callback
        self._model = None
        self._tokenizer = None
        self._moonshine_module: Any | None = None
        self._whisper_model: Any | None = None
        self._whisper_device: str | None = None
        self._whisper_compute_type: str | None = None
        self._recording_process: subprocess.Popen[Any] | None = None
        self._recording_lock = threading.Lock()
        self._cancel_recording = threading.Event()

    def cancel_current_recording(self) -> None:
        self._cancel_recording.set()
        with self._recording_lock:
            process = self._recording_process
        if process is not None and process.poll() is None:
            process.terminate()

    def is_available(self) -> bool:
        if self._uses_legacy_command():
            return self._command_executable_available(self._resolved_command_template())
        if not self._selected_recorder_available() or not self._numpy_is_usable():
            return False
        if self._uses_custom_audio_command():
            return self._command_executable_available(self._resolved_command_template())
        if self.backend == "whisper":
            return self._whisper_backend_available()
        if self.backend == "moonshine":
            return bool(self._find_moonshine_module_name())
        if self.backend == "auto":
            return self._whisper_backend_available() or bool(self._find_moonshine_module_name())
        return False

    def availability_message(self) -> str:
        command_message: str | None = None
        if self._uses_legacy_command() or self._uses_custom_audio_command():
            template = self._resolved_command_template()
            first = shlex.split(template)[0] if template else ""
            resolved = shutil.which(first) if first else None
            command_message = f"command={resolved or first or 'not configured'}"
            if self._uses_legacy_command():
                return command_message
        if not self._selected_recorder_available():
            return self._recorder_availability_message()
        if not self._numpy_is_usable():
            return "numpy is required for local recording and silence gating"
        if command_message is not None:
            return command_message
        if self.backend == "whisper":
            return "Whisper wrapper ready" if self._whisper_backend_available() else "no Whisper backend found"
        if self.backend == "moonshine":
            return "Moonshine package ready" if self._find_moonshine_module_name() else "Moonshine package not found"
        if self.backend == "auto":
            if self._whisper_backend_available():
                return "Whisper wrapper ready"
            if self._find_moonshine_module_name():
                return "Moonshine package ready"
            return "no STT backend found"
        return f"unknown STT backend: {self.backend}"

    def backend_label(self) -> str:
        if self.command_template:
            return "command"
        if self.backend == "whisper":
            return "Whisper"
        if self.backend == "moonshine":
            return "Moonshine"
        return self.backend

    def preload(self) -> None:
        if self._uses_legacy_command() or self._uses_audio_command():
            if not self.command_template and self.backend == "whisper":
                self._preload_whisper_model()
            return
        if self.backend == "moonshine":
            self._preload_moonshine_model()
        elif self.backend == "whisper":
            self._preload_whisper_model()

    def listen_once(self) -> str:
        if self._uses_legacy_command():
            return self._listen_with_legacy_command()
        samples = self._record_samples()
        if self._cancel_recording.is_set():
            return ""

        gate = self._gate_samples(samples)
        if not gate.has_speech:
            return ""
        trimmed = samples[gate.start_sample : gate.end_sample]
        if trimmed.size == 0:
            return ""

        if self._uses_custom_audio_command():
            wav_path = self._write_temp_wav(trimmed)
            try:
                return self._listen_with_audio_command(wav_path)
            finally:
                self._unlink_quietly(wav_path)
        if self.backend == "whisper":
            wav_path = self._write_temp_wav(trimmed)
            try:
                return self._transcribe_whisper_wav(wav_path)
            finally:
                self._unlink_quietly(wav_path)
        if self.backend == "moonshine":
            return normalize_command_text(self._transcribe_moonshine_samples(trimmed))
        if self.backend == "auto":
            if self._whisper_backend_available():
                wav_path = self._write_temp_wav(trimmed)
                try:
                    return self._transcribe_whisper_wav(wav_path)
                finally:
                    self._unlink_quietly(wav_path)
            return normalize_command_text(self._transcribe_moonshine_samples(trimmed))
        raise RuntimeError(f"Unsupported STT backend: {self.backend}")

    def _resolved_model(self, active_backend: str | None = None) -> str:
        if self.model:
            return self.model
        backend = active_backend or self.backend
        if backend == "moonshine":
            return DEFAULT_MOONSHINE_MODEL
        return DEFAULT_WHISPER_MODEL

    def _resolved_command_template(self) -> str:
        if self.command_template:
            return self.command_template
        if self.backend in {"whisper", "auto"}:
            wrapper = Path(__file__).with_name("stt_whisper_command.py")
            return (
                f"{shlex.quote(sys.executable)} {shlex.quote(str(wrapper))} "
                "--audio {audio} --backend auto --model {model} --sample-rate {sample_rate} "
                "--device {device} --language en"
            )
        return ""

    def _command_mapping(self, wav_path: Path | None = None) -> dict[str, str]:
        audio = str(wav_path or "")
        return {
            "audio": audio,
            "audio_path": audio,
            "wav": audio,
            "model": self._resolved_model(),
            "seconds": str(self.listen_seconds),
            "sample_rate": str(self.sample_rate),
            "device": self.device or "auto",
            "input_device": self.input_device,
            "backend": self.backend,
        }

    @staticmethod
    def _template_uses_audio(template: str) -> bool:
        return any(token in template for token in ("{audio}", "{audio_path}", "{wav}"))

    def _uses_audio_command(self) -> bool:
        template = self._resolved_command_template()
        return bool(template and self._template_uses_audio(template))

    def _uses_custom_audio_command(self) -> bool:
        return bool(self.command_template and self._template_uses_audio(self.command_template))

    def _uses_legacy_command(self) -> bool:
        return bool(self.command_template and not self._template_uses_audio(self.command_template))

    @staticmethod
    def _command_executable_available(template: str) -> bool:
        if not template:
            return False
        try:
            first = shlex.split(template)[0]
        except ValueError:
            return False
        if shutil.which(first):
            return True
        path = Path(first).expanduser()
        return path.exists() and os.access(path, os.X_OK)

    def _listen_with_legacy_command(self) -> str:
        template = self._resolved_command_template()
        argv = self._format_command(template, self._command_mapping())
        completed = subprocess.run(argv, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            if not message:
                message = "no stderr output"
            raise RuntimeError(f"STT command failed with exit code {completed.returncode}: {message}")
        return self._extract_transcript(completed.stdout)

    def _listen_with_audio_command(self, wav_path: Path) -> str:
        template = self._resolved_command_template()
        argv = self._format_command(template, self._command_mapping(wav_path))
        completed = subprocess.run(argv, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            if not message:
                message = "no stderr output"
            raise RuntimeError(f"STT command failed with exit code {completed.returncode}: {message}")
        return self._extract_transcript(completed.stdout)

    @staticmethod
    def _format_command(template: str, mapping: dict[str, str]) -> list[str]:
        argv = shlex.split(template.format(**mapping))
        if argv:
            path = Path(argv[0]).expanduser()
            if path.exists() and os.access(path, os.X_OK):
                argv[0] = str(path.resolve())
        return argv

    @staticmethod
    def _extract_transcript(stdout: str) -> str:
        try:
            from .stt_whisper_command import extract_transcript

            text = extract_transcript(stdout)
            if text:
                return text
        except Exception:
            pass
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        return normalize_command_text(lines[-1] if lines else "")

    def _record_samples(self) -> "Any":
        if self.recorder == "sounddevice":
            if not self._sounddevice_is_usable():
                raise RuntimeError("sounddevice recorder requested, but PortAudio/sounddevice is not available.")
            return self._record_with_sounddevice()
        if self.recorder == "arecord":
            if shutil.which("arecord") is None:
                raise RuntimeError("arecord recorder requested, but arecord is not installed.")
            return self._record_with_arecord()
        if self.recorder == "parec":
            if shutil.which("parec") is None:
                raise RuntimeError("parec recorder requested, but parec is not installed.")
            return self._record_with_parec()
        if self.recorder != "auto":
            raise RuntimeError(f"Unsupported recorder backend: {self.recorder}")

        if self._sounddevice_is_usable():
            try:
                return self._record_with_sounddevice()
            except Exception:
                if shutil.which("arecord") is None and shutil.which("parec") is None:
                    raise
        if shutil.which("arecord") is not None:
            try:
                return self._record_with_arecord()
            except Exception:
                if shutil.which("parec") is None:
                    raise
        if shutil.which("parec") is not None:
            return self._record_with_parec()
        raise RuntimeError("No audio recorder is available. Install PortAudio for sounddevice or install ALSA arecord.")

    def _record_with_sounddevice(self) -> "Any":
        import numpy as np
        import sounddevice as sd

        self._cancel_recording.clear()
        sample_count = int(max(0.1, self.listen_seconds) * self.sample_rate)
        audio = sd.rec(
            sample_count,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self._sounddevice_id(self.input_device),
        )
        self._notify_recording_started()
        sd.wait()
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def _record_with_arecord(self) -> "Any":
        wav_path = self._make_temp_wav_path()
        try:
            argv = [
                "arecord",
                "-q",
                "-f",
                "S16_LE",
                "-r",
                str(self.sample_rate),
                "-c",
                "1",
                "-d",
                str(max(1, int(math.ceil(self.listen_seconds)))),
            ]
            if self.input_device:
                argv.extend(["-D", self.input_device])
            argv.append(str(wav_path))
            rc, stderr = self._run_recording_process(argv)
            if self._cancel_recording.is_set():
                return self._empty_samples()
            if rc != 0:
                raise RuntimeError(f"arecord failed with exit code {rc}: {stderr.strip()}")
            return self._read_temp_wav(wav_path)
        finally:
            self._unlink_quietly(wav_path)

    def _record_with_parec(self) -> "Any":
        wav_path = self._make_temp_wav_path()
        try:
            with wav_path.open("wb") as handle:
                rc, stderr = self._run_recording_process(
                    [
                        "timeout",
                        str(max(1, int(math.ceil(self.listen_seconds)))),
                        "parec",
                        "--record",
                        "--file-format=wav",
                        "--format=s16le",
                        "--rate",
                        str(self.sample_rate),
                        "--channels",
                        "1",
                        "--device",
                        self.input_device or "@DEFAULT_SOURCE@",
                    ],
                    stdout=handle,
                )
            if self._cancel_recording.is_set():
                return self._empty_samples()
            if rc not in {0, 124}:
                raise RuntimeError(f"parec failed with exit code {rc}: {stderr.strip()}")
            return self._read_temp_wav(wav_path)
        finally:
            self._unlink_quietly(wav_path)

    def _run_recording_process(self, argv: list[str], stdout: Any = subprocess.PIPE) -> tuple[int, str]:
        self._cancel_recording.clear()
        process = subprocess.Popen(
            argv,
            stdout=stdout,
            stderr=subprocess.PIPE,
            text=True,
        )
        with self._recording_lock:
            self._recording_process = process
        self._notify_recording_started()
        try:
            _, stderr = process.communicate()
            return int(process.returncode or 0), stderr or ""
        finally:
            with self._recording_lock:
                if self._recording_process is process:
                    self._recording_process = None

    def _gate_samples(self, samples: "Any") -> SpeechGate:
        import numpy as np

        if samples.size == 0:
            return SpeechGate(False, 0.0, 0.0, 0.0, 0, 0)
        peak = float(np.max(np.abs(samples)))
        sample_values = samples.astype(np.float32, copy=False)
        rms = float(np.sqrt(np.mean(np.square(sample_values))))
        if not self.vad_enabled:
            return SpeechGate(True, rms, peak, len(samples) / float(self.sample_rate), 0, len(samples))

        frame_size = max(1, int(0.03 * self.sample_rate))
        hop_size = max(1, int(0.01 * self.sample_rate))
        if len(samples) < frame_size:
            has_speech = peak >= self.vad_peak_threshold or rms >= self.vad_rms_threshold
            seconds = len(samples) / float(self.sample_rate) if has_speech else 0.0
            return SpeechGate(has_speech, rms, peak, seconds, 0, len(samples) if has_speech else 0)

        frame_rms: list[float] = []
        starts: list[int] = []
        for start in range(0, len(samples) - frame_size + 1, hop_size):
            frame = samples[start : start + frame_size]
            frame_values = frame.astype(np.float32, copy=False)
            frame_rms.append(float(np.sqrt(np.mean(np.square(frame_values)))))
            starts.append(start)
        if not frame_rms:
            return SpeechGate(False, rms, peak, 0.0, 0, 0)

        values = np.asarray(frame_rms, dtype=np.float32)
        noise_floor = float(np.percentile(values, 20))
        threshold = max(self.vad_rms_threshold, noise_floor * 2.5)
        active = values >= threshold
        if not bool(active.any()) and peak >= self.vad_peak_threshold and rms >= self.vad_rms_threshold:
            active = values >= max(self.vad_rms_threshold * 0.6, noise_floor * 1.5)
        if not bool(active.any()):
            return SpeechGate(False, rms, peak, 0.0, 0, 0)

        first_idx = int(np.argmax(active))
        last_idx = len(active) - int(np.argmax(active[::-1])) - 1
        padding = int(0.15 * self.sample_rate)
        start_sample = max(0, starts[first_idx] - padding)
        end_sample = min(len(samples), starts[last_idx] + frame_size + padding)
        speech_seconds = max(0.0, (end_sample - start_sample) / float(self.sample_rate))
        has_speech = speech_seconds >= self.min_speech_seconds and (peak >= self.vad_peak_threshold or rms >= self.vad_rms_threshold)
        return SpeechGate(has_speech, rms, peak, speech_seconds, start_sample, end_sample if has_speech else 0)

    def _transcribe_moonshine_samples(self, samples: "Any") -> str:
        moonshine = self._load_moonshine_module()
        if hasattr(moonshine, "MoonshineOnnxModel") and hasattr(moonshine, "load_tokenizer"):
            import numpy as np

            model = self._preload_moonshine_model()
            if self._tokenizer is None:
                self._tokenizer = moonshine.load_tokenizer()
            tokens = model.generate(samples[np.newaxis, :])
            decoded = self._tokenizer.decode_batch(tokens)
            return decoded[0] if decoded else ""
        if hasattr(moonshine, "transcribe"):
            wav_path = self._write_temp_wav(samples)
            try:
                result = moonshine.transcribe(str(wav_path), self._resolved_model("moonshine"))
            finally:
                self._unlink_quietly(wav_path)
            return self._stringify_transcription(result)
        return self._transcribe_moonshine_legacy(samples)

    def _preload_moonshine_model(self) -> Any:
        if self._model is not None:
            return self._model
        moonshine = self._load_moonshine_module()
        if hasattr(moonshine, "MoonshineOnnxModel"):
            model_value = self._resolved_model("moonshine")
            candidate_kwargs = [
                {"model_name": model_value},
                {"model_path": model_value},
                {"models_dir": model_value},
                {},
            ]
            last_error: Exception | None = None
            for kwargs in candidate_kwargs:
                try:
                    self._model = moonshine.MoonshineOnnxModel(**kwargs)
                    break
                except Exception as exc:
                    last_error = exc
            if self._model is None:
                raise RuntimeError(f"Could not load Moonshine model: {last_error}") from last_error
            return self._model
        return None

    def _transcribe_moonshine_legacy(self, samples: "Any") -> str:
        import numpy as np

        from moonshine_onnx import MoonshineOnnxModel, load_tokenizer

        if self._model is None:
            model_value = self._resolved_model("moonshine")
            candidate_kwargs = [
                {"model_name": model_value},
                {"model_path": model_value},
                {"models_dir": model_value},
                {},
            ]
            last_error: Exception | None = None
            for kwargs in candidate_kwargs:
                try:
                    self._model = MoonshineOnnxModel(**kwargs)
                    break
                except Exception as exc:
                    last_error = exc
            if self._model is None:
                raise RuntimeError(f"Could not load Moonshine model: {last_error}") from last_error
            self._tokenizer = load_tokenizer()

        tokens = self._model.generate(samples[np.newaxis, :])
        decoded = self._tokenizer.decode_batch(tokens)
        return decoded[0] if decoded else ""

    def _preload_whisper_model(self) -> Any:
        if self._whisper_model is not None:
            return self._whisper_model
        if not importlib.util.find_spec("faster_whisper"):
            return None
        from faster_whisper import WhisperModel

        device = self._resolved_whisper_device()
        compute_type = self._resolved_whisper_compute_type(device)
        self._whisper_model = WhisperModel(
            self._resolved_model("whisper"),
            device=device,
            compute_type=compute_type,
        )
        self._whisper_device = device
        self._whisper_compute_type = compute_type
        return self._whisper_model

    def _transcribe_whisper_wav(self, wav_path: Path) -> str:
        model = self._preload_whisper_model()
        if model is None:
            return self._listen_with_audio_command(wav_path)
        segments, _info = model.transcribe(
            str(wav_path),
            beam_size=5,
            language="en",
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return normalize_command_text(" ".join(segment.text.strip() for segment in segments))

    def _resolved_whisper_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            from .stt_whisper_command import ctranslate2_cuda_available

            return "cuda" if ctranslate2_cuda_available() else "cpu"
        except Exception:
            return "cpu"

    @staticmethod
    def _resolved_whisper_compute_type(device: str) -> str:
        return "float16" if device == "cuda" else "int8"

    def _notify_recording_started(self) -> None:
        callback = self.recording_started_callback
        if callback is None:
            return
        callback()

    @staticmethod
    def _stringify_transcription(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, (list, tuple)):
            return " ".join(str(item) for item in result)
        return str(result)

    @staticmethod
    def _sounddevice_id(value: str) -> int | str | None:
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.lstrip("-").isdigit():
            return int(stripped)
        return stripped

    @staticmethod
    def _sounddevice_is_usable() -> bool:
        if not importlib.util.find_spec("sounddevice"):
            return False
        try:
            import sounddevice  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def _numpy_is_usable() -> bool:
        if not importlib.util.find_spec("numpy"):
            return False
        try:
            import numpy  # noqa: F401
        except Exception:
            return False
        return True

    @classmethod
    def _recorder_available(cls) -> bool:
        return cls._sounddevice_is_usable() or shutil.which("arecord") is not None or shutil.which("parec") is not None

    def _selected_recorder_available(self) -> bool:
        if self.recorder == "auto":
            return self._recorder_available()
        if self.recorder == "sounddevice":
            return self._sounddevice_is_usable()
        if self.recorder == "arecord":
            return shutil.which("arecord") is not None
        if self.recorder == "parec":
            return shutil.which("parec") is not None
        return False

    def _recorder_availability_message(self) -> str:
        if self.recorder == "sounddevice":
            return "sounddevice recorder requested, but PortAudio/sounddevice is not available"
        if self.recorder == "arecord":
            return "arecord recorder requested, but arecord is not installed"
        if self.recorder == "parec":
            return "parec recorder requested, but parec is not installed"
        if self.recorder != "auto":
            return f"unknown recorder backend: {self.recorder}"
        return "no recorder found; install sounddevice, ALSA arecord, or PulseAudio parec"

    @staticmethod
    def _whisper_backend_available() -> bool:
        try:
            from . import stt_whisper_command

            return stt_whisper_command.any_backend_available()
        except Exception:
            return False

    @staticmethod
    def _find_moonshine_module_name() -> str | None:
        for name in ("moonshine_onnx", "moonshine"):
            if importlib.util.find_spec(name):
                return name
        return None

    def _load_moonshine_module(self) -> Any:
        if self._moonshine_module is not None:
            return self._moonshine_module
        module_name = self._find_moonshine_module_name()
        if module_name is None:
            raise RuntimeError("Install useful-moonshine-onnx to enable Moonshine STT.")
        import importlib

        self._moonshine_module = importlib.import_module(module_name)
        return self._moonshine_module

    def _write_temp_wav(self, samples: "Any") -> Path:
        import numpy as np

        pcm = np.clip(samples, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        wav_path = self._make_temp_wav_path()
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm.tobytes())
        return wav_path

    @staticmethod
    def _make_temp_wav_path() -> Path:
        with tempfile.NamedTemporaryFile(prefix="dishcovery_mic_", suffix=".wav", delete=False) as handle:
            return Path(handle.name)

    def _read_temp_wav(self, wav_path: Path) -> "Any":
        import numpy as np

        with wave.open(str(wav_path), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        return samples

    @staticmethod
    def _unlink_quietly(path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            pass

    @staticmethod
    def _empty_samples() -> "Any":
        import numpy as np

        return np.asarray([], dtype=np.float32)


class MoonshineSTT(SpeechToText):
    def __init__(
        self,
        model_path: Path | None,
        command_template: str = "",
        listen_seconds: float = 2.0,
        sample_rate: int = 16_000,
        input_device: str = "",
    ) -> None:
        super().__init__(
            backend="command" if command_template else "moonshine",
            model=model_path or DEFAULT_MOONSHINE_MODEL,
            command_template=command_template,
            listen_seconds=listen_seconds,
            sample_rate=sample_rate,
            input_device=input_device,
            device="auto",
        )


class VoiceCommandThread:
    def __init__(
        self,
        stt: SpeechToText,
        pause_seconds: float = 0.15,
        listen_mode: str = "continuous",
        listen_on_start: bool = True,
    ) -> None:
        self.stt = stt
        self.pause_seconds = pause_seconds
        self.listen_mode = listen_mode
        self.listen_on_start = listen_on_start
        self.commands: queue.Queue[str] = queue.Queue()
        self.errors: queue.Queue[str] = queue.Queue()
        self.statuses: queue.Queue[tuple[str, str]] = queue.Queue()
        self._stop = threading.Event()
        self._listen_requested = threading.Event()
        self._listening = threading.Event()
        self._thread: threading.Thread | None = None
        self._previous_recording_started_callback = stt.recording_started_callback
        self.stt.recording_started_callback = self._recording_started

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive() and not self._stop.is_set():
            return
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._stop.clear()
        self._listen_requested.clear()
        self._listening.clear()
        self._thread = threading.Thread(target=self._run, name="voice-stt", daemon=True)
        self._thread.start()
        if self.listen_mode == "triggered" and self.listen_on_start:
            self._listen_requested.set()

    def stop(self) -> None:
        self._stop.set()
        self._listen_requested.set()
        self.stt.cancel_current_recording()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def request_listen(self, restart: bool = False, clear_pending: bool = False) -> None:
        if clear_pending:
            self.clear_pending(include_errors=True)
        if self._stop.is_set() or self._thread is None or not self._thread.is_alive():
            self.start()
        if restart and self.is_listening():
            self.stt.cancel_current_recording()
        self._listen_requested.set()
        self.statuses.put(("queued", "Voice recording queued"))

    def is_listening(self) -> bool:
        return self._listening.is_set()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def clear_pending(self, include_errors: bool = False) -> None:
        self._drain_queue(self.commands)
        self._drain_queue(self.statuses)
        if include_errors:
            self._drain_queue(self.errors)

    @staticmethod
    def _drain_queue(items: "queue.Queue[Any]") -> None:
        while True:
            try:
                items.get_nowait()
            except queue.Empty:
                return

    def _recording_started(self) -> None:
        callback = self._previous_recording_started_callback
        if callback is not None:
            callback()
        self.statuses.put(("recording", f"Recording speech for {self.stt.listen_seconds:.1f} seconds"))

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.listen_mode == "triggered":
                self._listen_requested.wait(timeout=0.2)
                if self._stop.is_set():
                    return
                if not self._listen_requested.is_set():
                    continue
                self._listen_requested.clear()
            try:
                self._listening.set()
                self.statuses.put(("starting", "Starting microphone"))
                if self.stt._uses_legacy_command():
                    self.statuses.put(("recording", f"Recording speech for {self.stt.listen_seconds:.1f} seconds"))
                text = self.stt.listen_once()
                self.statuses.put(("transcript", text))
                if text:
                    self.commands.put(text)
                elif self.listen_mode == "triggered":
                    self.statuses.put(("empty", "No speech recognized"))
                    time.sleep(self.pause_seconds)
            except Exception as exc:
                self.errors.put(f"{type(exc).__name__}: {exc}")
                self._stop.set()
            finally:
                self._listening.clear()
            if self.listen_mode == "continuous":
                time.sleep(self.pause_seconds)


class TextCommandThread:
    def __init__(self) -> None:
        self.commands: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="text-command-input", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text = input("> ")
            except EOFError:
                self._stop.set()
                return
            text = normalize_command_text(text)
            if text:
                self.commands.put(text)
