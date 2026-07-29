from __future__ import annotations

import argparse
import atexit
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DEFAULT_EDGE_LLM_PATH = PROJECT_ROOT / "TensorRT-Edge-LLM"
DEFAULT_WORKSPACE_DIR = Path("/home/danilopau/tensorrt-edgellm-workspace")
DEFAULT_MODEL_NAME = "Qwen3-VL-4B-Instruct"
DEFAULT_LLM_ENGINE_PROFILE = "default"
LLM_ENGINE_PROFILE_DIRS = {
    "default": "llm",
    "max-input-1024": "llm",
    "in1024": "llm",
    "max-input-4096": "llm_in4096_kv8192",
    "in4096": "llm_in4096_kv8192",
}
_READY_PREFIX = "EDGELLM_SERVER_READY "
_RESPONSE_PREFIX = "EDGELLM_SERVER_RESPONSE "
_ERROR_PREFIX = "EDGELLM_SERVER_ERROR "
_FAILED_RESPONSE_TEXT = "TensorRT Edge LLM cannot handle this request. Fails."


@dataclass(frozen=True)
class EdgeLLMConfig:
    edge_llm_path: Path
    workspace_dir: Path
    model_name: str
    llm_engine_profile: str
    llm_engine_dir: Path
    visual_engine_dir: Path
    inference_binary: Path
    persistent_binary: Path
    plugin_path: Path
    runtime_mode: str
    timeout_sec: float
    startup_timeout_sec: float
    keep_io: bool
    dump_profile: bool
    warmup: int
    temperature: float
    top_p: float
    top_k: int
    enable_thinking: bool


def add_edgellm_args(parser: argparse.ArgumentParser, default_backend: str = "transformers") -> None:
    parser.add_argument(
        "--vlm-backend",
        choices=("edgellm", "transformers"),
        default=os.environ.get("DISHCOVERY_VLM_BACKEND", default_backend),
        help=f"VLM runtime backend. Default: {default_backend}.",
    )
    parser.add_argument("--edgellm-root", type=Path, default=Path(os.environ.get("EDGE_LLM_PATH", DEFAULT_EDGE_LLM_PATH)))
    parser.add_argument("--edgellm-workspace", type=Path, default=Path(os.environ.get("WORKSPACE_DIR", DEFAULT_WORKSPACE_DIR)))
    parser.add_argument("--edgellm-model-name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument(
        "--edgellm-llm-engine-profile",
        default=os.environ.get(
            "EDGELLM_LLM_ENGINE_PROFILE",
            os.environ.get("DISHCOVERY_EDGELLM_LLM_ENGINE_PROFILE", DEFAULT_LLM_ENGINE_PROFILE),
        ),
        help=(
            "LLM engine profile under <workspace>/<model>/engines. Known values: "
            "default/max-input-1024 -> llm, max-input-4096 -> llm_in4096_kv8192. "
            "Any other value is treated as an engine directory name."
        ),
    )
    parser.add_argument("--edgellm-llm-engine-dir", type=Path, default=None)
    parser.add_argument("--edgellm-visual-engine-dir", type=Path, default=None)
    parser.add_argument("--edgellm-binary", type=Path, default=None)
    parser.add_argument("--edgellm-persistent-binary", type=Path, default=None)
    parser.add_argument("--edgellm-plugin-path", type=Path, default=None)
    parser.add_argument(
        "--edgellm-runtime-mode",
        choices=("subprocess", "persistent"),
        default=os.environ.get("DISHCOVERY_EDGELLM_RUNTIME_MODE", "subprocess"),
        help="Use the original one-shot llm_inference process or a persistent Edge-LLM runtime process.",
    )
    parser.add_argument("--edgellm-timeout-sec", type=float, default=180.0)
    parser.add_argument("--edgellm-startup-timeout-sec", type=float, default=240.0)
    parser.add_argument("--edgellm-keep-io", action="store_true", help="Keep temporary Edge-LLM request/response JSON files.")
    parser.add_argument("--edgellm-dump-profile", action="store_true")
    parser.add_argument("--edgellm-warmup", type=int, default=0)
    parser.add_argument("--edgellm-temperature", type=float, default=0.0)
    parser.add_argument("--edgellm-top-p", type=float, default=1.0)
    parser.add_argument("--edgellm-top-k", type=int, default=1)
    parser.add_argument("--edgellm-enable-thinking", action="store_true")


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    add_edgellm_args(parser)
    return parser.parse_known_args(argv)


def attach_wrapper_args(args: argparse.Namespace, wrapper_args: argparse.Namespace) -> argparse.Namespace:
    for key, value in vars(wrapper_args).items():
        setattr(args, key, value)
    if getattr(args, "vlm_backend", "") == "edgellm":
        args.vlm_family = "qwen"
    return args


def llm_engine_dir_name(profile: str) -> str:
    return LLM_ENGINE_PROFILE_DIRS.get(profile, profile)


def config_from_args(args: argparse.Namespace) -> EdgeLLMConfig:
    edge_llm_path = Path(args.edgellm_root).expanduser().resolve()
    workspace_dir = Path(args.edgellm_workspace).expanduser().resolve()
    model_name = str(args.edgellm_model_name)
    llm_engine_profile = str(getattr(args, "edgellm_llm_engine_profile", DEFAULT_LLM_ENGINE_PROFILE)).strip()
    if not llm_engine_profile:
        llm_engine_profile = DEFAULT_LLM_ENGINE_PROFILE
    model_root = workspace_dir / model_name
    llm_engine_dir = (
        Path(args.edgellm_llm_engine_dir).expanduser().resolve()
        if args.edgellm_llm_engine_dir
        else model_root / "engines" / llm_engine_dir_name(llm_engine_profile)
    )
    visual_engine_dir = (
        Path(args.edgellm_visual_engine_dir).expanduser().resolve()
        if args.edgellm_visual_engine_dir
        else model_root / "engines" / "visual"
    )
    inference_binary = (
        Path(args.edgellm_binary).expanduser().resolve()
        if args.edgellm_binary
        else edge_llm_path / "build" / "examples" / "llm" / "llm_inference"
    )
    persistent_binary = (
        Path(args.edgellm_persistent_binary).expanduser().resolve()
        if args.edgellm_persistent_binary
        else edge_llm_path / "build" / "examples" / "llm" / "llm_persistent_server"
    )
    plugin_path = (
        Path(args.edgellm_plugin_path).expanduser().resolve()
        if args.edgellm_plugin_path
        else edge_llm_path / "build" / "libNvInfer_edgellm_plugin.so"
    )
    return EdgeLLMConfig(
        edge_llm_path=edge_llm_path,
        workspace_dir=workspace_dir,
        model_name=model_name,
        llm_engine_profile=llm_engine_profile,
        llm_engine_dir=llm_engine_dir,
        visual_engine_dir=visual_engine_dir,
        inference_binary=inference_binary,
        persistent_binary=persistent_binary,
        plugin_path=plugin_path,
        runtime_mode=str(args.edgellm_runtime_mode),
        timeout_sec=float(args.edgellm_timeout_sec),
        startup_timeout_sec=float(args.edgellm_startup_timeout_sec),
        keep_io=bool(args.edgellm_keep_io),
        dump_profile=bool(args.edgellm_dump_profile),
        warmup=max(0, int(args.edgellm_warmup)),
        temperature=float(args.edgellm_temperature),
        top_p=float(args.edgellm_top_p),
        top_k=int(args.edgellm_top_k),
        enable_thinking=bool(args.edgellm_enable_thinking),
    )


def validate_config(config: EdgeLLMConfig) -> None:
    if config.runtime_mode not in {"subprocess", "persistent"}:
        raise ValueError(f"Unsupported Edge-LLM runtime mode: {config.runtime_mode}")
    if config.runtime_mode == "persistent" and config.dump_profile:
        raise ValueError("--edgellm-dump-profile is only supported with --edgellm-runtime-mode subprocess")
    required = {
        "TensorRT Edge-LLM root": config.edge_llm_path,
        "Edge-LLM plugin": config.plugin_path,
        "LLM engine": config.llm_engine_dir / "llm.engine",
        "LLM config": config.llm_engine_dir / "config.json",
        "LLM tokenizer": config.llm_engine_dir / "tokenizer.json",
        "Visual engine": config.visual_engine_dir / "visual" / "visual.engine",
        "Visual config": config.visual_engine_dir / "visual" / "config.json",
        "Visual preprocessor config": config.visual_engine_dir / "visual" / "preprocessor_config.json",
    }
    if config.runtime_mode == "persistent":
        required["llm_persistent_server binary"] = config.persistent_binary
    else:
        required["llm_inference binary"] = config.inference_binary
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("TensorRT Edge-LLM Qwen3-VL setup is incomplete:\n" + "\n".join(missing))
    executable = config.persistent_binary if config.runtime_mode == "persistent" else config.inference_binary
    if not os.access(executable, os.X_OK):
        label = "llm_persistent_server" if config.runtime_mode == "persistent" else "llm_inference"
        raise PermissionError(f"{label} is not executable: {executable}")


def verification_summary(config: EdgeLLMConfig) -> dict[str, Any]:
    validate_config(config)
    return {
        "edge_llm_path": str(config.edge_llm_path),
        "model_name": config.model_name,
        "llm_engine_profile": config.llm_engine_profile,
        "llm_engine": str(config.llm_engine_dir / "llm.engine"),
        "visual_engine": str(config.visual_engine_dir / "visual" / "visual.engine"),
        "plugin": str(config.plugin_path),
        "inference_binary": str(config.inference_binary),
        "persistent_binary": str(config.persistent_binary),
        "runtime_mode": config.runtime_mode,
    }


class EdgeLLMQwenScorer:
    family = "qwen"
    moe_tracker = None

    def __init__(self, config: EdgeLLMConfig, max_new_tokens: int) -> None:
        validate_config(config)
        self.config = config
        self.max_new_tokens = int(max_new_tokens)
        self.model_id = config.model_name
        self.last_io_dir: Path | None = None
        self._server: subprocess.Popen[str] | None = None
        self._server_logs: deque[str] = deque(maxlen=200)
        self._server_queue: queue.Queue[str] = queue.Queue()
        self._server_reader: threading.Thread | None = None
        self._server_lock = threading.Lock()
        if self.config.runtime_mode == "persistent":
            atexit.register(self.close)
            self.start()

    @classmethod
    def from_args(cls, args: argparse.Namespace, max_new_tokens: int) -> "EdgeLLMQwenScorer":
        return cls(config_from_args(args), max_new_tokens=max_new_tokens)

    def start(self) -> None:
        if self.config.runtime_mode != "persistent":
            return
        with self._server_lock:
            self._start_server_locked()

    def score(
        self,
        image_path: Path,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> str:
        effective_max_new_tokens = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        if effective_max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive for Edge-LLM generation")
        image_path = Path(image_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image does not exist: {image_path}")

        image_item: dict[str, Any] = {"type": "image", "image": str(image_path)}
        if min_pixels is not None and int(min_pixels) > 0:
            image_item["min_pixels"] = int(min_pixels)
        if max_pixels is not None and int(max_pixels) > 0:
            image_item["max_pixels"] = int(max_pixels)
        messages = [
            {
                "role": "user",
                "content": [
                    image_item,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        return self.score_messages(messages, max_new_tokens=effective_max_new_tokens)

    def score_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        effective_max_new_tokens = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        if effective_max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive for Edge-LLM generation")

        if self.config.keep_io:
            io_dir = ROOT / "demo_runs" / "edgellm_io" / time.strftime("%Y%m%d_%H%M%S")
            io_dir.mkdir(parents=True, exist_ok=True)
            return self._score_messages_in_dir(messages, effective_max_new_tokens, io_dir)

        with tempfile.TemporaryDirectory(prefix="dishcovery_edgellm_") as tmp:
            return self._score_messages_in_dir(messages, effective_max_new_tokens, Path(tmp))

    def score_token_logits_messages(
        self,
        messages: list[dict[str, Any]],
        token_ids: list[int],
        *,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        if self.config.runtime_mode != "persistent":
            raise RuntimeError("Edge-LLM token-logit scoring requires --edgellm-runtime-mode persistent.")
        if not token_ids:
            raise ValueError("token_ids must contain at least one token ID")
        effective_max_new_tokens = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        if effective_max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive for Edge-LLM token-logit scoring")

        if self.config.keep_io:
            io_dir = ROOT / "demo_runs" / "edgellm_io" / time.strftime("%Y%m%d_%H%M%S")
            io_dir.mkdir(parents=True, exist_ok=True)
            input_json = self._write_request_json_messages(messages, effective_max_new_tokens, io_dir)
            return self._request_with_persistent(input_json, effective_max_new_tokens, token_ids=token_ids)

        with tempfile.TemporaryDirectory(prefix="dishcovery_edgellm_") as tmp:
            input_json = self._write_request_json_messages(messages, effective_max_new_tokens, Path(tmp))
            return self._request_with_persistent(input_json, effective_max_new_tokens, token_ids=token_ids)

    def _edge_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["EDGE_LLM_PATH"] = str(self.config.edge_llm_path)
        env["WORKSPACE_DIR"] = str(self.config.workspace_dir)
        env["MODEL_NAME"] = self.config.model_name
        env["EDGELLM_PLUGIN_PATH"] = str(self.config.plugin_path)
        build_dir = str(self.config.edge_llm_path / "build")
        env["LD_LIBRARY_PATH"] = build_dir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        return env

    def _write_request_json_messages(self, messages: list[dict[str, Any]], max_new_tokens: int, io_dir: Path) -> Path:
        self.last_io_dir = io_dir
        input_json = io_dir / "request.json"
        payload = {
            "batch_size": 1,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "max_generate_length": max_new_tokens,
            "apply_chat_template": True,
            "add_generation_prompt": True,
            "enable_thinking": self.config.enable_thinking,
            "requests": [{"messages": messages}],
        }
        input_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return input_json

    def _score_in_dir(self, image_path: Path, prompt: str, max_new_tokens: int, io_dir: Path) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._score_messages_in_dir(messages, max_new_tokens, io_dir)

    def _score_messages_in_dir(self, messages: list[dict[str, Any]], max_new_tokens: int, io_dir: Path) -> str:
        input_json = self._write_request_json_messages(messages, max_new_tokens, io_dir)
        if self.config.runtime_mode == "persistent":
            return self._score_with_persistent(input_json, max_new_tokens)
        return self._score_with_subprocess(input_json, max_new_tokens, io_dir)

    def _score_with_subprocess(self, input_json: Path, max_new_tokens: int, io_dir: Path) -> str:
        output_json = io_dir / "response.json"
        profile_json = io_dir / "profile.json"
        cmd = [
            str(self.config.inference_binary),
            "--engineDir",
            str(self.config.llm_engine_dir),
            "--multimodalEngineDir",
            str(self.config.visual_engine_dir),
            "--inputFile",
            str(input_json),
            "--outputFile",
            str(output_json),
            "--maxGenerateLength",
            str(max_new_tokens),
        ]
        if self.config.warmup:
            cmd.extend(["--warmup", str(self.config.warmup)])
        if self.config.dump_profile:
            cmd.extend(["--dumpProfile", "--profileOutputFile", str(profile_json)])

        completed = subprocess.run(
            cmd,
            cwd=str(self.config.edge_llm_path),
            env=self._edge_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.config.timeout_sec,
            check=False,
        )
        if completed.returncode != 0:
            output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
            tail = output.strip()[-3000:] or "no Edge-LLM output captured"
            raise RuntimeError(f"TensorRT Edge-LLM inference failed with exit code {completed.returncode}: {tail}")
        if not output_json.exists():
            raise FileNotFoundError(f"TensorRT Edge-LLM did not write its output JSON: {output_json}")
        response = json.loads(output_json.read_text(encoding="utf-8"))
        responses = response.get("responses")
        if not isinstance(responses, list) or not responses:
            raise RuntimeError(f"TensorRT Edge-LLM output has no responses: {output_json}")
        text = str(responses[0].get("output_text") or "").strip()
        if text == _FAILED_RESPONSE_TEXT:
            raise RuntimeError(f"TensorRT Edge-LLM returned a failed response. See {output_json}")
        return text

    def _start_server_locked(self) -> None:
        if self._server is not None and self._server.poll() is None:
            return
        if self._server is not None:
            raise RuntimeError(f"Persistent TensorRT Edge-LLM server exited with code {self._server.returncode}")

        cmd = [
            str(self.config.persistent_binary),
            "--engineDir",
            str(self.config.llm_engine_dir),
            "--multimodalEngineDir",
            str(self.config.visual_engine_dir),
            "--maxGenerateLength",
            str(self.max_new_tokens),
        ]
        self._server = subprocess.Popen(
            cmd,
            cwd=str(self.config.edge_llm_path),
            env=self._edge_env(),
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        self._server_queue = queue.Queue()
        self._server_reader = threading.Thread(target=self._pump_server_output, daemon=True)
        self._server_reader.start()

        deadline = time.monotonic() + self.config.startup_timeout_sec
        while time.monotonic() < deadline:
            payload = self._read_protocol_payload_locked(deadline, expect_ready=True)
            if payload is None:
                continue
            if payload.get("ok"):
                return
            raise RuntimeError(f"Persistent TensorRT Edge-LLM server failed to start: {payload}")
        raise TimeoutError(
            "Timed out waiting for persistent TensorRT Edge-LLM server startup. "
            f"Recent output:\n{self._server_log_tail()}"
        )

    def _read_protocol_payload_locked(self, deadline: float, *, expect_ready: bool = False) -> dict[str, Any] | None:
        if self._server is None:
            raise RuntimeError("Persistent TensorRT Edge-LLM server is not running")

        while time.monotonic() < deadline:
            if self._server.poll() is not None:
                raise RuntimeError(
                    "Persistent TensorRT Edge-LLM server exited "
                    f"with code {self._server.returncode}. Recent output:\n{self._server_log_tail()}"
                )
            timeout = max(0.0, min(1.0, deadline - time.monotonic()))
            try:
                line = self._server_queue.get(timeout=timeout)
            except queue.Empty:
                continue
            if line.startswith(_READY_PREFIX):
                if expect_ready:
                    return json.loads(line[len(_READY_PREFIX) :])
                continue
            if line.startswith(_RESPONSE_PREFIX):
                return json.loads(line[len(_RESPONSE_PREFIX) :])
            if line.startswith(_ERROR_PREFIX):
                return json.loads(line[len(_ERROR_PREFIX) :])
        return None

    def _pump_server_output(self) -> None:
        if self._server is None or self._server.stdout is None:
            return
        try:
            for line in self._server.stdout:
                clean = line.rstrip("\n")
                self._server_logs.append(clean)
                self._server_queue.put(clean)
        except Exception as exc:
            clean = f"Persistent TensorRT Edge-LLM output reader failed: {exc}"
            self._server_logs.append(clean)
            self._server_queue.put(clean)

    def _request_with_persistent(
        self,
        input_json: Path,
        max_new_tokens: int,
        *,
        token_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        with self._server_lock:
            self._start_server_locked()
            if self._server is None or self._server.stdin is None:
                raise RuntimeError("Persistent TensorRT Edge-LLM server stdin is unavailable")

            request_id = f"{time.time_ns()}"
            command = {
                "id": request_id,
                "input_file": str(input_json),
                "batch_size": 1,
                "max_generate_length": max_new_tokens,
            }
            if token_ids is not None:
                command["score_token_ids"] = [int(token_id) for token_id in token_ids]
            self._server.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
            self._server.stdin.flush()

            deadline = time.monotonic() + self.config.timeout_sec
            while time.monotonic() < deadline:
                response = self._read_protocol_payload_locked(deadline)
                if response is None:
                    continue
                if response.get("id", request_id) != request_id:
                    continue
                if not response.get("ok"):
                    raise RuntimeError(
                        "Persistent TensorRT Edge-LLM inference failed: "
                        f"{response.get('error', response)}\nRecent output:\n{self._server_log_tail()}"
                    )
                return response
            raise TimeoutError(
                "Timed out waiting for persistent TensorRT Edge-LLM inference. "
                f"Recent output:\n{self._server_log_tail()}"
            )

    def _score_with_persistent(self, input_json: Path, max_new_tokens: int) -> str:
        response = self._request_with_persistent(input_json, max_new_tokens)
        responses = response.get("responses")
        if not isinstance(responses, list) or not responses:
            raise RuntimeError(f"Persistent TensorRT Edge-LLM response has no responses: {response}")
        text = str(responses[0].get("output_text") or "").strip()
        if text == _FAILED_RESPONSE_TEXT:
            raise RuntimeError(f"Persistent TensorRT Edge-LLM returned a failed response: {response}")
        return text

    def _server_log_tail(self) -> str:
        return "\n".join(list(self._server_logs)[-40:]) or "no server output captured"

    def close(self) -> None:
        with self._server_lock:
            if self._server is None:
                return
            proc = self._server
            self._server = None
            if proc.poll() is not None:
                return
            try:
                if proc.stdin is not None:
                    proc.stdin.write(json.dumps({"shutdown": True}, separators=(",", ":")) + "\n")
                    proc.stdin.flush()
                proc.wait(timeout=5)
            except Exception:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
