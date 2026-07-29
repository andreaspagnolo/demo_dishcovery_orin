# Dishcovery Quantized Inference on NVIDIA Jetson AGX Orin

## Reproduction baseline

This repository is the minimal NVIDIA Jetson AGX Orin package for reproducing
two quantized 350-image Dishcovery results and running the calorie-estimation
demo:

1. Task 1 ingredient recognition with fixed top-20 candidates.
2. Task 2 caption retrieval with fixed top-5 candidates.
3. Calorie estimation using SigLIP2 top-20 ingredient candidates and the
   quantized Task 1 Qwen3-VL-4B engine.

The deployed stack is hybrid precision:

- SigLIP2 `timm/ViT-gopt-16-SigLIP2-384`: TensorRT FP16 visual engine;
- Task 1 Qwen3-VL-4B-Instruct: EdgeLLM INT4 AWQ language engine and FP16
  visual engine;
- Task 2 Qwen3-VL-Reranker-2B: EdgeLLM INT4 AWQ language engine and FP16
  visual engine.

Only code and small reproducibility inputs are stored in Git. Images and
multi-gigabyte TensorRT/EdgeLLM artifacts are downloaded from Google Drive.

## Contents

- [1. Repository and asset layout](#1-repository-and-asset-layout)
- [2. Reference platform](#2-reference-platform)
- [3. Download external assets](#3-download-external-assets)
- [4. Build the runtime environment](#4-build-the-runtime-environment)
- [5. Verify the setup](#5-verify-the-setup)
- [6. Reproduce the 350-image results](#6-reproduce-the-350-image-results)
- [7. Run the quantized demo](#7-run-the-quantized-demo)
- [8. Rebuild the quantized models](#8-rebuild-the-quantized-models)
- [9. Reproducibility notes](#9-reproducibility-notes)

---

## 1. Repository and asset layout

Repository map:

```text
benchmark_inputs/
├── caches/                  Frozen SigLIP2 text/caption embeddings
├── calories/                Calorie and portion table plus ingredient labels
├── demo_showcase/           Curated good-predictions image list
├── task1/                   Task 1 manifest, labels, and ground truth
└── task2/                   Task 2 manifest, captions, and ground truth
code/
├── quantization/            SigLIP2 and Qwen/EdgeLLM build helpers
├── edgellm_qwen.py          Persistent EdgeLLM client
├── orin_calorie_demo.py     Calorie table, portion, and composition logic
├── orin_calorie_demo_edgellm.py  Quantized EdgeLLM calorie backend
├── orin_measurements.py     Latency, RAM, power, and nvpmodel measurement
├── orin_task1_pipeline.py   Task 1 evaluation/inference
└── orin_task2_pipeline.py   Task 2 evaluation/inference
demo/                        Browser-demo routing, history, nutrition, and UI assets
demo_web_app.py              Browser demo server entry point
config/                      Platform, checksums, and expected metrics
patches/                     Persistent EdgeLLM server and reranker-logit patch
reference_results/           Per-image predictions from the reference runs
scripts/
├── run_benchmarks.py        Fixed Task 1/Task 2 reproduction entry point
└── verify_setup.py          Input, image, runtime, and engine verification
```

After downloading and extracting the external package, the repository must
have this additional ignored layout:

```text
external_assets/
├── images/
│   ├── task1_350/                         # 350 img_*.jpg files
│   ├── task2_350/                         # 350 class/image.jpg files
│   └── showcase/                          # Optional demo images
└── models/
    ├── shared_SigLIP2_FP16_TensorRT/
    │   └── visual_fp16.engine
    ├── task1_Qwen3-VL-4B-AWQ-INT4/
    │   └── engines/
    │       ├── llm_in4096_kv8192/
    │       └── visual/visual/
    └── task2_Qwen3-VL-Reranker-2B-INT4/
        ├── engine_in1024/llm/
        └── engine/visual/visual/
```

TensorRT-Edge-LLM is kept next to this repository:

```text
parent_directory/
├── demo_dishcovery_orin/
└── TensorRT-Edge-LLM/
```

## 2. Reference platform

The archived measurements were produced with:

```text
Device:                 NVIDIA Jetson AGX Orin 64 GB
Power mode:             MODE_50W, nvpmodel mode 3
JetPack:                6.2.2
L4T:                    36.5.0
Kernel:                 5.15.185-tegra
CUDA:                   12.6
TensorRT:               10.3.0
Python:                 3.10.20
PyTorch:                2.8.0
torchvision:            0.23.0
TensorRT-Edge-LLM:      f9cc74623d95d7acf1addab6026b9d410ba81f52
```

The machine-readable contract is in `config/platform.json`. TensorRT engines
are not portable across arbitrary CUDA, TensorRT, GPU, or JetPack versions.
Use the downloaded engines only on a compatible Orin stack; rebuild them on
the target device otherwise.

Set the reference power mode before measuring latency or RAM:

```bash
sudo nvpmodel -m 3
sudo jetson_clocks
nvpmodel -q
```

## 3. Download external assets

The model and image package is stored in:

[Google Drive — Dishcovery rebuttal quantized assets](https://drive.google.com/drive/folders/194XVy0C3XyMuthyqAMJkII7oxLV4yNcu)

### Browser download

Download the Drive folder, copy its contents into `external_assets/`, and
extract the image archive:

```bash
mkdir -p external_assets/images
unzip external_assets/images/dishcovery_rebuttal_images_780.zip \
  -d external_assets/images
```

The archive also contains `showcase/`; the benchmark uses only `task1_350/`
and `task2_350/`.

### Command-line download from an authorized Drive account

Install and configure `rclone`:

```bash
sudo apt-get update
sudo apt-get install -y rclone unzip
rclone config
```

Then replace `mydrive` with the remote name created by `rclone config`:

```bash
mkdir -p external_assets
rclone copy \
  "mydrive:Dishcovery_rebuttal_quantized_2026-07-13" \
  external_assets \
  --progress

unzip external_assets/images/dishcovery_rebuttal_images_780.zip \
  -d external_assets/images
```

Do not commit `external_assets/`; it is excluded by `.gitignore`.

## 4. Build the runtime environment

### 4.1 Python environment

JetPack-compatible PyTorch and torchvision wheels must be installed before
the small Python requirements. Do not use the ordinary PyPI PyTorch package:
the benchmark requires an `aarch64` build with CUDA support for JetPack 6.2
and CUDA 12.6.

Create the virtual environment with access to the Python packages supplied by
JetPack:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install the Python dependencies required by the Jetson PyTorch wheels from the
standard PyPI index. Pinning SymPy is necessary because the Ubuntu system copy
is too old for PyTorch 2.8.0, and the local virtual-environment copy must take
precedence over `/usr/lib/python3/dist-packages`:

```bash
python -m pip install --index-url https://pypi.org/simple \
  filelock \
  typing-extensions \
  "sympy==1.13.3" \
  networkx \
  jinja2 \
  fsspec \
  numpy \
  pillow \
  cffi
```

Install the validated JetPack 6 / CUDA 12.6 `aarch64` wheels. `--no-deps` is
intentional: the Jetson wheel index contains the platform-specific wheels but
not every generic Python dependency.

```bash
python -m pip install --no-cache-dir --no-deps \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  torch==2.8.0 \
  torchvision==0.23.0
```

Then install the repository requirements and check the environment:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

Verify that the expected packages are imported from the virtual environment and
that CUDA is available before continuing:

```bash
python - <<'PY'
import sympy
import torch
import torchvision

print("SymPy:", sympy.__version__, sympy.__file__)
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch is installed, but CUDA is not available")

print("GPU:", torch.cuda.get_device_name(0))
PY
```

The expected versions are PyTorch `2.8.0`, torchvision `0.23.0`, and a CUDA
`12.6` build. The exact GPU name may vary slightly, but it must identify the
Jetson Orin GPU.

TensorRT Python bindings are supplied by JetPack under the system Python
packages. The pipeline automatically checks
`/usr/lib/python3.10/dist-packages` when importing TensorRT.

### 4.2 Build the patched TensorRT-Edge-LLM runtime

Clone the exact source revision:

```bash
cd ..
git clone --recurse-submodules \
  https://github.com/NVIDIA/TensorRT-Edge-LLM.git
git -C TensorRT-Edge-LLM checkout \
  f9cc74623d95d7acf1addab6026b9d410ba81f52
git -C TensorRT-Edge-LLM submodule update --init --recursive
cd demo_dishcovery_orin
```

Add the persistent server and the token-logit interface required by Task 2:

```bash
cp patches/llm_persistent_server.cpp \
  ../TensorRT-Edge-LLM/examples/llm/llm_persistent_server.cpp

git -C ../TensorRT-Edge-LLM apply \
  "$PWD/patches/tensorrt_edgellm_reranker_logits.patch"
```

Build on JetPack 6.2 Orin. CuTe DSL must be disabled for this pinned
TensorRT-Edge-LLM revision on the reference CUDA 12.6 stack. Enabling it with
`-DENABLE_CUTE_DSL=ALL` can fail during linking with unresolved
`cudaLibrary*` and `cudaKernelSetAttributeForDevice` symbols, and it is not
required by these Dishcovery engines.

Remove a previous failed or differently configured build directory, configure,
and compile:

```bash
rm -rf ../TensorRT-Edge-LLM/build

cmake -S ../TensorRT-Edge-LLM -B ../TensorRT-Edge-LLM/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCUDA_CTK_VERSION=12.6 \
  -DENABLE_CUTE_DSL=OFF

cmake --build ../TensorRT-Edge-LLM/build -j"$(nproc)"
```

The build is successful only if the command finishes without `Error 1`,
`Error 2`, or `undefined reference` messages. A final line such as
`Built target NvInfer_edgellm_plugin` confirms the plugin target, but both
runtime artifacts must be checked explicitly:

```bash
ls -lh \
  ../TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so \
  ../TensorRT-Edge-LLM/build/examples/llm/llm_persistent_server

test -e ../TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so \
  && test -x ../TensorRT-Edge-LLM/build/examples/llm/llm_persistent_server \
  && echo "TensorRT-Edge-LLM build completed successfully"
```

`libNvInfer_edgellm_plugin.so` may be a symbolic link to a versioned library
such as `libNvInfer_edgellm_plugin.so.1`; this is normal as long as the target
exists. The reproduction scripts require these resolved paths:

```text
../TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
../TensorRT-Edge-LLM/build/examples/llm/llm_persistent_server
```

If TensorRT-Edge-LLM is elsewhere, set:

```bash
export EDGE_LLM_PATH=/absolute/path/to/TensorRT-Edge-LLM
```

## 5. Verify the setup

First verify the Python runtime. This separate check is required because
`scripts/verify_setup.py` validates repository inputs, images, runtime files,
and engines, but it does not currently prove that PyTorch can be imported or
that CUDA is visible:

```bash
python - <<'PY'
import torch
import torchvision

print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in the active Python environment")

print("GPU:", torch.cuda.get_device_name(0))
PY
```

Next run the quick repository, image-count, runtime, and engine-path check:

```bash
python scripts/verify_setup.py --skip-engine-hashes
```

Every row must be reported as `OK`, followed by:

```text
Setup is complete and matches the archived inputs.
```

Finally, verify all Git inputs, all 700 benchmark images, the runtime binaries,
and the five key engine hashes:

```bash
python scripts/verify_setup.py
```

Hashing the approximately 6 GB of engine files takes some time. The fixed input
and engine hashes are also documented in `config/sha256.txt`.

## 6. Reproduce the 350-image results

The wrappers contain the full fixed CLI contract. Print the resolved command
without running inference:

```bash
python scripts/run_benchmarks.py task1 --print-only
python scripts/run_benchmarks.py task2 --print-only
```

Run Task 1 with the legacy prompt/fusion/selector and fixed top-20 candidate
list:

```bash
python scripts/run_benchmarks.py task1
```

Run Task 2 with the 4,940-caption bank, fixed top-5 candidate list,
SigLIP-guarded final selection, and gap 3.0:

```bash
python scripts/run_benchmarks.py task2
```

Run both sequentially:

```bash
python scripts/run_benchmarks.py all
```

Outputs are written to:

```text
run_outputs/
├── task1/result.json
├── task1/predictions.csv
├── task2/result.json
└── task2/predictions.csv
```

The wrapper fails if accuracy differs from the archived values:

| Task | Fixed policy | Expected quality | Reference latency |
| --- | --- | ---: | ---: |
| Task 1 | Legacy fixed top-20 | F1 `0.757877`; P/R `0.787931/0.730032` | avg `0.978 s`; P95 `2.359 s` |
| Task 2 | Guarded fixed top-5 | caption acc `0.671429`; class acc `0.874286` | avg `1.052 s`; P95 `2.111 s` |

Task 2 reference RAM was `2515.93 MiB` using
`peak system-used RAM - baseline system-used RAM`.

Accuracy should be deterministic with identical artifacts. Latency and RAM
vary with thermals, clocks, power mode, background processes, and JetPack.
Archived per-image predictions are under `reference_results/`.

### Exact Task 1 contract

```text
Rows/order:                 committed 350-row manifest, in list order
Seed:                       7
Candidate policy:           fixed top-20
Prompt:                     legacy present/possible
Maximum generated tokens:   96
SigLIP/Qwen weights:         0.25 / 0.75
Final selector:             global row-z threshold_ratio
Selector values:            threshold 3.5, ratio 0.85, max 7 labels
Skip-VLM relative gap:       0.25
```

### Exact Task 2 contract

```text
Rows/order:                 committed 350-row manifest
Seed:                       42
Caption bank:               all 4,940 captions
Caption text:               class_caption
SigLIP candidates:          fixed top-5
Reranker candidates:        fixed top-5
Final policy:               siglip_guarded
Guard gap:                  3.0
Reranker scoring:           patched next-token true/false logits
```

## 7. Run the quantized demo

The browser application runs the quantized Task 1, Task 2, and calorie routes
on an uploaded image or on the curated 80-image good-predictions showcase. It
uses the same FP16 SigLIP2 engine, INT4 Task 1 Qwen engine, INT4 Task 2 reranker,
and fixed candidate policies as this package.

```bash
python demo_web_app.py \
  --host 127.0.0.1 \
  --port 8787 \
  --demo-subset good \
  --preload-backends task1,task2_fast,calories
```

Open [http://127.0.0.1:8787](http://127.0.0.1:8787) in a browser on the Orin.
For a remote browser, use the Orin’s LAN address for `--host` and open the
corresponding address from the client machine. Press `Ctrl+C` in the terminal to
stop the server.

The initial model preload can take time; the terminal shows its progress. The
web UI lets you choose a showcase image or upload a local image, then run
**Task 1**, **Task 2**, **Estimate calories**, or both Task 1 and Task 2.
Per-request JSON traces and nutrition history are saved under `demo_runs/web/`.

For a quick UI-only check that does not load models, use:

```bash
python demo_web_app.py --mock-backend
```

### Optional browser STT and TTS models

The quantized pipelines do not require speech models. To enable the browser
voice-command button, install `faster-whisper` and download its non-quantized
`base.en` CTranslate2 model into the ignored external-assets directory:

```bash
python -m pip install "faster-whisper>=1.1.0" "huggingface_hub>=0.23"

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Systran/faster-whisper-base.en",
    local_dir="external_assets/audio/faster-whisper-base.en",
)
PY
```

To enable spoken responses, install Piper and download the matching Lessac
voice files:

```bash
python -m pip install "piper-tts==1.4.2"

python - <<'PY'
from huggingface_hub import hf_hub_download

for filename in (
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
):
    hf_hub_download(
        repo_id="rhasspy/piper-voices",
        filename=filename,
        local_dir="external_assets/tts/piper",
    )
PY

test -f external_assets/audio/faster-whisper-base.en/model.bin
test -f external_assets/tts/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx
test -f external_assets/tts/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

Start the web app with the downloaded paths explicitly selected:

```bash
python demo_web_app.py \
  --host 127.0.0.1 \
  --port 8787 \
  --demo-subset good \
  --preload-backends task1,task2_fast,calories \
  --web-stt-model external_assets/audio/faster-whisper-base.en \
  --piper-model external_assets/tts/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx \
  --piper-config external_assets/tts/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

The STT and TTS models are ordinary FP32/FP16 application models; they are
separate from the quantized benchmark engines and are not included in the
Drive model package. If they are not installed, image upload and all three
quantized routes continue to work; only voice input or spoken playback is
unavailable.

The calorie result is an estimate: the showcase has no quantity or calorie
ground truth, so the web route demonstrates inference rather than calorie
accuracy. Speech input and Piper text-to-speech are optional and are disabled
gracefully when their external packages or models are unavailable.

## 8. Rebuild the quantized models

Downloaded engines reproduce the archived machine most closely. Rebuilding is
required when TensorRT, CUDA, JetPack, or the target GPU differs.

Qwen quantization/export is normally performed on an x86-64 Linux host with an
NVIDIA GPU. TensorRT engine compilation must then be performed on the target
Orin. Keep TensorRT-Edge-LLM pinned to the commit above for both stages.

### 8.1 Prepare TensorRT-Edge-LLM on the x86 export host

```bash
git clone --recurse-submodules \
  https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git checkout f9cc74623d95d7acf1addab6026b9d410ba81f52

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install ".[tools]"

export EDGE_LLM_PATH="$PWD"
export PYTHONPATH="$EDGE_LLM_PATH${PYTHONPATH:+:$PYTHONPATH}"
```

### 8.2 Quantize and export Task 1 Qwen3-VL-4B

```bash
export DISH_BUILD_ROOT=/absolute/path/to/model_build

tensorrt-edgellm-quantize llm \
  --model_dir Qwen/Qwen3-VL-4B-Instruct \
  --output_dir "$DISH_BUILD_ROOT/Qwen3-VL-4B-Instruct/quantized_int4_awq" \
  --quantization int4_awq

tensorrt-edgellm-export \
  "$DISH_BUILD_ROOT/Qwen3-VL-4B-Instruct/quantized_int4_awq" \
  "$DISH_BUILD_ROOT/Qwen3-VL-4B-Instruct/onnx" \
  --externalize-weights int4_ffn
```

Copy the resulting `onnx/` directory to the Orin, then build:

```bash
export EDGE_LLM_PATH=/absolute/path/to/TensorRT-Edge-LLM
export DISH_BUILD_ROOT=/absolute/path/to/model_build

"$EDGE_LLM_PATH/build/examples/llm/llm_build" \
  --onnxDir "$DISH_BUILD_ROOT/Qwen3-VL-4B-Instruct/onnx/llm" \
  --engineDir "$PWD/external_assets/models/task1_Qwen3-VL-4B-AWQ-INT4/engines/llm_in4096_kv8192" \
  --maxBatchSize 1 \
  --maxInputLen 4096 \
  --maxKVCacheCapacity 8192

"$EDGE_LLM_PATH/build/examples/multimodal/visual_build" \
  --onnxDir "$DISH_BUILD_ROOT/Qwen3-VL-4B-Instruct/onnx/visual" \
  --engineDir "$PWD/external_assets/models/task1_Qwen3-VL-4B-AWQ-INT4/engines/visual" \
  --minImageTokens 128 \
  --maxImageTokens 512 \
  --maxImageTokensPerImage 512
```

### 8.3 Quantize and export Task 2 Qwen3-VL-Reranker-2B

On the x86 export host:

```bash
tensorrt-edgellm-quantize llm \
  --model_dir Qwen/Qwen3-VL-Reranker-2B \
  --output_dir "$DISH_BUILD_ROOT/Qwen3-VL-Reranker-2B/quantized_int4_awq" \
  --quantization int4_awq

tensorrt-edgellm-export \
  "$DISH_BUILD_ROOT/Qwen3-VL-Reranker-2B/quantized_int4_awq" \
  "$DISH_BUILD_ROOT/Qwen3-VL-Reranker-2B/onnx" \
  --externalize-weights int4_ffn
```

Copy `onnx/` to the Orin, then build the exact 1024/1024 LLM profile and
reference visual profile:

```bash
"$EDGE_LLM_PATH/build/examples/llm/llm_build" \
  --onnxDir "$DISH_BUILD_ROOT/Qwen3-VL-Reranker-2B/onnx/llm" \
  --engineDir "$PWD/external_assets/models/task2_Qwen3-VL-Reranker-2B-INT4/engine_in1024/llm" \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 1024

"$EDGE_LLM_PATH/build/examples/multimodal/visual_build" \
  --onnxDir "$DISH_BUILD_ROOT/Qwen3-VL-Reranker-2B/onnx/visual" \
  --engineDir "$PWD/external_assets/models/task2_Qwen3-VL-Reranker-2B-INT4/engine/visual" \
  --minImageTokens 4 \
  --maxImageTokens 1024 \
  --maxImageTokensPerImage 512
```

`code/quantization/build_qwen_edgellm.py` provides the same
download/quantize/export/build workflow as a convenience for machines where
all stages are available in one environment.

### 8.4 Export and build SigLIP2 FP16

Install the build-only dependencies:

```bash
python -m pip install -r requirements-build.txt
```

Export the visual encoder:

```bash
python code/quantization/export_siglip2_visual_onnx.py \
  --output model_build/siglip2/visual.onnx
```

Build the engine on the target Orin:

```bash
python code/quantization/build_siglip2_fp16_engine.py \
  --onnx model_build/siglip2/visual.onnx \
  --output external_assets/models/shared_SigLIP2_FP16_TensorRT/visual_fp16.engine \
  --trtexec /usr/src/tensorrt/bin/trtexec
```

The input contract is `pixel_values:1x3x384x384`, with FP16 enabled and a
4,096 MiB TensorRT workspace.

After any rebuild, run:

```bash
python scripts/verify_setup.py --skip-engine-hashes
python scripts/run_benchmarks.py all
```

A rebuilt TensorRT engine may have a different SHA-256 because engine
serialization depends on the builder and platform. The fixed accuracy
verification is the final compatibility test.

## 9. Reproducibility notes

- The Task 1 list is the full-difficulty 350 subset with SHA-256
  `ebbdde7f...2342f47`; it is not the older “significant subset.”
- Task 2 uses the 350-image list but the complete 4,940-caption candidate bank.
- Calorie estimation reuses the Task 1 INT4 Qwen3-VL-4B and shared FP16
  SigLIP2 engines; it does not require another model download.
- The cached embeddings are required because the serialized TensorRT SigLIP2
  backend contains only the visual encoder.
- The Task 2 reranker must use `logit_score`; generated yes/no text is not the
  same scoring method.
- The persistent server patch is required for Task 2 and is also used for the
  low-overhead Task 1 runtime.
- Model loading and cached text-bank loading are excluded from the reported
  per-query latency.
- `ram.delta_system_used_mib`, not process RSS, is the reported RAM definition.
