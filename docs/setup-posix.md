# Linux / macOS setup

This is the native Python + llama.cpp path; no Docker or company repository is
required. Run commands from the cloned benchmark root. Internet is needed for
initial package/model downloads. These commands have not run the experiment.

## 1. Clone and install Python dependencies

Install Git, Python 3.12, CMake and a C/C++ build toolchain for your OS. On macOS
the toolchain can come from Xcode Command Line Tools; on Linux use your distribution's
compiler/build packages. Then:

```bash
git clone --branch codex/topic-aware-rag-benchmark --single-branch https://github.com/pw-Chen-km/Aiut_coop_project.git topic-aware-rag-benchmark
cd topic-aware-rag-benchmark
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[embedding,test,prepare]' 'huggingface_hub>=1,<3'
```

## 2. Prepare local models

```bash
.venv/bin/hf download intfloat/multilingual-e5-small config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json sentencepiece.bpe.model --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 --local-dir models/e5
.venv/bin/hf download bartowski/Qwen_Qwen3.5-4B-GGUF Qwen_Qwen3.5-4B-Q4_K_M.gguf --revision 4168f45a16a1290d65a4ec0fa312ae917a4c15d6 --local-dir models/qwen
.venv/bin/python scripts/record_model.py models/qwen/Qwen_Qwen3.5-4B-Q4_K_M.gguf
```

Alternatively point the config to an existing local E5 directory and run the
server with an existing Qwen 4B GGUF. Record its SHA; equal model names alone do
not prove equal weights. The public conversion above is not verified identical
to any checkpoint used in previous company experiments.

## 3. Build and start the model server (CPU)

Building from the pinned source avoids depending on a prebuilt Linux binary's
glibc version. This builds the model server only and does not run questions.

```bash
mkdir -p vendor
git clone --branch b12449 --depth 1 https://github.com/ggml-org/llama.cpp.git vendor/llama.cpp
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON -DGGML_METAL=OFF -DGGML_CUDA=OFF
cmake --build vendor/llama.cpp/build --config Release --target llama-server -j 2
vendor/llama.cpp/build/bin/llama-server --model models/qwen/Qwen_Qwen3.5-4B-Q4_K_M.gguf --alias Qwen3.5-4B-Q4_K_M --host 127.0.0.1 --port 8073 --ctx-size 8192 --parallel 1 --batch-size 256 --ubatch-size 128 --n-gpu-layers 0 --jinja --reasoning off
```

Keep the server terminal running. In another terminal, change back to the same
benchmark directory:

```bash
curl --fail http://127.0.0.1:8073/health
curl --fail http://127.0.0.1:8073/v1/models
cp configs/emanual.yaml configs/local.yaml
.venv/bin/python scripts/validate_labels.py --config configs/local.yaml
.venv/bin/python scripts/build_index.py --config configs/local.yaml
.venv/bin/python -m pytest
.venv/bin/python -m ruff check src scripts tests
.venv/bin/python scripts/run_experiment.py --config configs/local.yaml
```

The last command is a plan, **not an experiment**. The index must be built on this
machine; no local Mac index is included in Git. If rebuilding later, choose a new
`index_dir`. Use an appropriate llama.cpp GPU build if desired and record that
deployment consistently across methods; the commands above deliberately use CPU.

## 4. After reviewing the scripts

```bash
# Real Jev classification and retrieval, without final answer generation:
.venv/bin/python scripts/run_experiment.py --config configs/local.yaml --execute --skip-answer

# Real full experiment, same top-5 answer context in each method:
.venv/bin/python scripts/run_experiment.py --config configs/local.yaml --execute
```

Run these separately as needed; each creates a new `results/<run-id>/` directory.
See the root README for result fields, errors, data limitations and supported
server endpoints. Actual live model compatibility remains to be tested; the
prepared package has offline and mocked integration validation only.

Build reference: [official llama.cpp build guide](https://github.com/ggml-org/llama.cpp/blob/b12449/docs/build.md).
