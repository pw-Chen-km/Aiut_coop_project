# Topic-aware RAG benchmark

這是 `Aiut_coop_project` 的獨立實驗分支 **`codex/topic-aware-rag-benchmark`**。
此分支的根目錄就是 benchmark 專案；`main` 保留原本的 RAG 演算法核心。
不需要公司 GitLab、VPN、plugin、PostgreSQL、RAG API 或 Docker。

**目前完成資料準備與離線／模擬測試，尚未跑正式 Qwen 實驗。**
下面提供另一台電腦從零開始的步驟。Windows 使用 PowerShell；Linux/macOS 請看
[對應安裝指令](docs/setup-posix.md)。首次安裝及下載模型需要網路，模型備妥後可在本地執行。

## 下載後包含什麼

| 已放在此分支 | 要在新電腦另外準備 |
|---|---|
| Benchmark 程式、prompts、tests | Python 3.12 與 Python 套件 |
| E-Manual 原始及整理後資料、topic labels | E5 模型，約 470 MB 權重 |
| E5 tokenizer | Qwen 4B Q4_K_M GGUF，數 GB |
| 實驗設定與說明 | llama.cpp server 執行檔 |
| 66 題、102 段來源、115 個 chunks | 在該台電腦重建 E5 vectors |

**Git 不包含模型權重、`.venv`、本機路徑、已建立的 index 或實驗結果。**
不需要 Git LFS。程式不會自行啟動／重啟服務或切換模型。

## Windows：從 clone 到可以執行實驗

以下以 Windows x64 CPU 為例。先安裝 Git 與 Python 3.12。
需要有足夠空間放 Qwen、E5 和 Python 環境；記憶體需求也包含模型 context、KV cache 和暫存 buffer。

### 1. Clone 正確分支

```powershell
git clone --branch codex/topic-aware-rag-benchmark --single-branch https://github.com/pw-Chen-km/Aiut_coop_project.git topic-aware-rag-benchmark
cd topic-aware-rag-benchmark
```

後面所有命令都在這個資料夾執行。使用獨立資料夾，不要覆蓋另一個正在使用的 repo。

### 2. 建立 Python 環境

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[embedding,test,prepare]" "huggingface_hub>=1,<3"
```

這份教學直接使用 `.venv` 內的執行檔，不需要修改 PowerShell execution policy。
`requirements-prepared.txt` 是原準備環境的版本紀錄；新平台請依上面方式安裝並重建 index。

### 3. 下載固定版本的 E5 與 Qwen

以下公開模型不需要公司帳號。版本與來源也記錄在 [model-sources.json](configs/model-sources.json)。

```powershell
.\.venv\Scripts\hf.exe download intfloat/multilingual-e5-small config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json sentencepiece.bpe.model --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 --local-dir models/e5
.\.venv\Scripts\hf.exe download bartowski/Qwen_Qwen3.5-4B-GGUF Qwen_Qwen3.5-4B-Q4_K_M.gguf --revision 4168f45a16a1290d65a4ec0fa312ae917a4c15d6 --local-dir models/qwen
.\.venv\Scripts\python.exe scripts/record_model.py models/qwen/Qwen_Qwen3.5-4B-Q4_K_M.gguf
```

公開 GGUF 是 Qwen3.5-4B 的第三方量化轉換。沒有證據顯示它與先前公司使用的檔案
完全相同；比較舊實驗前須核對 SHA-256，不能只看 `4B` 名稱。
`record_model.py` 只記錄檔案 hash，不載入模型、不跑問題。

若已有模型，可自行複製到上述路徑，或在後面的 config／啟動命令指定既有位置。

### 4. 取得 llama.cpp 並啟動 Qwen

以下固定使用官方 [b12449 release](https://github.com/ggml-org/llama.cpp/releases/tag/b12449)
的 Windows x64 CPU 執行檔；保留解壓後的所有 DLL。

```powershell
New-Item -ItemType Directory -Force vendor | Out-Null
Invoke-WebRequest -Uri "https://github.com/ggml-org/llama.cpp/releases/download/b12449/llama-b12449-bin-win-cpu-x64.zip" -OutFile "vendor/llama.zip"
if ((Get-FileHash "vendor/llama.zip" -Algorithm SHA256).Hash.ToLowerInvariant() -ne "461d13cd0c991427dccabb89535a1545089f36d7e4bf78e698fc8061a3344bf0") { throw "llama.cpp archive checksum mismatch" }
Expand-Archive -Path "vendor/llama.zip" -DestinationPath "vendor/llama" -Force
$llamaServer = Get-ChildItem "vendor/llama" -Recurse -Filter "llama-server.exe" | Select-Object -First 1
if (-not $llamaServer) { throw "llama-server.exe not found" }
& $llamaServer.FullName --model models/qwen/Qwen_Qwen3.5-4B-Q4_K_M.gguf --alias Qwen3.5-4B-Q4_K_M --host 127.0.0.1 --port 8073 --ctx-size 8192 --parallel 1 --batch-size 256 --ubatch-size 128 --n-gpu-layers 0 --jinja --reasoning off
```

保持這個 PowerShell 視窗開啟；這是模型服務，還沒有執行 benchmark。
`--model` 必須存在，否則 server 不會自動知道要載入哪個 GGUF。

另開 PowerShell，`cd` 回同一個專案資料夾，檢查模型載入完成：

```powershell
Invoke-RestMethod http://127.0.0.1:8073/health
Invoke-RestMethod http://127.0.0.1:8073/v1/models
```

如果模型在另一台主機，只需把 config 的 `model_endpoint` 改成可連線的網址；
`127.0.0.1` 永遠指目前執行腳本的這台電腦。

### 5. 建立本機設定與 E5 index

```powershell
Copy-Item configs/emanual.yaml configs/local.yaml
.\.venv\Scripts\python.exe scripts/validate_labels.py --config configs/local.yaml
.\.venv\Scripts\python.exe scripts/build_index.py --config configs/local.yaml
```

第一次 clone 不會有 `artifacts/emanual-e5-v2`，以上命令會建立它。
如果已經有相同 index，**不要重複 build**；若要改 tokenizer、套件、chunk 或模型，
先把 `index_dir` 改成新的資料夾，例如 `artifacts/emanual-e5-v3`，再重建。

設定檔的重要欄位：

```yaml
e5_model_dir: models/e5
index_dir: artifacts/emanual-e5-v2
model_endpoint: http://127.0.0.1:8073
expected_model_marker: 4b
context_tokens: 8192
candidate_k: 20
final_k: 5
rrf_k: 60
soft_weights: [1, 1, 1]
```

`soft_weights: [1,1,1]` 表示三路等權，各為 `1/3`。
E5 build 只處理來源段落，不跑測試問題。資料已包含於 Git，通常不需要重新下載或 parsing。

### 6. 先檢查，暫不執行實驗

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe scripts/run_experiment.py --config configs/local.yaml
```

最後一行只會顯示 `prepared_not_run`、資料數量和實驗設定，**不連模型、不跑問題**。
Tests 只使用離線資料和模擬回應。此時應確認 `index_present` 是 `true`。

### 7. 確認腳本後才開始實驗

Retrieval-only：會呼叫 Qwen 做 Jev topic scoring，但不生成最終答案。

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --config configs/local.yaml --execute --skip-answer
```

完整比較：三種方法都跑 66 題；每種方法只用 top-5 生成答案。

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --config configs/local.yaml --execute
```

這兩個指令**不是前置檢查**，有 `--execute` 才會開始正式推理。本次交付未執行它們。
兩次執行會建立兩個不同的結果資料夾，完整比較不會沿用前一次的 topic 分數。

## 結果在哪裡

每次執行會印出 `results/<run-id>`，不覆蓋舊結果。

- `summary.csv`：用 Excel 開啟，看 evidence recall@5/10/20、耗時、回答／拒答／錯誤數量。
- `baseline.jsonl`、`hard_topic.jsonl`、`soft_topic.jsonl`：逐題問題、全部可用參考回答、evidence、檢索內容與生成答案。
- `topic_scores.jsonl`：Jev 的 token 分數、機率、耗時與錯誤。
- `answer_review.jsonl`：逐題語意評分欄位；尚未評分時 accuracy 是 null。
- `manifest.json`：模型識別、資料／程式／index hashes、完成或失敗狀態。

`summary.csv` 可以使用 PowerShell 開啟：

```powershell
Get-ChildItem results -Recurse -Filter summary.csv
# 把下面的 <run-id> 換成剛才顯示的資料夾名稱
Invoke-Item "results/<run-id>/summary.csv"
```

也請把 `artifacts/qwen-checksum.json` 與該次結果一起保存，以記錄實際使用的 GGUF。

## 部署範圍與已驗證程度

Jev 需要 llama.cpp 的 `/apply-template`、`/tokenize`、`/completion` raw token log
probabilities，以及 `/v1/models`、`/props`；答案使用 `/v1/chat/completions`。
只有文字生成介面的服務，例如只有 OpenAI chat endpoint，不能直接替代這個 scorer。

目前已驗證離線資料、E5 index 與 mock integration；公開 release 的下載名稱及 SHA
也已核對。**尚未在 Windows 實機或實際 Qwen server 完成 end-to-end Jev 測試**，
不能把 unit tests 通過當成模型實驗已通過。若 server 未提供需要的分數，腳本會記錄
錯誤，不會假造機率或偷偷改成一般 JSON classifier。

## 常見問題

| 狀況 | 處理 |
|---|---|
| `Connection refused` | Qwen 尚未啟動、未載入完成，或 endpoint／port 不對；先檢查 `/health` |
| model identity 不符合 `4b` | 確認部署的是 4B，並使用上述 `--alias`；不要只改標籤掩蓋不同模型 |
| 找不到 E5／weights | 檢查 `models/e5` 是否含實際 `model.safetensors`；路徑寫在 `configs/local.yaml` |
| index directory already exists | 沿用已完成 index，或用新的 `index_dir` 重建 |
| embedding dependencies changed／stale index | 在新的 index directory 用目前環境重新 build |
| candidate outside returned logits | `max_n_probs` 限制內缺少必要 token；該結果是 error，不能當正常機率 |
| no raw token log probabilities | 確認是支援所需 `/completion` 格式的 llama.cpp；不能只靠 chat API |
| `std::bad_alloc`／記憶體不足 | 關閉其他模型、維持 `--parallel 1`，必要時降低 batch／ubatch；不得把 failed rows 當成功 |
| 想用 NVIDIA GPU | 換成相容的 llama.cpp GPU build 並調整 GPU layers；所有比較保持相同部署，記錄差異 |

## 專案結構與工作流

```text
configs/                    可攜實驗設定、固定模型來源
assets/e5-tokenizer/         Tokenizer（不含 weights）
data/emanual/raw/            原始資料與來源版本
data/emanual/processed/      Sources、chunks、questions、validation report
data/emanual/annotations/    Taxonomy、source topic labels
src/topic_rag/               純 benchmark 演算法
prompts/                    Topic 與 answer prompts
scripts/                    準備、檢查、build index、執行實驗
tests/                      離線／模擬測試
docs/                       中文流程、Jev 細節與實驗 protocol
models/                     本機下載的模型，Git 忽略
vendor/                     本機 llama.cpp，Git 忽略
artifacts/                  本機 E5 index／模型 checksum，Git 忽略
results/                    本機實驗輸出，Git 忽略
```

```text
Offline: sources → source topic labels → exact-span chunks → E5 vectors
Online: query → baseline / Jev hard / Jev soft → ranked candidates
        → recall@5/10/20 → top-5 → same Qwen 4B answer → results
```

若要重新產生資料，使用 `scripts/prepare_data.py`；下載原始資料的
`scripts/download_data.py` 是可選操作。來源已是文字，不需要 PDF／OCR。

完整方法：[中文交接](docs/workflow.zh-TW.md)、[實驗 protocol](docs/experiment-protocol.md)、
[Jev 實作](docs/jev-implementation.md)。

## Methods and evaluation

The three conditions use identical source text, E5 weights, questions and answer
prompt:

| Condition | Candidate search | Fusion |
|---|---|---|
| `baseline` | Global BM25 top-20 + E5 top-20 | 0.5 BM25 + 0.5 E5 RRF |
| `hard_topic` | Restrict to predicted topic, then top-20 per retriever | 0.5 BM25 + 0.5 E5 RRF |
| `soft_topic` | Same global candidate union as baseline | 1/3 BM25 + 1/3 E5 + 1/3 topic RRF |

`rrf_k=60`. For a source with topics 2 and 6, its chunks receive topic mass
`p("2") + p("6")`. Rank this mass within the candidate union; equal masses share
the same rank. RRF combines **ranks**, not raw BM25/cosine/probability values.
The answer model only receives top-5. Recall@10/@20 does not generate extra answers.

## Results

```text
results/<UTC-timestamp>-<unique-id>/
├── manifest.json            Model identity, data/code/index hashes, run status
├── config.json              Resolved configuration
├── topic_scores.jsonl       Full suffix token scores, probabilities and timings
├── baseline.jsonl
├── hard_topic.jsonl
├── soft_topic.jsonl          References, evidence, retrieved passages, answer, error
├── answer_review.jsonl      Empty semantic score/reason fields for later review
├── summary.json
└── summary.csv              Recall, response counts, errors and latency
```

Evidence recall checks exact source offsets, not answer-word overlap. Source
recall is reported separately. Retrieval metrics survive answer-generation
failures. Summaries include both successful-retrieval recall and a conservative
score treating retrieval failures as zero. No answer F1 or invented accuracy is
reported; `answer_accuracy` stays null until semantic review.

## Data limitations

The pinned test split contains 132 records, **66 unique questions** and **102
unique supplied passages**. Two records have invalid evidence keys and are
excluded from normalized references; their questions remain through valid
duplicate records. All raw records remain available. Prepared data has 115
chunks, 130 valid reference responses and 538 question/evidence associations.

RAGBench reference responses and evidence annotations are model-generated; they
are not a human-verified QA answer set. Fourteen retained references are marked
unsupported by the dataset and must not be treated as authoritative answers.
The corpus is the union of supplied test passages, not the original full manuals.
This is a controlled exploratory retrieval study, not an official RAGBench score.

Source labels were authored by Codex after reading the 102 source passages, with
no question/answer use for labeling. There are 10 numeric options, including
`"10"`; no `T01` mapping. This is a source-derived, in-domain taxonomy, not a
held-out-domain evaluation. Freeze it before scoring questions.

Jev uses complete candidate suffix likelihoods, including `]`, followed by
softmax. The HTTP implementation evaluates a shared suffix trie sequentially;
it does **not** reproduce the paper's optimized parallel KV-cache latency. See
[Jev implementation](docs/jev-implementation.md) for the exact equation, request
contract, failure modes and scoring tests.

## References

- [RAGBench data and dataset card](https://huggingface.co/datasets/galileo-ai/ragbench), pinned at `97808f3e5fd16ede40bbff6c2949af8139b2eb7b`; dataset license CC-BY-4.0. Derived data retains attribution and records transformations.
- [LLM-as-Jev paper, v3](https://arxiv.org/html/2610.02076v3), Sections 3.2–3.3 and Appendix A.
- [llama.cpp server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
- [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small); included tokenizer files retain their upstream model attribution.
