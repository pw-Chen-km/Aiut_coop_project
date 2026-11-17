# Topic-aware RAG：交接與待確認流程

## 目前完成的範圍

專案獨立在 `topic-aware-rag-benchmark/`。既有 `qursor-llm-rag` 的 topic 草稿
已先備份，再移出；production repo 維持原本 commit，工作目錄乾淨。
沒有改動既有 RAG 的 CI、API、Docker、Compose 或 plugin。
GitHub 交付使用獨立的 `codex/topic-aware-rag-benchmark` 分支；新電腦安裝請從根目錄 README 開始。

準備好資料、source topic 標註、chunks、E5 vectors、三種檢索方法、Jev 計分器、
統一回答器和測試。**尚未執行正式問答或 topic classification 實驗。**

## 資料和 preprocessing

目前使用 RAGBench 的 E-Manual test split，內容是電視操作手冊段落。

| 項目 | 數量／處理方式 |
|---|---|
| 原始紀錄 | 132 筆，完整保留 |
| 不重複問題 | 66 題，重複的生成回答合併到同一題 |
| 不重複來源 | 102 段，依原文 SHA-256 建立 stable ID |
| Chunks | 115 個，最多 384 E5 tokens，保留完整原文與 offset |
| Topic | 10 類，ID 固定為 `"1"` 到 `"10"` |
| 可用參考回答 | 130 筆，其中 14 筆在原資料被標為不完全受來源支持 |
| Evidence | 538 個 question/evidence 對應 |
| 無效紀錄 | 2 筆 evidence key 帶多餘句點，無法精確對應，列入 validation report |

兩筆無效紀錄不參與評估，但相同問題仍有其他有效紀錄，因此保留 66 題。
這些參考回答並不是人工確認的 GT，後續評分要同時看原始來源。

沒有 PDF，故不需要 Docling 或 OCR。資料原本就有 source text 與句子 key。
Evidence 直接對應 source ID + 字元範圍，不需要硬建立 section tree。
`title` 只是來源第一句的衍生標題，並非猜測的 PDF 章節；`section_path=[]`。

102 段來源已由 Codex 讀取並標註主題，標註只看來源，不用問題或參考回答。
可在 `data/emanual/annotations/source_topics.jsonl` 看每段的 labels、理由與原文 excerpt。
chunk 暫時繼承來源 labels；這個選擇簡單，但包含很多主題的來源會讓 soft 分數較高，
後續要把這個影響與 topic-aware 本身的效果分開分析。

## 整體流程

```text
Offline（已完成）
固定版本的 E-Manual 原始資料
  → 來源去重與 stable ID
  → evidence key 精確對應原文 offset
  → Codex 來源標註：10 個 topic，可多標籤
  → 保留句子邊界的 chunking
  → E5 vectors + chunk/source/label validation

Experiment（等你確認才執行）
同一個 query
  ├─ baseline：全庫 BM25 top-20 + E5 top-20 → 兩路 RRF
  └─ Qwen 4B 的 Jev topic probabilities（只算一次）
       ├─ hard：argmax topic → 先縮小 corpus → BM25/E5 → 兩路 RRF
       └─ soft：全庫 BM25/E5 candidates → 加 topic rank → 三路 RRF

三種方法分別得到 ranked chunks
  → 計算 evidence recall@5、@10、@20
  → 只把各自 top-5 交給相同 Qwen 4B、相同 prompt 回答
  → 保存每題來源、完整參考答案、生成答案、errors、latency
  → 後續逐題語意評分（不算 lexical F1）
```

Soft 的計算固定為：

```text
topic_mass(chunk) = sum(p(topic) for topic in chunk.labels)

RRF score = (1/3)/(60 + BM25_rank)
          + (1/3)/(60 + E5_rank)
          + (1/3)/(60 + topic_rank)
```

缺少 BM25/E5 rank 的項目貢獻 0；topic 同分則同 rank。
候選集是 BM25 top-20 與 E5 top-20 的 union，最多 40 個。
Soft 無法救回 union 以外的證據，hard 則可能因錯選 topic 排除正確證據。

## Jev 是怎麼實作的

模型看到 query 與 `[1] ... [10]` topic 選項。程式計算 `1]`、`2]`、…、`10]`
完整後綴每個 token 的 conditional log probability，累加後再做 softmax。
所以程式輸出 keys 是 `"1"`、`"2"`、…、`"10"`，沒有 T01，也不是請模型寫機率。

這裡重現論文的完整候選 likelihood 公式。HTTP 版逐個 trie node 計分，每次
取得一個 decode step 的 logits；它尚未重現論文平行 shared-KV 的最佳化，
不能事先宣稱會比較快。完整說明見 `docs/jev-implementation.md`。

## 建議的實驗順序

1. 先確認 taxonomy、source labels、資料限制與三種方法的控制變數。
2. 核對腳本、Jev suffix 計分與 RRF；目前只做這一步，未開始實驗。
3. 核准後跑 retrieval-only：66 題 × 3 方法，比較 evidence recall。
4. 再跑 top-5 answer generation，逐題評分 correctness/completeness/faithfulness。
5. 分析 baseline→hard、baseline→soft 的 paired changes，並列出錯誤與失敗案例。

同一個 Qwen 4B 同時作 topic scorer 與答案模型，不下載另一個 classifier。
這次不加入 history rewrite 或 LambdaMART。若之後要比較一般生成式分類與
Jev 的速度，再作獨立 ablation，避免同時變更過多因素。

## 現在可以安全執行的檢查

```bash
cd topic-aware-rag-benchmark
.venv/bin/python scripts/run_experiment.py --config configs/local.yaml
.venv/bin/python -m pytest
.venv/bin/python -m ruff check src scripts tests
```

第一個指令只印計畫，不送 LLM request。正式實驗必須明確加 `--execute`；
詳細命令在 README。Windows 可以用 `.venv\Scripts\python.exe`。

結果會存於 `results/<run-id>/`，分開保存三種方法與 summary。新的 run 不覆蓋舊 run。
`answer_review.jsonl` 預留 score/reason；未評分前 accuracy 是 null。

## 結論可用到哪個程度

這是「supplied test passages」組成的小型 corpus；不是完整原始電視手冊庫，
也不是公司 15 份 PDF 的正式部署測試。可快速驗證演算法方向，但不能直接把結果
宣稱成 production accuracy 或官方 RAGBench benchmark 分數。
正式實驗前也還需要確認現有 llama.cpp server 能提供本計分器需要的 raw logits；
目前通過的是模擬 HTTP 與離線資料測試，沒有把模擬結果當成實際模型結果。
