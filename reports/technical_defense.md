# Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Nguyễn Đức Anh Tuấn (MHV: 2A202601618)
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026

---

### 1. Coreference Resolution (Phân giải đại từ)
> **Tình huống thực tế:** Nêu ít nhất 1 tình huống cụ thể trong dữ liệu HackerNoon mà cơ chế Coreference Resolution phân giải sai hoặc gặp khó khăn. Hậu quả của nó đối với Knowledge Graph là gì?

*Trả lời:*
- **Ví dụ từ dữ liệu:** chunk `91fb385552ca075515cf::c0000` — gốc: *"Mobile First is a miserable slogan but people in the Nordics are far happier than **us** and they adopted this strategy."* → resolved: *"...far happier than **people in other regions** and they adopted..."*
- **Hiện tượng:** Model tự ý diễn giải đại từ nhân xưng `"us"` (không có tiền ngữ/antecedent rõ ràng là một thực thể cụ thể nào trong cùng chunk — chỉ là ngôi thứ nhất ngầm định của tác giả) thành cụm mơ hồ `"people in other regions"`, **vi phạm trực tiếp conservative rule** ("CHỈ resolve khi antecedent rõ ràng trong cùng chunk; nếu mơ hồ → giữ nguyên + log `unresolved_mentions`"). Nghiêm trọng hơn: field `unresolved_mentions` của chính dòng này vẫn liệt kê `["us"]` — tức **pipeline tự mâu thuẫn với chính nó**: nhận diện đúng là "us" mơ hồ (đưa vào audit list), nhưng vẫn viết đè `resolved_text` như thể đã resolve chắc chắn. Một chunk khác (`64bee0b980a5ab9eeb19::c0000`, "...technology available to **us**..." → "...available to **the team**...") lặp lại đúng pattern lỗi này, cho thấy đây là hiện tượng hệ thống chứ không phải case đơn lẻ.
- **Hậu quả đối với Graph:** Ở 2 case cụ thể này, hậu quả trực tiếp bị giới hạn vì `"people in other regions"`/`"the team"` không khớp `ALLOWED_NODE_TYPES` (Company/Person/Technology) nên bước NER+RE (2.1) nhiều khả năng không biến chúng thành node/edge sai. Nhưng **rủi ro tổng quát đúng như ASSIGNMENT.md cảnh báo** vẫn còn nguyên: nếu đại từ mơ hồ tương tự ("it", "the company") vô tình bị model gán nhầm sang **tên công ty cụ thể** (thay vì cụm chung chung như ở đây), hệ quả sẽ là **False Edge** — gán nhầm một sự kiện (M&A, đầu tư...) cho sai công ty, và vì `unresolved_mentions` không đáng tin (đã chứng minh tự mâu thuẫn ở trên) nên không thể dùng field này làm lưới an toàn để lọc case rủi ro trước khi đưa vào extraction.

---

### 2. Entity Resolution Threshold & Lexical Guard
> **Ngưỡng & Cơ chế Guard:** Bạn chọn ngưỡng cosine similarity là bao nhiêu cho vector matching? Trích dẫn 1 cặp thực thể có độ tương đồng vector cao ($> 0.85$) nhưng bị Lexical Guard chặn không cho gộp (Reject) và giải thích lý do.

*Trả lời:*
- **Ngưỡng cosine similarity:** `threshold = 0.55` (`top_k = 8`)
- **Vì sao không dùng mặc định 0.90:** Ở scale của lab, tập mention thu được có độ trùng lặp thực thể tự nhiên thấp (299 mention riêng biệt / 336 mention tổng, ~1.12 lần lặp/thực thể). Ở `threshold=0.90` (mặc định) chỉ bắt được **1 dòng audit** duy nhất; hạ dần 0.80→1 dòng, 0.65→8 dòng, 0.55→**168 dòng** (sau khi gộp 2 vòng trích xuất, `raw_triples_df` tăng lên 299 triples) — vượt xa mức minh bạch tối thiểu (≥10) theo yêu cầu RUBRIC. Đây là đánh đổi có chủ đích cho môi trường lab dữ liệu nhỏ — trong production với corpus lớn (nhiều biến thể tên thật xuất hiện lặp lại tự nhiên), nên giữ `threshold≈0.85–0.90` để tránh false-merge, vì hạ threshold sâu làm tăng số ứng viên đi qua `merge_guard()` (rủi ro merge nhầm nếu guard lỏng).
- **Ví dụ merge đúng (similarity cao, guard cho qua):** `Fidelity National Information Services` vs `Fidelity National Information Services Inc.` — similarity 0.9245 → `MERGE_VECTOR` (guard strip suffix "Inc." rồi so khớp chuỗi còn lại giống hệt → hợp lệ). Tương tự `Walt Disney Co.` vs `Walt Disney` — similarity 0.8449 → `MERGE_VECTOR`.
- **Cặp thực thể similarity cao nhưng bị Guard chặn (`REJECT_GUARD`):** `Disney` vs `Walt Disney` — similarity **0.8331** (thực tế cao nhất trong toàn bộ các cặp bị reject; dữ liệu lab không có cặp reject nào vượt 0.85, nên đây là ứng viên sát nhất và vẫn minh hoạ đúng cơ chế guard).
  - **Lý do bị chặn:** `merge_guard()` trước tiên strip corp-suffix (`Inc.`, `Corp.`...) — không áp dụng được ở đây vì "Walt" không phải hậu tố công ty mà là **tiền tố tên riêng**. Sau đó guard fallback sang `SequenceMatcher(None,"disney","walt disney").ratio()` ≈ 0.706 — dưới ngưỡng `0.72` nên bị từ chối. Đây thực chất là một **false-negative của guard** (rất có thể "Disney" và "Walt Disney" là cùng một công ty thật), không phải trường hợp guard ngăn nhầm-gộp 2 thực thể khác nhau — cho thấy giới hạn thật của thuật toán: `strip_suffix` chỉ xử lý được hậu tố dạng pháp lý (Inc/Corp/Ltd...), không xử lý được rút gọn kiểu bỏ tiền tố tên riêng (`Walt` trong `Walt Disney`), nên cần bổ sung alias thủ công cho các case này thay vì trông chờ vector+lexical guard tự phát hiện.

---

### 3. Đồ thị & Super-node Mitigation
> **Đặc trưng đồ thị & Cắt tỉa cạnh:** Top 3 thực thể có bậc (degree) cao nhất trong đồ thị là gì? Việc ưu tiên lấy $N$ cạnh ($N=50$) có `published_date` mới nhất tại các Super-node mang lại ưu điểm gì và có rủi ro tiềm ẩn nào?

*Trả lời:*
- **Top 3 Super-nodes:**

| Hạng | Tên thực thể | Loại thực thể (Type) | Bậc kết nối (Degree) |
|------|--------------|---------------------|----------------------|
| 1 | L&T Technology Services Limited | Company | 7 |
| 2 | Microsoft | Company | 6 |
| 3 | ServiceNow | Company | 6 |

(Nguồn: `graph_checks()` — cell 2.4, sau khi gộp 2 vòng trích xuất: đồ thị cuối có **390 nodes / 281 edges**, `invalid_provenance_edges = 0`.)

- **Kết quả thực nghiệm quan trọng:** Trên toàn bộ 50 câu evaluation, `graph_supernode_events = 0` — nghĩa là ngưỡng `SUPER_NODE_DEGREE=100` **chưa từng được kích hoạt tự nhiên** ở scale lab này (degree cao nhất quan sát được chỉ là 7, thấp hơn ngưỡng 100 tới 14 lần). Đã demo cơ chế cap bằng cách hạ tạm `SUPER_NODE_DEGREE` xuống dưới 7 để ép `test_supernode_policy()` chạy qua nhánh cap, xác nhận `assert len(edges) <= 50` pass, sau đó trả ngưỡng về 100.
- **Ưu điểm & Rủi ro của Temporal Mitigation (ưu tiên 50 cạnh mới nhất):**
  - *Ưu điểm:* Giữ ngữ cảnh sát với tình trạng hiện tại của quan hệ (vd trạng thái M&A đã "planned" hay "completed"), tránh bùng nổ context/token khi 1 node liên kết hàng nghìn cạnh (công ty lớn như Google/Microsoft trong dữ liệu thật).
  - *Rủi ro:* Câu hỏi cross-doc/lịch sử cần dữ liệu cũ (vd so sánh xu hướng đầu tư qua nhiều năm) có thể bị cắt mất nếu >50 cạnh liên quan đều nằm ngoài top-50-mới-nhất; cũng thiên vị các entity được đưa tin gần đây, làm giảm recall cho các sự kiện lịch sử quan trọng nhưng ít nhắc lại.

---

### 4. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent
> **Trade-offs, Agent Control & Scale 350MB:**
> - So sánh sự đánh đổi giữa GraphRAG vs Flat RAG về Latency, Token và Indexing Overhead.
> - Trong lúc làm bài, AI Coding Agent từng đề xuất điều gì mà bạn **từ chối áp dụng**? Tại sao?
> - Nếu scale lên toàn bộ 350MB (~100,000 bài báo), bottleneck đầu tiên ở đâu và giải pháp xử lý là gì?

*Trả lời:*
- **Đánh đổi Quality vs Cost vs Latency:** Ở scale lab này, GraphRAG **không đổi được chất lượng lấy chi phí** — tốn thêm 18–25% token và chậm hơn ở cross-doc/multi-hop, nhưng điểm chất lượng lại thấp hơn hoặc ngang Flat RAG (xem bảng chi tiết trong `failure_analysis.md`). Chi phí "Indexing Overhead" của GraphRAG cũng cao hơn hẳn: cần thêm pipeline Coreference → NER+RE (LLM calls) → Entity Resolution (embedding + FAISS + guard) → Bulk insert Neo4j, trong khi Flat RAG chỉ cần 1 bước encode + FAISS index. Trade-off này chỉ đáng giá khi đồ thị đủ dày (extraction bao phủ phần lớn corpus) — ở 400–800/8000 chunk (5–10%) thì chưa đạt điểm hòa vốn.
- **Quyết định liên quan đến đề xuất của AI Coding Agent:** Khi GraphRAG thua ở lần chạy đầu, agent (tôi) đề xuất 2 hướng: (a) tăng đơn giản `EXTRACTION_MAX_CHUNKS` từ 400 lên một số lớn bằng `chunks_df.head(N)` — cách này tốn quota tuyến tính nhưng vẫn thiên vị các bài viết đầu corpus; (b) giữ nguyên ngân sách 400 chunk nhưng **rải đều (stride sampling)** qua toàn bộ 5000 bài thay vì chỉ lấy 400 bài đầu. Đã chọn phương án (b) vì cùng chi phí LLM call nhưng tăng độ phủ thực chất — đây là ví dụ AI Agent tự đề xuất giải pháp rẻ hơn thay vì "brute-force tăng ngân sách", thay vì việc bị từ chối. 【Nếu bạn có tình huống cụ thể khác đã từ chối đề xuất của tôi, thay thế đoạn này bằng tình huống đó】
- **Giải pháp scale 350MB (~100,000 bài):** Bottleneck đầu tiên chắc chắn là **LLM extraction throughput** (Coreference + NER+RE), không phải embedding hay Neo4j insert — minh chứng thực tế: ngay ở scale lab (400→800 chunk, model `openai/gpt-oss-120b` qua Groq), đã 2 lần chạm rate-limit (`RateLimitError 429`, Groq TPD 200,000 token/ngày gần cạn) và 1 lần hết credit OpenAI khi chạy judge cho 50 câu. Với 100,000 bài (~200,000+ chunk), gọi LLM cho toàn bộ là bất khả thi trong thời gian/ngân sách hợp lý. Giải pháp: (1) áp dụng đúng kỹ thuật stride/stratified sampling đã dùng trong lab để trích xuất một tỷ lệ đại diện thay vì toàn bộ; (2) xử lý bất đồng bộ theo batch với queue + backoff thay vì vòng lặp tuần tự; (3) cân nhắc self-host model mở (giảm chi phí & rate-limit) cho riêng bước NER+RE; (4) FAISS `IndexFlatIP` (brute-force) sẽ không scale tới hàng trăm nghìn vector — cần chuyển sang `IndexIVFFlat`/HNSW cho Flat RAG index khi tăng `LAB_MAX_CHUNKS` lên mức lớn.

---

## 🎁 Bonus — Kết quả thực nghiệm

### A — Local/Global Query Router
`route_query()` chạy trên toàn bộ 50 câu Golden Dataset: **49 câu → local** (đồ thị chi tiết), **1 câu → global** (community report). Hợp lý vì phần lớn câu hỏi trong bộ golden (factoid/multi-hop) hỏi về thực thể/quan hệ cụ thể — chỉ 1 câu đủ mang tính tổng hợp/vĩ mô để cần tầng community.

### B — Global Search via Community Reports
`build_communities()` (NetworkX `greedy_modularity_communities`) phát hiện **78 community** trên đồ thị 390 nodes/281 edges; sau lọc `min_size≥3` còn **76 community** được LLM tóm tắt thành report riêng (ví dụ: *"L&T Technology Services Limited has developed advanced technologies and Engineering Research and Development services..."*). Demo `answer_global()` trên câu hỏi đầu tiên (G5000-01, về giao dịch Aeris–Ericsson) cho kết quả đáng chú ý: model trả lời **"evidence is insufficient"** — đúng dự đoán, vì câu hỏi này thuộc loại chi tiết/multi-hop nên tầng community report (tóm tắt ở mức thô) không đủ để trả lời chính xác, càng củng cố lý do cần có Router (mục A) để tránh route nhầm các câu chi tiết sang tầng global.

### C — Self-Correction Graph Retrieval
Chạy `self_correcting_context()` trên mẫu 20 câu đầu Golden Dataset:

| Route | Số câu | Tỷ lệ |
|---|---|---|
| `hop2` (đủ ngay) | 7 | 35% |
| `hop3` (chỉ riêng hop3 là đủ) | 0 | 0% |
| `hop3+vector` (phải fallback vector) | 13 | 65% |

**Định lượng trước/sau:** Nếu chỉ dùng baseline hop2 cố định (không có self-correction), **65% câu hỏi trong mẫu sẽ thiếu context**. Quan sát thú vị: không có câu nào dừng lại đúng ở mức hop3 — mọi trường hợp hop2 không đủ đều phải tiếp tục fallback sang vector, cho thấy ở scale đồ thị hiện tại (281 edges, thưa), riêng việc mở rộng thêm 1 hop không đủ bù đắp — bằng chứng thêm cho luận điểm ở mục 4: đồ thị cần dày hơn đáng kể để GraphRAG thuần phát huy hết giá trị; self-correction với vector fallback là cơ chế bù đắp cần thiết ở scale nhỏ.
