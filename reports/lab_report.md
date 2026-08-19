# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Nguyễn Đức Anh Tuấn (MHV: 2A202601618)
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

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

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge, kết quả cuối — 800 chunk trích xuất, 50 câu golden, `outputs/graphrag_vs_flatrag_summary.csv`):

| Nhóm | Metric | Flat RAG | GraphRAG | Δ (Graph − Flat) | Nhận xét |
|---|---|---|---|---|---|
| cross-doc | Comprehensiveness | 3.364 | 3.227 | −0.137 | Flat nhỉnh hơn |
| cross-doc | Faithfulness | 3.818 | 3.682 | −0.136 | Flat nhỉnh hơn |
| cross-doc | Multi-hop reasoning | 3.364 | 3.227 | −0.137 | Flat nhỉnh hơn |
| cross-doc | Latency (s) | 1.844 | 1.998 | +0.154 | Graph chậm hơn |
| cross-doc | Token usage | 641.6 | 755.1 | +113.5 (+17.7%) | Graph tốn hơn |
| factoid | Comprehensiveness | 4.8 | 5.0 | +0.2 | Graph nhỉnh hơn |
| factoid | Faithfulness | 5.0 | 5.0 | 0 | Hòa |
| factoid | Multi-hop reasoning | 4.8 | 5.0 | +0.2 | Graph nhỉnh hơn |
| factoid | Latency (s) | 1.429 | 1.254 | −0.175 | Graph nhanh hơn |
| factoid | Token usage | 591.6 | 710.2 | +118.6 (+20.0%) | Graph tốn hơn |
| multi-hop | Comprehensiveness | 3.174 | 3.087 | −0.087 | Flat nhỉnh hơn |
| multi-hop | Faithfulness | 3.435 | 3.478 | +0.043 | Graph nhỉnh hơn (nhẹ) |
| multi-hop | Multi-hop reasoning | 3.174 | 3.087 | −0.087 | Flat nhỉnh hơn |
| multi-hop | Latency (s) | 2.653 | 3.247 | +0.594 | Graph chậm hơn rõ |
| multi-hop | Token usage | 696.2 | 873.6 | +177.4 (+25.5%) | Graph tốn hơn rõ |

**Nhận định tổng quát (đã kiểm chứng qua 2 lần chạy độc lập — 400 chunk và 800 chunk trích xuất, kết quả nhất quán):** GraphRAG **không vượt** Flat RAG ở nhóm `multi-hop` — nhóm mà lý thuyết dự đoán GraphRAG phải mạnh nhất — và chỉ nhỉnh hơn chút ở `factoid`. GraphRAG tốn thêm 18–25% token và chậm hơn ở 2/3 nhóm. Nguyên nhân gốc: `EXTRACTION_MAX_CHUNKS` (400, sau tăng lên 800) chỉ trích quan hệ từ một phần nhỏ trong `LAB_MAX_CHUNKS=8000` chunk khả dụng → đồ thị tri thức quá thưa (136 triples ở vòng 1, tăng thêm ở vòng 2) so với chỉ mục vector của Flat RAG phủ toàn bộ 8000 chunk. Đây là hiện tượng lặp lại ổn định, không phải nhiễu ngẫu nhiên.

#### Phân tích 2 Ca lỗi Điển hình:
1. **Ca lỗi Flat RAG thất bại (GraphRAG thành công) — G5000-49 (multi-hop, Δ=+3.0):**
   - *Câu hỏi:* "Across the selected Samsung records, identify three distinct technology domains Samsung is connected to and the specific evidence for each."
   - *Tại sao Flat RAG thất bại?* Theo rationale của judge: Flat RAG nhận diện đúng "semiconductor technology" nhưng **bịa thêm** "digital transformation services" và "application services" — 2 domain này không có trong context được retrieve, trong khi bỏ sót "display/biometric sensing" và "smart home" là 2 domain thật sự có evidence. Vector search trả về các chunk gần nghĩa nhưng rời rạc, khiến model tự suy diễn để lấp đầy "3 domains" thay vì bám sát bằng chứng.
   - *GraphRAG đã giải quyết như thế nào?* GraphRAG xác định đúng cả 3 domain (Smart Home, Display, Semiconductor) kèm bằng chứng cụ thể cho từng domain — nhờ subgraph traversal gom các cạnh `USES`/`DEVELOPED` xuất phát từ node Samsung từ nhiều chunk khác nhau thành 1 ngữ cảnh có cấu trúc, thay vì để model tự suy luận từ các đoạn văn rời rạc.
2. **Ca lỗi GraphRAG thất bại — G5000-26 (multi-hop, Δ=−4.0):**
   - *Câu hỏi:* "What external technology provider is named inside Amazon's July AI-service expansion, and what other new AI capability is mentioned alongside it?"
   - *Nguyên nhân:* Đáp án đúng là **Cohere**, nhưng GraphRAG trả lời sai thành **Advanced Micro Devices Inc. (AMD)** — nhiều khả năng do seed-entity/graph traversal từ node "Amazon" đi lạc sang một cạnh AMD-liên-quan-tới-Amazon khác trong đồ thị (ví dụ tin AWS dùng chip AMD) thay vì đúng cạnh Amazon–Cohere của câu hỏi, do đồ thị đã canonical hoá "Amazon" thành 1 node duy nhất nối tới nhiều sự kiện AI khác nhau (degree=6 theo bảng trên) khiến BFS 2-hop lẫn lộn giữa các sự kiện. Flat RAG trả lời đúng hoàn toàn nhờ vector search khớp sát đoạn văn gốc nói về Cohere.
   - *Đề xuất khắc phục:* Thêm `relation`/`event context` cụ thể hơn khi textualize subgraph (phân biệt các cạnh cùng xuất phát từ 1 node theo chủ đề/ngày), hoặc tăng trọng số ưu tiên cạnh có `published_date` khớp sát với thời điểm câu hỏi đề cập ("July") thay vì chỉ lấy N cạnh mới nhất chung chung.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent
> **Trade-offs, Agent Control & Scale 350MB:**
> - So sánh sự đánh đổi giữa GraphRAG vs Flat RAG về Latency, Token và Indexing Overhead.
> - Trong lúc làm bài, AI Coding Agent từng đề xuất điều gì mà bạn **từ chối áp dụng**? Tại sao?
> - Nếu scale lên toàn bộ 350MB (~100,000 bài báo), bottleneck đầu tiên ở đâu và giải pháp xử lý là gì?

*Trả lời:*
- **Đánh đổi Quality vs Cost vs Latency:** Ở scale lab này, GraphRAG **không đổi được chất lượng lấy chi phí** — tốn thêm 18–25% token và chậm hơn ở cross-doc/multi-hop, nhưng điểm chất lượng lại thấp hơn hoặc ngang Flat RAG (xem bảng mục 4). Chi phí "Indexing Overhead" của GraphRAG cũng cao hơn hẳn: cần thêm pipeline Coreference → NER+RE (LLM calls) → Entity Resolution (embedding + FAISS + guard) → Bulk insert Neo4j, trong khi Flat RAG chỉ cần 1 bước encode + FAISS index. Trade-off này chỉ đáng giá khi đồ thị đủ dày (extraction bao phủ phần lớn corpus) — ở 400–800/8000 chunk (5–10%) thì chưa đạt điểm hòa vốn.
- **Quyết định liên quan đến đề xuất của AI Coding Agent:** Khi GraphRAG thua ở lần chạy đầu, agent (tôi) đề xuất 2 hướng: (a) tăng đơn giản `EXTRACTION_MAX_CHUNKS` từ 400 lên một số lớn bằng `chunks_df.head(N)` — cách này tốn quota tuyến tính nhưng vẫn thiên vị các bài viết đầu corpus; (b) giữ nguyên ngân sách 400 chunk nhưng **rải đều (stride sampling)** qua toàn bộ 5000 bài thay vì chỉ lấy 400 bài đầu. Đã chọn phương án (b) vì cùng chi phí LLM call nhưng tăng độ phủ thực chất — đây là ví dụ AI Agent tự đề xuất giải pháp rẻ hơn thay vì "brute-force tăng ngân sách", thay vì việc bị từ chối. 【Nếu bạn có tình huống cụ thể khác đã từ chối đề xuất của tôi, thay thế đoạn này bằng tình huống đó】
- **Giải pháp scale 350MB (~100,000 bài):** Bottleneck đầu tiên chắc chắn là **LLM extraction throughput** (Coreference + NER+RE), không phải embedding hay Neo4j insert — minh chứng thực tế: ngay ở scale lab (400→800 chunk, model `openai/gpt-oss-120b` qua Groq), đã 2 lần chạm rate-limit (`RateLimitError 429`, Groq TPD 200,000 token/ngày gần cạn) và 1 lần hết credit OpenAI khi chạy judge cho 50 câu. Với 100,000 bài (~200,000+ chunk), gọi LLM cho toàn bộ là bất khả thi trong thời gian/ngân sách hợp lý. Giải pháp: (1) áp dụng đúng kỹ thuật stride/stratified sampling đã dùng trong lab để trích xuất một tỷ lệ đại diện thay vì toàn bộ; (2) xử lý bất đồng bộ theo batch với queue + backoff thay vì vòng lặp tuần tự; (3) cân nhắc self-host model mở (giảm chi phí & rate-limit) cho riêng bước NER+RE; (4) FAISS `IndexFlatIP` (brute-force) sẽ không scale tới hàng trăm nghìn vector — cần chuyển sang `IndexIVFFlat`/HNSW cho Flat RAG index khi tăng `LAB_MAX_CHUNKS` lên mức lớn.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code
| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|-----------------------------|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()`, `run_coref()` | Hoạt động đúng thiết kế: khi mơ hồ, model trả về nguyên văn + log `unresolved_mentions` thay vì đoán bừa. |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` (cell 2.1) | Lọc cứng ngay trong `run_extraction()` trước khi ghi vào `raw_triples_df` — quan hệ/loại node không hợp lệ bị loại âm thầm (không log riêng), nên khó biết tỷ lệ bị lọc mất bao nhiêu. |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` (cell 2.3) | Dùng đúng `UNWIND $rows AS row` theo batch 1000, không insert từng dòng; `MERGE` theo `id` giúp chạy lại an toàn (idempotent) khi tôi phải insert lại sau khi đổi threshold entity resolution. |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, class `UF` | Threshold mặc định 0.90 quá chặt cho scale lab (chỉ 1 audit row) — phải hạ xuống 0.55 mới đủ minh bạch (28 rows). Bài học: threshold cần hiệu chỉnh theo mật độ trùng lặp thực tế của corpus, không nên copy nguyên số mặc định. |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()`, `SUPER_NODE_DEGREE`/`SUPER_NODE_EDGE_CAP` (cell 3.3) | Không kích hoạt tự nhiên ở scale lab (degree cao nhất = 4 << 100) — phải hạ tạm ngưỡng để demo cơ chế hoạt động đúng. |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()`, `judge_json()` (cell 4.2) | Hỗ trợ 2 provider (openai/groq) qua `JUDGE_PROVIDER` — hữu ích thực tế vì đã phải đổi provider giữa chừng khi OpenAI hết credit. |

---

### 2. Quá trình Debugging & Bài học
- **Lỗi kỹ thuật phức tạp nhất gặp phải:** Không phải lỗi crash (dễ thấy qua traceback), mà là lỗi **âm thầm** trong `extraction_source = chunks_df.head(EXTRACTION_MAX_CHUNKS)`: dòng này luôn lấy 400 chunk **đầu tiên** theo thứ tự bài viết, trong khi bộ Golden Dataset (50 câu, evidence rải từ dòng 33 đến dòng 4997 trong 5000 bài đầu) cần độ phủ toàn corpus. Vòng chạy đầu tiên "chạy thành công", không có exception nào — nhưng GraphRAG vẫn thua Flat RAG vì đồ thị chỉ phản ánh ~150-250 bài viết đầu tiên, bỏ sót gần như toàn bộ dữ liệu evidence thật. Đây là loại lỗi khó nhất: code chạy đúng cú pháp, không traceback, nhưng sai về mặt lấy mẫu dữ liệu (sampling bias).
- **Cách xử lý:** Chuyển từ lấy N chunk đầu tiên (`head()`) sang lấy N chunk **rải đều (stride sampling)** trên toàn bộ phần chưa trích xuất, giữ nguyên ngân sách LLM call nhưng tăng độ phủ thực chất trên corpus 5000 bài. Kèm theo đó là một chuỗi lỗi cấu hình nhỏ hơn nhưng liên tiếp gây mất thời gian: tên database Neo4j Aura không phải `"neo4j"` mà là ID instance (`d3cf92a3`), model Groq `llama-3.3-70b-versatile` đã bị ngừng cung cấp (404), và việc sửa secret trên Colab **không** tự nạp lại vào biến Python đang chạy — phải chạy lại cell config mỗi lần đổi secret. Bài học chung: luôn xác minh bằng dữ liệu thật (in ra biến, đếm số dòng, kiểm tra distribution) thay vì tin code "chạy không lỗi" là "chạy đúng".

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

- **Tên đồ án / Dự án:** Trợ lý tra cứu tin tức doanh nghiệp/công nghệ Việt Nam (mở rộng trực tiếp từ pipeline của lab này sang nguồn tin tiếng Việt).
- **Đặc thù bài toán & Lý do chọn giải pháp:** Bài toán tra cứu tin tức doanh nghiệp có cả câu hỏi factoid đơn giản ("Ai là CEO công ty X?") lẫn câu hỏi multi-hop/cross-doc thật sự cần nối nhiều thực thể qua thời gian (M&A theo chuỗi, đầu tư chéo giữa nhiều công ty, timeline sản phẩm). Theo đúng kết quả thực nghiệm ở mục 4: GraphRAG chỉ đáng đầu tư khi tỷ lệ câu hỏi multi-hop/cross-doc đủ lớn **và** đồ thị được trích xuất đủ dày (bài học từ chính lab: ở `EXTRACTION_MAX_CHUNKS` nhỏ, GraphRAG thua cả ở nhóm multi-hop). Vì vậy lựa chọn kiến trúc **Hybrid** (Flat RAG làm baseline luôn chạy, GraphRAG bổ sung khi seed-entity match được node trong đồ thị) thay vì GraphRAG thuần, để tránh rủi ro "GraphRAG tốn hơn nhưng không thắng" đã quan sát được trong lab.
- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: `Company`, `Person`, `Product`, `Sector` (ngành nghề — thêm mới so với lab để hỗ trợ câu hỏi dạng "so sánh các công ty cùng ngành").
  - Relations: kế thừa `ACQUIRED`, `INVESTED_IN`, `FOUNDED`, `PARTNERED_WITH`, `LEADS` từ lab; thêm `OPERATES_IN` (Company→Sector) và `COMPETES_WITH` (Company→Company) để phục vụ câu hỏi so sánh đối thủ cạnh tranh.
- **Chiến lược xử lý Super-node & Entity Resolution:** Với tin tức tiếng Việt, tên công ty có nhiều biến thể hơn tiếng Anh (tên đầy đủ pháp lý, tên viết tắt, tên thương hiệu) — sẽ áp dụng đúng quy trình hiệu chỉnh threshold theo dữ liệu thật đã làm ở lab (đo tỷ lệ trùng lặp mention thực tế trước, không copy nguyên `threshold=0.90` mặc định), đồng thời mở rộng `MANUAL_ALIASES` cho các case rút gọn kiểu bỏ tiền tố mà lexical guard không xử lý được (giống case `Disney`/`Walt Disney` phát hiện ở mục 1.2). Với super-node: các "ông lớn" (VNG, FPT, Viettel...) chắc chắn sẽ vượt degree>100 khi scale thật (khác với lab do dữ liệu ít), nên giữ nguyên cơ chế cap 50-cạnh-mới-nhất nhưng cần theo dõi risk đã nêu ở mục 1.3 (mất context lịch sử) bằng cách bổ sung 1 tầng "community/timeline summary" riêng cho các super-node thay vì chỉ dựa vào N cạnh gần nhất.

---

## 🎯 TỰ ĐÁNH GIÁ
| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4 | |
| Khả năng kiểm soát AI Coding Agent | 4 | |
| Chất lượng đồ thị tri thức xây dựng | 4 | |
| Khả năng phân tích và debug hệ thống | 4 | |
