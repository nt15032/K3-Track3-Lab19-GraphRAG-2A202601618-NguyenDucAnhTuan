# Reflection & Action Plan — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Nguyễn Đức Anh Tuấn (MHV: 2A202601618)
**Khóa học:** AICB-K34 · Track 3: GraphRAG

---

## 1. Mapping Bài giảng vào Code
| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|-----------------------------|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()`, `run_coref()` | Hoạt động đúng thiết kế: khi mơ hồ, model trả về nguyên văn + log `unresolved_mentions` thay vì đoán bừa. |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` (cell 2.1) | Lọc cứng ngay trong `run_extraction()` trước khi ghi vào `raw_triples_df` — quan hệ/loại node không hợp lệ bị loại âm thầm (không log riêng), nên khó biết tỷ lệ bị lọc mất bao nhiêu. |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` (cell 2.3) | Dùng đúng `UNWIND $rows AS row` theo batch 1000, không insert từng dòng; `MERGE` theo `id` giúp chạy lại an toàn (idempotent) khi tôi phải insert lại sau khi đổi threshold entity resolution. |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, class `UF` | Threshold mặc định 0.90 quá chặt cho scale lab (chỉ 1 audit row) — phải hạ xuống 0.55 mới đủ minh bạch (168 rows sau khi gộp 2 vòng trích xuất). Bài học: threshold cần hiệu chỉnh theo mật độ trùng lặp thực tế của corpus, không nên copy nguyên số mặc định. |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()`, `SUPER_NODE_DEGREE`/`SUPER_NODE_EDGE_CAP` (cell 3.3) | Không kích hoạt tự nhiên ở scale lab (degree cao nhất = 7 << 100) — phải hạ tạm ngưỡng để demo cơ chế hoạt động đúng. |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()`, `judge_json()` (cell 4.2) | Hỗ trợ 2 provider (openai/groq) qua `JUDGE_PROVIDER` — hữu ích thực tế vì đã phải đổi provider giữa chừng khi OpenAI hết credit. |
| **Global Search / Community Summarization** | Bonus B | `build_communities()`, `summarize_communities()`, `answer_global()` | 78 community phát hiện được (NetworkX modularity), 76 community đủ lớn (≥3 node) được LLM tóm tắt — cho thấy corpus dù nhỏ vẫn có cấu trúc cụm rõ rệt. |
| **Self-Correction Retrieval** | Bonus C | `self_correcting_context()`, `context_sufficient()` | 65% câu hỏi mẫu cần fallback quá hop2 — chứng minh giá trị thực tế của cơ chế tự sửa so với retrieval tĩnh 1 tầng. |

---

## 2. Quá trình Debugging & Bài học
- **Lỗi kỹ thuật phức tạp nhất gặp phải:** Không phải lỗi crash (dễ thấy qua traceback), mà là lỗi **âm thầm** trong `extraction_source = chunks_df.head(EXTRACTION_MAX_CHUNKS)`: dòng này luôn lấy 400 chunk **đầu tiên** theo thứ tự bài viết, trong khi bộ Golden Dataset (50 câu, evidence rải từ dòng 33 đến dòng 4997 trong 5000 bài đầu) cần độ phủ toàn corpus. Vòng chạy đầu tiên "chạy thành công", không có exception nào — nhưng GraphRAG vẫn thua Flat RAG vì đồ thị chỉ phản ánh ~150-250 bài viết đầu tiên, bỏ sót gần như toàn bộ dữ liệu evidence thật. Đây là loại lỗi khó nhất: code chạy đúng cú pháp, không traceback, nhưng sai về mặt lấy mẫu dữ liệu (sampling bias).
- **Cách xử lý:** Chuyển từ lấy N chunk đầu tiên (`head()`) sang lấy N chunk **rải đều (stride sampling)** trên toàn bộ phần chưa trích xuất, giữ nguyên ngân sách LLM call nhưng tăng độ phủ thực chất trên corpus 5000 bài. Kèm theo đó là một chuỗi lỗi cấu hình nhỏ hơn nhưng liên tiếp gây mất thời gian: tên database Neo4j Aura không phải `"neo4j"` mà là ID instance (`d3cf92a3`), model Groq `llama-3.3-70b-versatile` đã bị ngừng cung cấp (404), và việc sửa secret trên Colab **không** tự nạp lại vào biến Python đang chạy — phải chạy lại cell config mỗi lần đổi secret. Bài học chung: luôn xác minh bằng dữ liệu thật (in ra biến, đếm số dòng, kiểm tra distribution) thay vì tin code "chạy không lỗi" là "chạy đúng".

---

## 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

- **Tên đồ án / Dự án:** Trợ lý tra cứu tin tức doanh nghiệp/công nghệ Việt Nam (mở rộng trực tiếp từ pipeline của lab này sang nguồn tin tiếng Việt).
- **Đặc thù bài toán & Lý do chọn giải pháp:** Bài toán tra cứu tin tức doanh nghiệp có cả câu hỏi factoid đơn giản ("Ai là CEO công ty X?") lẫn câu hỏi multi-hop/cross-doc thật sự cần nối nhiều thực thể qua thời gian (M&A theo chuỗi, đầu tư chéo giữa nhiều công ty, timeline sản phẩm). Theo đúng kết quả thực nghiệm (xem `failure_analysis.md`): GraphRAG chỉ đáng đầu tư khi tỷ lệ câu hỏi multi-hop/cross-doc đủ lớn **và** đồ thị được trích xuất đủ dày (bài học từ chính lab: ở `EXTRACTION_MAX_CHUNKS` nhỏ, GraphRAG thua cả ở nhóm multi-hop). Vì vậy lựa chọn kiến trúc **Hybrid** (Flat RAG làm baseline luôn chạy, GraphRAG bổ sung khi seed-entity match được node trong đồ thị) thay vì GraphRAG thuần, để tránh rủi ro "GraphRAG tốn hơn nhưng không thắng" đã quan sát được trong lab.
- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: `Company`, `Person`, `Product`, `Sector` (ngành nghề — thêm mới so với lab để hỗ trợ câu hỏi dạng "so sánh các công ty cùng ngành").
  - Relations: kế thừa `ACQUIRED`, `INVESTED_IN`, `FOUNDED`, `PARTNERED_WITH`, `LEADS` từ lab; thêm `OPERATES_IN` (Company→Sector) và `COMPETES_WITH` (Company→Company) để phục vụ câu hỏi so sánh đối thủ cạnh tranh.
- **Chiến lược xử lý Super-node & Entity Resolution:** Với tin tức tiếng Việt, tên công ty có nhiều biến thể hơn tiếng Anh (tên đầy đủ pháp lý, tên viết tắt, tên thương hiệu) — sẽ áp dụng đúng quy trình hiệu chỉnh threshold theo dữ liệu thật đã làm ở lab (đo tỷ lệ trùng lặp mention thực tế trước, không copy nguyên `threshold=0.90` mặc định), đồng thời mở rộng `MANUAL_ALIASES` cho các case rút gọn kiểu bỏ tiền tố mà lexical guard không xử lý được (giống case `Disney`/`Walt Disney` phát hiện trong lab). Với super-node: các "ông lớn" (VNG, FPT, Viettel...) chắc chắn sẽ vượt degree>100 khi scale thật (khác với lab do dữ liệu ít), nên giữ nguyên cơ chế cap 50-cạnh-mới-nhất nhưng cần theo dõi risk mất context lịch sử bằng cách bổ sung 1 tầng "community/timeline summary" riêng cho các super-node thay vì chỉ dựa vào N cạnh gần nhất (đúng như đã thử nghiệm ở Bonus B/community reports).

---

## 🎯 Tự đánh giá
| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4 | |
| Khả năng kiểm soát AI Coding Agent | 4 | |
| Chất lượng đồ thị tri thức xây dựng | 4 | |
| Khả năng phân tích và debug hệ thống | 4 | |
