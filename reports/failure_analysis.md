# Failure Analysis — Flat RAG vs GraphRAG — Lab 19

**Học viên:** Nguyễn Đức Anh Tuấn (MHV: 2A202601618)
**Khóa học:** AICB-K34 · Track 3: GraphRAG

---

## Bảng tổng hợp Benchmark (LLM-as-a-Judge, kết quả cuối — 800 chunk trích xuất, 50 câu golden, `outputs/graphrag_vs_flatrag_summary.csv`)

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

---

## Phân tích 2 Ca lỗi Điển hình

### 1. Ca lỗi Flat RAG thất bại (GraphRAG thành công) — G5000-49 (multi-hop, Δ=+3.0)
- **Câu hỏi:** "Across the selected Samsung records, identify three distinct technology domains Samsung is connected to and the specific evidence for each."
- **Tại sao Flat RAG thất bại?** Theo rationale của judge: Flat RAG nhận diện đúng "semiconductor technology" nhưng **bịa thêm** "digital transformation services" và "application services" — 2 domain này không có trong context được retrieve, trong khi bỏ sót "display/biometric sensing" và "smart home" là 2 domain thật sự có evidence. Vector search trả về các chunk gần nghĩa nhưng rời rạc, khiến model tự suy diễn để lấp đầy "3 domains" thay vì bám sát bằng chứng.
- **GraphRAG đã giải quyết như thế nào?** GraphRAG xác định đúng cả 3 domain (Smart Home, Display, Semiconductor) kèm bằng chứng cụ thể cho từng domain — nhờ subgraph traversal gom các cạnh `USES`/`DEVELOPED` xuất phát từ node Samsung từ nhiều chunk khác nhau thành 1 ngữ cảnh có cấu trúc, thay vì để model tự suy luận từ các đoạn văn rời rạc.

### 2. Ca lỗi GraphRAG thất bại — G5000-26 (multi-hop, Δ=−4.0)
- **Câu hỏi:** "What external technology provider is named inside Amazon's July AI-service expansion, and what other new AI capability is mentioned alongside it?"
- **Nguyên nhân:** Đáp án đúng là **Cohere**, nhưng GraphRAG trả lời sai thành **Advanced Micro Devices Inc. (AMD)** — nhiều khả năng do seed-entity/graph traversal từ node "Amazon" đi lạc sang một cạnh AMD-liên-quan-tới-Amazon khác trong đồ thị (ví dụ tin AWS dùng chip AMD) thay vì đúng cạnh Amazon–Cohere của câu hỏi, do đồ thị đã canonical hoá "Amazon" thành 1 node duy nhất nối tới nhiều sự kiện AI khác nhau (degree=6) khiến BFS 2-hop lẫn lộn giữa các sự kiện. Flat RAG trả lời đúng hoàn toàn nhờ vector search khớp sát đoạn văn gốc nói về Cohere.
- **Đề xuất khắc phục:** Thêm `relation`/`event context` cụ thể hơn khi textualize subgraph (phân biệt các cạnh cùng xuất phát từ 1 node theo chủ đề/ngày), hoặc tăng trọng số ưu tiên cạnh có `published_date` khớp sát với thời điểm câu hỏi đề cập ("July") thay vì chỉ lấy N cạnh mới nhất chung chung.
