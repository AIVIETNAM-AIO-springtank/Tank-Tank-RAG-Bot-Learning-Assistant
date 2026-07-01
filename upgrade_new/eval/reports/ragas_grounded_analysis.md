# Phân Tích Đánh Giá RAGAS Grounded

Ngày tạo: 2026-06-28

## Phạm Vi

Báo cáo này đánh giá RAG pipeline trên bộ kiểm thử grounded mới:

- Testset: `upgrade_new/eval/testset.grounded.jsonl`
- Gói báo cáo:
  - `upgrade_new/eval/reports/ragas_grounded_report.md`
  - `upgrade_new/eval/reports/ragas_grounded_report.csv`
  - `upgrade_new/eval/reports/ragas_grounded_report.json`
- Các pipeline được so sánh:
  - `vector`
  - `hybrid_rrf_rerank_mmr`
- Provider sinh câu trả lời và LLM judge trong lần chạy này: Cohere `command-r7b-12-2024`

## Cơ Chế Chấm Điểm

Điểm trong báo cáo được tính bằng thư viện RAGAS. Project không tự viết công thức chi tiết cho từng metric như `faithfulness`, `answer_relevancy`, `context_precision`, hoặc `context_recall`.

Luồng đánh giá:

1. Hệ thống chạy RAG pipeline để lấy context và sinh câu trả lời.
2. Project đưa `question`, `answer`, `retrieved_contexts` và `ground_truth/reference` vào `ragas.evaluate(...)`.
3. RAGAS dùng logic nội bộ của thư viện kết hợp với LLM judge để chấm điểm từng record.
4. Project lấy điểm RAGAS trả về, tính trung bình bằng công thức `sum(values) / len(values)`, rồi format ra báo cáo.

Ý nghĩa các metric:

- `faithfulness`: câu trả lời có được hỗ trợ bởi context đã truy xuất hay không.
- `answer_relevancy`: câu trả lời có liên quan và trả lời đúng trọng tâm câu hỏi hay không.
- `context_precision`: các context được truy xuất có liên quan và được xếp hạng tốt hay không.
- `context_recall`: context truy xuất có bao phủ đủ thông tin cần thiết trong ground truth hay không.

Lưu ý: code mặc định dùng Gemini nếu không cấu hình `RAGAS_JUDGE_PROVIDER`, nhưng report này ghi nhận `ragas_judge_model` là `command-r7b-12-2024`, nên lần chạy báo cáo này tương ứng với Cohere judge.

## Chỉ Số Cuối Cùng

| Pipeline | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
| --- | ---: | ---: | ---: | ---: |
| Vector only | 0.958 | 0.725 | 0.972 | 0.750 |
| Hybrid RRF + Cohere Rerank + MMR | 1.000 | 0.729 | 1.000 | 1.000 |

## Diễn Giải

Hybrid RRF + Cohere Rerank + MMR đang tốt hơn vector-only trên testset grounded:

- Faithfulness tăng từ `0.958` lên `1.000`.
- Context precision tăng từ `0.972` lên `1.000`.
- Context recall tăng từ `0.750` lên `1.000`.
- Answer relevancy gần như ngang nhau, quanh `0.73`.

Kết quả này ủng hộ hướng retrieval mới: dùng RRF để hợp nhất kết quả vector và keyword, dùng rerank để chọn ứng viên liên quan hơn, rồi dùng MMR để giảm trùng lặp và giữ độ đa dạng của context.

## Vấn Đề Còn Lại

- Answer relevancy chưa cao, chỉ quanh `0.73`. Nguyên nhân có thể là câu trả lời còn hơi dài hoặc quá bao quát so với ground truth ngắn.
- Metadata documents vẫn có một số row `empty`; nên lọc hoặc giảm ưu tiên metadata rỗng trong retrieval.
- Testset hiện mới có 6 câu, cần mở rộng lên 20-30 câu gồm fact, summary, comparison, table/code/math/image-placeholder nếu muốn đánh giá nghiêm túc hơn.
- Báo cáo hiện lưu `ragas_judge_model` nhưng chưa lưu rõ `ragas_judge_provider`; nên bổ sung field này để tránh nhầm giữa Gemini và Cohere.

## Khuyến Nghị

Dùng `hybrid_rrf_rerank_mmr` làm default retrieval pipeline cho bản upgrade hiện tại.

Bước tiếp theo nên làm:

1. Lọc metadata Notion rỗng hoặc title `empty` khỏi indexing, hoặc hạ weight metadata-only documents.
2. Mở rộng grounded testset theo từng chủ đề trong Notion/PDF.
3. Thêm per-source recall checks để biết mỗi câu hỏi cần đúng page/heading nào.
4. Khi Gemini có quota trở lại, chạy lại cùng testset để so sánh Cohere và Gemini ở cả vai trò generation/judge hoặc sử dụng đo benchmark của các LLM khác như GPT-4o...
5. Cập nhật code sinh report để ghi cả `ragas_judge_provider` và `ragas_judge_model`.
