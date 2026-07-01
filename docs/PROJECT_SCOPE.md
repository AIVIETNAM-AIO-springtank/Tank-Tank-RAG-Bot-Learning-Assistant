# Project Scope - Tank Tank Bot

## 1. Task Information

- Task ID: `AIO-S1-001`
- Summary: Chốt project vision và scope nộp bài
- Epic: `EPIC-01` - Architecture, Baseline & Upgrade Separation
- Sprint: Sprint 1 - Architecture, Baseline & Upgrade
- Sprint date: `2026-06-20`
- Task deadline: `2026-06-21`
- Final project deadline: `2026-06-30`
- Issue type: Story
- Priority: P0

## 2. Project Vision

Tank Tank Bot là một trợ lý học tập dùng kiến trúc RAG (Retrieval Augmented Generation) để hỏi đáp dựa trên tài liệu học tập của AIO 2026.

Project không chỉ dừng ở PDF chatbot theo đề bài ban đầu. Mục tiêu nộp bài là có một baseline đúng yêu cầu đề, sau đó phát triển bản upgrade có cấu trúc rõ ràng hơn, dễ demo hơn và cải thiện các điểm yếu quan trọng của baseline.

## 3. Problem

Người học có nhiều tài liệu học tập như PDF bài giảng, ghi chú, nội dung Notion hoặc dữ liệu học tập có cấu trúc. Việc đọc thủ công toàn bộ tài liệu để tìm câu trả lời mất thời gian và khó duy trì khi tài liệu tăng lên.

Project giải quyết vấn đề này bằng cách:

- Cho phép người dùng đưa tài liệu vào hệ thống.
- Cắt nhỏ nội dung thành các chunk có thể truy xuất.
- Tạo embedding và lưu vào vector database.
- Truy xuất các đoạn liên quan khi người dùng đặt câu hỏi.
- Dùng LLM để tạo câu trả lời tiếng Việt dựa trên context tìm được.
- Hiển thị nguồn tham khảo để người dùng đối chiếu lại tài liệu gốc.

## 4. Users

Người dùng chính:

- Học viên AIO 2026 muốn tra cứu nhanh nội dung học tập.
- Người demo hoặc chấm bài muốn thấy rõ sự khác biệt giữa baseline và upgrade.
- Developer của project muốn có công cụ hỗ trợ ôn tập, kiểm thử RAG và trình bày project mạch lạc.

Nhu cầu chính của user:

- Upload hoặc đồng bộ tài liệu học tập.
- Đặt câu hỏi bằng tiếng Việt.
- Nhận câu trả lời ngắn gọn, đúng trọng tâm.
- Biết câu trả lời dựa trên nguồn nào.
- Có thể demo được bản gốc và bản cải tiến trong thời gian ngắn.

## 5. Data Sources

### MVP

- PDF học tập upload từ local.
- Đây là nguồn dữ liệu bắt buộc vì bám sát đề bài Project 1.2 PDF RAG Chatbot.

### P1

- Notion CSV export hoặc dữ liệu học tập có cấu trúc.
- Mục tiêu là mở rộng project từ PDF chatbot thành Tank Tank Bot.
- Dữ liệu có thể gồm title, week, date, module, lecturer, label, URL hoặc page ID.

### P2

- Notion API sync hoặc incremental sync nếu còn thời gian.
- Mục tiêu là đồng bộ dữ liệu học tập tự động hơn, nhưng không phải điều kiện bắt buộc cho bản nộp tối thiểu.

## 6. Project Versions

Project gồm 2 bản logic:

### `baseline/`

`baseline/` là bản gốc theo đề bài PDF RAG Chatbot. Bản này dùng để:

- Đối chiếu với yêu cầu đề bài.
- Demo phiên bản ban đầu.
- So sánh với các cải tiến trong upgrade.

Quy tắc:

- Không tự ý sửa baseline nếu task không yêu cầu.
- Không trộn code upgrade vào baseline.
- Baseline cần giữ được vai trò reference version.

### `upgrade/`

`upgrade/` là bản cải tiến được build từ baseline. Bản này dùng để:

- Module hóa flow RAG.
- Cải thiện độ tin cậy của retrieval và generation.
- Thêm metadata, source citation, persistent vector database, error handling và evaluation.
- Mở rộng sang Notion hoặc nguồn học tập có cấu trúc nếu còn thời gian.

Quy tắc:

- Các cải tiến kỹ thuật nằm trong upgrade.
- Code cần rõ ràng, có docstring và dễ giải thích khi nộp bài.
- Không hard-code token, API key hoặc thông tin cá nhân.

Ghi chú hiện trạng workspace: repository đã chuẩn hóa theo hai thư mục chính `baseline/` và `upgrade/`. Các artifact cũ hoặc không thuộc cấu trúc chính được gom vào `others/`.

## 7. Scope

### MVP - Bắt buộc cho bản nộp

MVP cần đủ để chứng minh project hoạt động end-to-end và bám sát rubric AIO:

- Baseline chạy được theo đề bài PDF RAG Chatbot.
- Upgrade có cấu trúc module hóa cơ bản.
- Upload và xử lý PDF local.
- Chunk nội dung, tạo embedding và lưu vào vector database.
- Truy xuất context liên quan cho câu hỏi.
- Sinh câu trả lời tiếng Việt dựa trên context.
- Persistent vector database để không mất dữ liệu khi restart.
- Metadata/source citation để đối chiếu câu trả lời với tài liệu.
- Error handling cơ bản cho PDF rỗng/lỗi, Ollama chưa chạy hoặc model thiếu.
- Documentation đủ để người khác hiểu cách chạy và phạm vi project.

### P1 - Nên làm nếu MVP ổn định

P1 giúp project chuyên nghiệp hơn nhưng vẫn giữ scope hợp lý trước deadline:

- Chat memory ngắn để xử lý câu hỏi follow-up đơn giản.
- Sidebar/config cho model, `k`, chunk size và chunk overlap.
- File hash/cache để tránh index lại cùng một tài liệu.
- Evaluation set nhỏ gồm câu hỏi trong tài liệu, ngoài tài liệu và follow-up.
- UI hiển thị trạng thái xử lý, số chunk, nguồn tham khảo và debug retrieval khi cần.
- Notion CSV export hoặc data source có cấu trúc nếu MVP đã ổn định.

### P2 - Mở rộng nếu còn thời gian

P2 chỉ làm khi MVP và P1 không còn rủi ro blocking:

- Hybrid search kết hợp vector search và keyword/BM25.
- Reranking để chọn context tốt hơn.
- OCR hoặc table extraction cho PDF scan/bảng phức tạp.
- Multi-document management.
- Notion API sync hoặc incremental sync hoàn chỉnh.

## 8. Out Of Scope Before 30/06/2026

Các phần sau không nằm trong scope bắt buộc trước deadline:

- Production authentication hoặc phân quyền nhiều user.
- Cloud deployment bắt buộc.
- Multi-user database phức tạp.
- Fine-tuning LLM hoặc embedding model.
- Xây dựng UI production-grade ngoài phạm vi demo.
- Tự động xử lý hoàn hảo mọi loại PDF scan, bảng phức tạp hoặc ảnh.
- Hard-code token, API key hoặc thông tin cá nhân.
- Mở rộng scope vượt backlog hiện tại nếu chưa hoàn thành MVP/P1.

## 9. Definition Of Done For Scope

Task `AIO-S1-001` được xem là hoàn thành khi:

- Problem được mô tả rõ ràng.
- User chính và nhu cầu của user được xác định.
- Data sources được chia theo MVP/P1/P2.
- Project xác nhận có 2 bản logic: `baseline/` và `upgrade/`.
- Scope MVP/P1/P2 được giới hạn rõ.
- Out-of-scope trước `2026-06-30` được nêu rõ để tránh mở rộng quá deadline.
- Tài liệu được lưu trong `docs/`.
- Không có code nào bị sửa trong task này.
