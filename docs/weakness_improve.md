# Weakness & Improvement Plan - Project 1.2 RAG Chatbot

## 1. Tổng quan project

Project 1.2 yêu cầu xây dựng chatbot hỏi đáp tài liệu học tập bằng kiến trúc RAG (Retrieval Augmented Generation). Baseline trong tài liệu và code mẫu `full_code_output/chatbot_app_native.py` đang sử dụng:

- Giao diện: Streamlit.
- Đọc PDF: `pypdf`.
- Chunking: hàm `chunk_text()` tự viết, cắt theo độ dài ký tự với overlap.
- Embedding: Ollama model `bge-m3`.
- Vector database: ChromaDB với `chromadb.Client()` đang chạy in-memory.
- LLM sinh câu trả lời: Ollama model `vicuna:7b-v1.5-q5_1`.
- Chat history: lưu trong `st.session_state.chat_history` để hiển thị trên UI.

Mục tiêu cải tiến là biến baseline học tập này thành một ứng dụng demo RAG chắc chắn hơn, có thể triển khai trong khoảng thời gian từ ngày 19/06/2026 đến ngày 30/06/2026. Hướng đi phù hợp là "cân bằng": vẫn bám sát rubric AIO, nhưng có thêm các cải tiến thực tế về chất lượng truy xuất, độ tin cậy, trải nghiệm người dùng và khả năng kiểm thử.

## 2. Điểm yếu hiện tại

### 2.1. Data ingestion - Đọc và trích xuất tài liệu

Baseline dùng `pypdf` để đọc text từ PDF. Cách này đơn giản và phù hợp với tài liệu text có layout thẳng, nhưng có các hạn chế lớn:

- Bảng biểu có thể bị vỡ cấu trúc hàng/cột, làm mất quan hệ giữa cột và giá trị.
- Hình ảnh, biểu đồ, công thức, screenshot và nội dung scan có thể bị bỏ qua.
- Tài liệu có layout phức tạp, header/footer lặp lại hoặc nhiều cột có thể bị trích xuất sai thứ tự.
- Code hiện tại ghép text của tất cả trang thành một chuỗi lớn, chưa lưu metadata như `page_number`, `source_file`, `chunk_id`.

Tác động: chatbot có thể trả lời sai vì context ban đầu đã sai hoặc thiếu. Nếu không có metadata trang, người dùng không thể đối chiếu câu trả lời với tài liệu gốc.

### 2.2. Chunking - Cắt nhỏ văn bản

Hàm `chunk_text(text, size=1000, overlap=200)` cắt theo paragraph và độ dài ký tự. Cách này dễ hiểu nhưng chưa nhận biết cấu trúc ngữ nghĩa:

- Có thể cắt đứt một ý đang liền mạch nếu paragraph dài hoặc layout PDF bị trích xuất không tốt.
- Heading có thể nằm ở chunk này, nội dung nằm ở chunk khác, làm retrieval mất ngữ cảnh.
- Chunk size và overlap đang hard-code, chưa có cấu hình từ UI hoặc config.
- Chưa lưu relation giữa chunk và page/section nên khó debug khi retrieval lấy sai context.

Tác động: embedding của chunk có thể kém đại diện cho nội dung thật, dẫn đến truy xuất sai hoặc thiếu thông tin.

### 2.3. Vector database - Lưu trữ và quản lý dữ liệu

Code hiện tại dùng `chromadb.Client()` nên vector database nằm trong bộ nhớ tạm:

- Tắt app hoặc restart là mất database.
- Upload lại cùng một file sẽ phải đọc, chunk và embed lại từ đầu.
- Collection được tạo theo timestamp `rag_{int(time.time())}`, tránh trộn dữ liệu nhưng khó tái sử dụng.
- Chưa có file hash để nhận diện tài liệu trùng lặp.
- Chưa có cơ chế quản lý nhiều tài liệu hoặc xóa collection cũ.

Tác động: trải nghiệm dùng thực tế bị chậm, tốn RAM, tốn thời gian embedding và khó demo liên tục.

### 2.4. Retrieval - Tìm context liên quan

Baseline chỉ dùng vector search với `n_results=k`, mặc định `k=4`:

- Vector search tìm theo ngữ nghĩa tốt, nhưng có thể kém với keyword chính xác như tên model, mã lỗi, thuật ngữ, số liệu.
- `k=4` cố định không phù hợp mọi loại câu hỏi. Câu hỏi tổng hợp cần nhiều context hơn, câu hỏi đơn giản có thể cần ít hơn.
- Chưa có score threshold để nhận diện khi context không đủ liên quan.
- Chưa có reranking để sắp xếp lại các chunk trước khi đưa vào LLM.
- Chưa hiển thị điểm similarity/distance để debug chất lượng retrieval.

Tác động: LLM có thể nhận context sai, thiếu hoặc nhiều nhiễu, làm tăng nguy cơ hallucination.

### 2.5. Generation và chat memory

Prompt hiện tại có yêu cầu "nếu ngữ cảnh không có thông tin thì nói không biết", đây là điểm tốt. Tuy nhiên vẫn còn các hạn chế:

- Model `vicuna:7b-v1.5-q5_1` có khả năng tiếng Việt ở mức cơ bản, có thể trả lời lủng củng hoặc kém ổn định với câu hỏi học thuật.
- `chat_history` chỉ được dùng để hiển thị trên Streamlit, không được truyền vào `ollama.chat()`. Vì vậy chatbot không có conversation memory thực sự.
- Các câu hỏi nối tiếp như "phần đó thì sao?" hoặc "tóm tắt lại ý trên" có thể fail vì retrieval chỉ dựa vào câu hỏi mới.
- Câu trả lời chưa có citation/source, người dùng không biết thông tin lấy từ trang nào.
- Prompt chưa yêu cầu định dạng câu trả lời nhất quán, chưa bắt buộc chỉ dựa trên source.

Tác động: chatbot khó dùng trong hỏi đáp liền mạch và khó tạo niềm tin vì thiếu nguồn đối chiếu.

### 2.6. UI/UX - Trải nghiệm người dùng

Giao diện Streamlit baseline đã có upload PDF, nút xử lý, chat input và lịch sử hội thoại. Tuy nhiên:

- Chưa có thông tin cấu hình model đang dùng.
- Chưa có preview tài liệu, số trang, số chunk, thời gian xử lý, trạng thái Ollama/model.
- Chưa hiển thị source/citation bên dưới câu trả lời.
- Chưa có warning khi người dùng upload PDF không có text hoặc file quá lớn.
- Chưa có chức năng reset document/vector database rõ ràng.
- Chưa có setting cơ bản như `k`, model LLM, chunk size, overlap.

Tác động: người dùng khó hiểu hệ thống đang làm gì, khó tin kết quả và khó debug khi app lỗi.

### 2.7. Robustness và performance

Code baseline thích hợp để học, nhưng còn mong manh khi chạy thực tế:

- Chưa bắt lỗi khi Ollama chưa chạy, model chưa được pull, embedding fail, PDF lỗi, hoặc text extract rỗng.
- Embedding tất cả chunk trong một request có thể chậm hoặc lỗi với file lớn.
- Chưa có cache để tránh tính toán lại.
- Chưa có logging để truy lỗi.
- Chưa có cấu hình riêng cho đường dẫn database, model, chunking.
- Chưa xử lý trường hợp collection rỗng hoặc query khi chưa có tài liệu.

Tác động: app dễ crash trong demo và khó sửa lỗi nhanh.

### 2.8. Testing và evaluation

Project hiện tại chưa có bộ kiểm thử riêng:

- Chưa có unit test cho `chunk_text`, `process_pdf`, `retrieve`, `rag`.
- Chưa có golden question set để đánh giá chất lượng hỏi đáp.
- Chưa có metric đơn giản cho retrieval, ví dụ top-k có chứa trang đúng hay không.
- Chưa có test trường hợp lỗi: PDF rỗng, Ollama down, model missing, câu hỏi ngoài tài liệu.
- Chưa có checklist acceptance để biết khi nào cải tiến được xem là xong.

Tác động: khó đảm bảo các thay đổi sau này không làm hỏng luồng RAG cơ bản.

## 3. Phương án improve theo ưu tiên

### P0 - Bắt buộc nên làm trước 30/06

P0 là nhóm cải tiến tác động lớn, chi phí vừa phải, nên đưa vào sprint đầu tiên.

1. Chuyển ChromaDB sang persistent storage.
   - Dùng `chromadb.PersistentClient(path="./chroma_db")`.
   - Lưu database qua các lần restart.
   - Acceptance: restart app xong vẫn có thể dùng lại document đã xử lý nếu collection tồn tại.

2. Lưu metadata cho chunk.
   - Mỗi chunk cần có `source_file`, `page_number`, `chunk_id`.
   - Nếu chunk gồm nhiều trang, lưu range hoặc danh sách trang.
   - Acceptance: mỗi câu trả lời có thể hiển thị nguồn như "Trang 5, Trang 7".

3. Thêm citation/source vào câu trả lời.
   - Sau retrieval, hiển thị các chunk/source được sử dụng.
   - Không cần bắt LLM tự tạo citation phức tạp trong giai đoạn đầu; có thể hiển thị source ở UI bên dưới câu trả lời.
   - Acceptance: người dùng nhìn thấy ít nhất page number và tên file cho từng câu trả lời.

4. Thêm error handling cơ bản.
   - Bắt lỗi Ollama chưa chạy, model chưa có, PDF không trích xuất được text, upload rỗng.
   - Hiển thị thông báo thân thiện trên UI thay vì stack trace.
   - Acceptance: các lỗi phổ biến không làm app crash trang.

5. Cải thiện chat memory mức cơ bản.
   - Truyền một số message gần nhất vào `ollama.chat()` hoặc viết lại câu hỏi mới dựa trên lịch sử ngắn.
   - Giới hạn số message để tránh prompt quá dài.
   - Acceptance: chatbot trả lời được câu hỏi follow-up đơn giản như "phần này là gì?" sau một câu hỏi trước đó.

6. Cấu hình model và tham số RAG.
   - Đưa `LLM_MODEL`, `EMBED_MODEL`, `chunk_size`, `chunk_overlap`, `k` vào config hoặc sidebar.
   - Acceptance: có thể thay đổi model/k/chunking mà không sửa nhiều nơi trong code.

### P1 - Nên làm nếu P0 ổn định

P1 giúp project trông chuyên nghiệp hơn, nhưng vẫn khả thi trong thời gian ngắn.

1. Chunking thông minh hơn.
   - Tách text theo page trước, sau đó chunk theo paragraph/sentence.
   - Ưu tiên giữ heading gần với nội dung.
   - Acceptance: chunk không mất page metadata và ít cắt đứt câu hơn baseline.

2. File hash và cache.
   - Tính hash từ nội dung file PDF.
   - Nếu file đã xử lý, tải lại collection thay vì embed lại.
   - Acceptance: upload lại cùng file nhanh hơn và không tạo collection trùng lặp không cần thiết.

3. UI polish cho demo.
   - Hiển thị tên file, số trang, số chunk, model đang dùng, thời gian xử lý.
   - Thêm khu vực "Nguồn tham khảo" có thể expand để xem chunk.
   - Acceptance: người xem demo hiểu được pipeline đang chạy và có thể đối chiếu nguồn.

4. Bộ evaluation nhỏ.
   - Tạo file câu hỏi mẫu gồm câu hỏi có trong tài liệu, câu hỏi ngoài tài liệu, câu hỏi follow-up.
   - Ghi expected behavior thay vì phải có exact answer tuyệt đối.
   - Acceptance: có checklist kiểm thử trước khi nộp project.

5. Logging và debug mode.
   - Log số chunk, top-k source, distance/similarity nếu có.
   - Debug mode chỉ hiện khi cần.
   - Acceptance: khi câu trả lời sai, có đủ thông tin để xem retrieval lấy chunk nào.

### P2 - Mở rộng nếu còn thời gian

P2 nên để sau khi P0/P1 đã hoàn thành, tránh làm loãng scope trước deadline.

1. Hybrid search.
   - Kết hợp vector search với keyword/BM25 để bắt tốt tên riêng, mã số, thuật ngữ.
   - Acceptance: các câu hỏi cần exact keyword cho kết quả ổn định hơn vector-only.

2. Reranking.
   - Lấy top 10-20 chunk ban đầu, sau đó rerank để chọn top context cuối.
   - Acceptance: top context liên quan hơn trong bộ evaluation.

3. OCR/table extraction.
   - Thử `pdfplumber`, PyMuPDF hoặc OCR cho file scan/bảng biểu.
   - Acceptance: bảng đơn giản giữ được quan hệ hàng/cột tốt hơn `pypdf`.

4. Multi-document management.
   - Cho phép quản lý nhiều file, chọn document đang chat, xóa document.
   - Acceptance: người dùng có thể upload nhiều PDF mà không trộn ngữ cảnh ngoài ý muốn.

## 4. Bảng mapping weakness - improvement

| Weakness | Impact | Improvement | Priority | Acceptance Criteria |
|---|---|---|---|---|
| `pypdf` trích xuất kém với bảng, ảnh, layout phức tạp | Context đầu vào sai hoặc thiếu | Giữ `pypdf` cho MVP, thêm metadata trang; P2 thử `pdfplumber`/OCR cho bảng và scan | P0/P2 | Câu trả lời có source page; file có bảng được ghi nhận là case cần xử lý nâng cao |
| Không lưu metadata trang/source | Không thể đối chiếu nguồn | Lưu `source_file`, `page_number`, `chunk_id` vào Chroma metadata | P0 | Mỗi retrieved chunk có metadata đầy đủ |
| Chunking cắt theo ký tự/paragraph đơn giản | Mất ngữ cảnh, retrieval kém | Chunk theo page + paragraph/sentence, giữ heading gần nội dung | P1 | Chunk có kích thước hợp lý, không cắt đứt câu phổ biến, vẫn truy ngược được page |
| ChromaDB in-memory | Mất database khi restart | Dùng `chromadb.PersistentClient(path="./chroma_db")` | P0 | Restart app không mất vector database đã tạo |
| Upload trùng file vẫn embed lại | Tốn thời gian và RAM | Tính file hash, cache collection theo hash | P1 | Upload lại file cũ không tạo collection mới và nhanh hơn |
| Retrieval chỉ vector search với `k=4` cố định | Context có thể sai/thiếu | Cho cấu hình `k`, hiện source/distance, P2 hybrid search/rerank | P0/P2 | Có thể đổi `k`; top source hiện trên UI; evaluation cải thiện với keyword query |
| Chưa có score threshold | LLM có thể trả lời khi context yếu | Thêm logic cảnh báo khi retrieval score kém hoặc không có context | P1 | Câu hỏi ngoài tài liệu được trả lời "không có thông tin" ổn định hơn |
| `chat_history` chỉ để hiển thị, không đưa vào LLM | Follow-up question fail | Truyền lịch sử ngắn vào `ollama.chat()` hoặc condense question | P0 | Trả lời được ít nhất 3 câu follow-up mẫu |
| Model Vicuna tiếng Việt hạn chế | Câu trả lời có thể lủng củng | Cấu hình model để thử `qwen2.5`, `llama3.2`, `gemma2` tùy máy | P0 | Đổi model qua config/sidebar mà không sửa logic RAG |
| Không có citation | Người dùng khó tin câu trả lời | Hiện "Nguồn tham khảo" gồm file/page/chunk preview | P0 | Mỗi answer có danh sách source nếu retrieval thành công |
| UI tối giản | Khó demo và khó debug | Thêm thông tin file, số trang, số chunk, model, status xử lý | P1 | Sau upload, UI hiện đầy đủ thông tin tài liệu và trạng thái |
| Thiếu error handling | App dễ crash khi Ollama/PDF/model lỗi | Try/except với thông báo thân thiện | P0 | Ollama down, missing model, PDF rỗng không làm app crash |
| Embedding batch chưa tối ưu | File lớn có thể chậm/lỗi | Batch embedding theo nhóm chunk và hiện progress | P1 | File dài xử lý có progress và không fail vì request quá lớn |
| Chưa có test/evaluation | Khó biết improve có thật sự tốt | Tạo unit test và bộ câu hỏi evaluation nhỏ | P1 | Có checklist test trước khi nộp và pass các case chính |

## 5. Đề xuất scope thực hiện đến 30/06/2026

Với deadline ngắn, không nên làm tất cả P2. Scope hợp lý:

- Ngày 19/06 - 21/06: hoàn thiện P0 về persistent DB, metadata, citation, error handling.
- Ngày 22/06 - 24/06: thêm memory cơ bản, config model/tham số, cải thiện UI.
- Ngày 25/06 - 27/06: thêm chunking theo page, file hash/cache, evaluation set nhỏ.
- Ngày 28/06 - 29/06: test end-to-end, sửa lỗi, chuẩn bị demo.
- Ngày 30/06: đóng băng phiên bản nộp, viết README/hướng dẫn chạy nếu cần.

Kết quả mong đợi tối thiểu: app hỏi đáp PDF chạy ổn định trên local, có lưu vector database, có source citation theo trang, có xử lý lỗi cơ bản, có memory đơn giản, có checklist kiểm thử.

## 6. Acceptance criteria tổng

Project sau khi improve nên đạt các điều kiện sau:

- Upload một PDF text-based và xử lý thành công.
- Restart app không làm mất database nếu document đã được index.
- Mỗi câu trả lời dựa trên retrieval đều hiển thị nguồn tham khảo với tên file và trang.
- Câu hỏi ngoài nội dung tài liệu không bị bịa trả lời.
- Câu hỏi follow-up đơn giản có thể dùng lịch sử hội thoại ngắn.
- Nếu Ollama chưa chạy, model thiếu, PDF rỗng hoặc extraction fail, UI hiển thị thông báo lỗi rõ ràng.
- Có ít nhất một bộ câu hỏi evaluation nhỏ để test trước khi nộp.
- Scope vẫn giữ gần với rubric AIO: hiểu pipeline RAG, chunking, embedding, vector DB, retrieval, LLM generation và Streamlit UI.

## 7. Ghi chú cho bước backlog tiếp theo

Sau khi bạn duyệt tài liệu này, có thể phân ra Scrum backlog theo các epic:

- Epic 1: Core RAG reliability.
- Epic 2: Source citation và document metadata.
- Epic 3: Chat UX và memory.
- Epic 4: Evaluation và testing.
- Epic 5: Optional advanced retrieval.

Mỗi backlog item nên có các property như priority, sprint, story point, acceptance criteria, dependency, owner/status. Các priority trong tài liệu này có thể chuyển thẳng thành P0/P1/P2 cho backlog.
