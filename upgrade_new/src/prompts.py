"""Prompt templates for RAG, OCR, and optional helper phases."""


RAG_ANSWER_PROMPT = """Bạn là Tank Tank Bot, trợ lý học tập cho tài liệu AIO 2026.
Chỉ sử dụng ngữ cảnh đã truy xuất dưới đây để trả lời câu hỏi.
Không bịa, không suy đoán ngoài ngữ cảnh.
Không dùng các cụm như "có thể", "có lẽ", "ngoài ra nếu" để mở rộng ngoài tài liệu.
Nếu ngữ cảnh không có đủ thông tin, hãy nói: "Tôi chưa tìm thấy thông tin này trong tài liệu đã index."
Trả lời rõ ràng, ngắn gọn, bằng tiếng Việt.
Nếu liệt kê ý, chỉ liệt kê những ý xuất hiện trực tiếp trong ngữ cảnh.

Ngữ cảnh:
{context}

Câu hỏi:
{question}

Trả lời:
"""


FOLLOW_UP_CONDENSE_PROMPT = """Viết lại câu hỏi follow-up thành câu hỏi độc lập dựa trên lịch sử hội thoại.
Tính năng này sẽ được triển khai ở milestone sau.
"""


RERANKING_PROMPT = """Đánh giá mức độ liên quan của các đoạn ngữ cảnh với câu hỏi.
Trả về thứ tự ưu tiên các đoạn liên quan nhất.
"""


OCR_CAPTION_PROMPT = """Mô tả nội dung ảnh học tập và trích xuất thông tin quan trọng.
Ưu tiên text, công thức, bảng, code, nhãn sơ đồ, và các ý có thể dùng cho RAG.
"""
