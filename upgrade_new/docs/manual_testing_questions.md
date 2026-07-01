# Manual Testing Questions - OCR/Vision, Tables, Notion Hash Writeback

Use this checklist after starting Streamlit, syncing Notion, and optionally uploading a PDF with images/tables. Expected answers should be grounded in indexed sources and include citations.

| ID | Question / Action | Purpose | Data needed | Expected result |
| --- | --- | --- | --- | --- |
| T01 | Sync Notion, then check the sync status line. | Verify Notion sync summary counters. | Notion database configured. | Status shows pages, chunks, `tables`, `OCR units`, `hash written`, and `hash write failed`. If hash write is enabled and permission is correct, `hash written` should increase. |
| T02 | Open a synced Notion row and inspect the `Content Hash` property. | Verify Notion hash writeback. | `ENABLE_NOTION_HASH_WRITE=true` and Notion `Update content` permission enabled. | `Content Hash` contains a SHA-256-like hex value after successful sync. It should not be empty for indexed pages. |
| T03 | Click `Sync Notion` again after a successful sync. | Verify incremental skip by hash/last edited. | Previous sync already completed. | Most unchanged pages should be skipped or hash-unchanged. Chunks indexed should be low or zero unless Notion content changed. |
| T04 | `Tuần 1 học những bài nào?` | Test Notion metadata retrieval. | Notion lessons for week 1. | Answer lists relevant week 1 lessons and cites Notion sources. It should not cite `empty` or `Emty`. |
| T05 | `Tóm tắt bài Basic Python.` | Test Notion content retrieval and heading-aware chunks. | Basic Python lesson indexed. | Answer summarizes only indexed Basic Python content, with citation pointing to the Basic Python lesson/heading. |
| T06 | `Trong bài Basic Python, phần giới thiệu Python nói gì?` | Regression test for repeated heading bug. | Basic Python content indexed. | Source preview should show the heading once per chunk, not repeated before every bullet. |
| T07 | `Office Hours tuần 2 có những nội dung chính nào?` | Test broad Notion retrieval and MMR diversity. | M01W02 Office Hours indexed. | Answer combines multiple relevant chunks without repeating nearly identical context. |
| T08 | `Các bài liên quan đến Git/Github nói về điều gì?` | Test keyword + semantic hybrid retrieval. | Git/Github Notion lesson indexed. | Answer retrieves Git/Github lesson even if wording mixes Vietnamese and English. |
| T09 | Upload a PDF with a table, index it, then ask: `Bảng trong PDF này nói về những cột và giá trị nào?` | Test PDF table extraction with `pdfplumber`. | A text-based PDF containing tables. | Answer cites PDF page/table source. Source title should include page and table index if table chunk is retrieved. |
| T10 | Upload a scanned/image-only PDF page, index it, then ask: `Trang scan trong PDF này có nội dung gì?` | Test PDF page OCR/Vision. | Scanned or image-only PDF and `ENABLE_OCR=true`. | Answer uses OCR text. PDF status should show `OCR units > 0`; if Vision fails, status shows `Vision errors`. |
| T11 | Upload a PDF containing embedded images/diagrams, then ask: `Hình/diagram trong PDF giải thích gì?` | Test PDF image-block OCR/Vision. | PDF with embedded images and `ENABLE_OCR=true`. | Answer includes extracted image/diagram text if Gemini Vision succeeds, with PDF page citation. |
| T12 | Sync a Notion page that contains an image, then ask: `Hình trong bài đó mô tả nội dung gì?` | Test Notion image OCR/Vision. | Notion image block with URL accessible to the API. | Answer includes OCR/Vision extracted text or gracefully says not found if image URL/API failed. Debug metadata may show `ocr_provider` or `ocr_error`. |
| T13 | Sync a Notion page that contains a table, then ask: `Bảng trong bài này có những thông tin nào?` | Test Notion table rendering as markdown. | Notion table/table_row blocks. | Answer cites Notion content source with block type table. Retrieved source preview should look like a markdown table. |
| T14 | `Nội dung nào trong tài liệu nói về RAG pipeline hoặc retrieval?` | Test cross-source retrieval after uploading RAG PDF and syncing Notion. | Notion and RAG PDF indexed. | Answer may cite both PDF and Notion if both contain relevant material. Context should not be duplicated heavily. |
| T15 | `Hãy đưa ra thông tin không có trong tài liệu: lịch thi chính thức của năm sau là ngày nào?` | Test hallucination guardrail. | Any indexed data. | System should say it cannot find the information in indexed documents instead of inventing a date. |
| T16 | Turn on debug retrieval, ask `Tóm tắt bài Basic Python`, and expand sources. | Test retrieval debug and metadata enrichment. | Debug retrieval enabled. | Debug shows RRF/rerank/MMR fields plus metadata such as `source_type`, `heading_path`, `block_types`, and OCR/table metadata when applicable. |

## Pass Criteria

- No answer should cite `Notion:empty`, `Emty`, or blank metadata-only rows.
- Notion hash writeback should not block indexing if Notion rejects an update; it should increment `hash write failed`.
- OCR/Vision failures should not block PDF indexing or Notion sync.
- Table chunks should preserve page/title metadata and be readable as markdown tables.
- Out-of-document questions should produce a refusal/unknown answer grounded in the indexed corpus.
