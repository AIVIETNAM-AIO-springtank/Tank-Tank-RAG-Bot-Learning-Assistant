# Codex Workflow - Tank Tank Bot

## 1. Purpose

This document defines the standard way to work with Codex for the Tank Tank Bot project. Every implementation request should be tied to one backlog Task ID and one clear folder scope.

Primary goals:

- Keep Codex aligned with the backlog.
- Avoid accidental changes outside the selected task.
- Preserve the separation between `baseline/` and `upgrade/`.
- Make every task reviewable, testable and easy to explain during submission.

## 2. Required Context For Every Codex Session

Before asking Codex to implement a task, provide or confirm:

- Current Task ID.
- Backlog CSV path.
- Expected folder scope.
- Files expected to change, if known.
- Acceptance criteria.
- Rule that Codex must not work on multiple tasks at once.

If the task is unclear, Codex must ask before editing files.

## 3. Opening Prompt Template

Use this prompt at the start of a new Codex session:

```text
Bạn đang làm việc trong project Tank Tank Bot.

Backlog CSV nằm tại:
- aio_rag_scrum_backlog_notion_v2_baseline_upgrade.csv

Hãy đọc backlog để hiểu các field:
- Task ID
- Summary
- Epic / Epic Name
- Issue Type
- Priority
- Labels
- Sprint / Sprint Date
- Deadline
- Time Estimate
- Description / Acceptance Criteria

Project có 2 bản logic:
1. baseline/: bản gốc theo đề bài PDF RAG Chatbot, dùng để đối chiếu và demo bản ban đầu.
2. upgrade/: bản cải tiến được build từ baseline, dùng để phát triển tính năng improve.

Quy tắc:
- Không tự ý làm nhiều task cùng lúc.
- Không sửa baseline/ nếu task không yêu cầu.
- Cải tiến kỹ thuật nằm trong upgrade/.
- Planning/docs nằm trong docs/.
- Không hard-code token, API key hoặc thông tin cá nhân.
- Code phải rõ ràng, có docstring, dễ giải thích khi nộp bài.

Nếu đã đọc và hiểu backlog, hãy trả lời:

YES

Project understanding:
* ...

Backlog understanding:
* ...

Working rules I will follow:
1. ...
2. ...
3. ...
4. ...
5. ...

Current status:
* Waiting for Task ID.
```

## 4. Backlog Understanding Confirmation Prompt

Use this prompt when you want Codex to confirm it has read and understood the backlog:

```text
Hãy đọc toàn bộ backlog CSV và xác nhận:

1. Tổng số row.
2. Các Issue Type có trong backlog.
3. Các sprint chính và deadline.
4. Quy tắc folder scope baseline/ vs upgrade/.
5. Cách bạn sẽ xử lý khi tôi đưa một Task ID.

Không sửa file trong bước này.
```

Expected Codex behavior:

- Read the backlog CSV.
- Summarize structure and current working rules.
- Wait for a Task ID.
- Do not edit files.

## 5. Task Execution Prompt Template

Use this template for every implementation or documentation task:

```text
Thực hiện Task ID: <TASK_ID>

Yêu cầu:
1. Đọc dòng task <TASK_ID> trong backlog CSV.
2. Tóm tắt lại mục tiêu task.
3. Nêu file dự kiến tạo/sửa.
4. Nêu acceptance criteria.
5. Chỉ thực hiện đúng phạm vi task này.

Quy tắc:
- Không làm task khác.
- Không sửa baseline/ nếu task không yêu cầu.
- Nếu là cải tiến kỹ thuật, sửa trong upgrade/.
- Nếu là planning/documentation, tạo trong docs/.
- Không hard-code token, API key hoặc thông tin cá nhân.
- Nếu task chưa rõ, hỏi lại trước khi sửa file.

Trước khi sửa file, hãy nói ngắn gọn:
- Mục tiêu task
- File dự kiến tạo/sửa
- Acceptance criteria

Sau khi làm xong, báo lại:
- File đã sửa
- Cách test/verify
- Acceptance criteria đã đạt hay chưa
- Có cần commit không
```

## 6. Folder Scope Rules

### `baseline/`

Purpose:

- Original PDF RAG Chatbot based on the assignment.
- Used for comparison and demo of the initial version.

Rules:

- Treat as read-only unless the current Task ID explicitly targets baseline.
- Do not add upgrade features here.
- Do not refactor baseline during unrelated tasks.

### `upgrade/`

Purpose:

- Improved implementation built from baseline.
- Contains technical improvements such as metadata, citation, persistent DB, Notion ingestion, memory and evaluation.

Rules:

- Put technical improvements here.
- Keep code modular and explainable.
- Use docstrings for public functions/modules.
- Keep model names, DB paths and RAG parameters configurable where practical.

### `docs/`

Purpose:

- Planning documents.
- Architecture notes.
- Workflow guides.
- Audit reports.
- Demo scripts.
- Evaluation notes.

Rules:

- Put documentation tasks here.
- Keep docs aligned with the selected Task ID.
- Avoid documenting features as done if they are only planned.

### Current Physical Folder Note

The backlog uses logical names `baseline/` and `upgrade/`. The current workspace may temporarily use `Baseline_Project/` and `Upgrade_Project/`. Until a folder-normalization task is executed, Codex must verify the actual path before editing.

## 7. Review Rules After Each Task

After completing a task, Codex must report:

- Task ID completed.
- Files created or changed.
- Whether code was changed.
- Test or verification command/check performed.
- Acceptance criteria status.
- Remaining limitations or follow-up tasks, if any.

For code tasks, also report:

- How to run the changed feature.
- Any dependency or environment requirement.
- Any known failure mode.

For documentation tasks, also report:

- Sections created.
- How the document maps to backlog acceptance criteria.

## 8. Commit Checklist

Before committing, verify:

- The commit covers one Task ID only.
- `git status --short` has no unrelated changes.
- No `.env`, token, API key, local vector DB, cache or installer is staged.
- Baseline was not modified unless required by the task.
- Tests or validation checks were run.
- Documentation reflects the actual state of the project.

Suggested commit message format:

```text
<TASK_ID>: <short summary>
```

Examples:

```text
AIO-S1-011: document Codex workflow
AIO-S2-003: persist Chroma vector store
```

## 9. Task Handoff Format

At the end of each task, use this concise format:

```text
Đã thực hiện <TASK_ID>.

Files changed:
- ...

Verification:
- ...

Acceptance criteria:
- ...

Notes:
- ...
```

## 10. Escalation Rules

Codex should ask before editing when:

- The Task ID does not exist in the backlog.
- The task scope conflicts with folder rules.
- The task requires secrets, tokens or private credentials.
- The expected file already exists and replacing it could destroy user work.
- The acceptance criteria are ambiguous enough to affect implementation.

Codex should not ask when:

- The needed information is discoverable from the backlog or repository.
- The task has a clear output path and acceptance criteria.
- A reasonable default is already documented in `docs/PROJECT_CONTEXT.md`.
