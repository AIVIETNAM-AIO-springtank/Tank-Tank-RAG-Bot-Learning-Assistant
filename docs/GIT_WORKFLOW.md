# Git Workflow

## Branch Strategy

Use a simple branch strategy for Tank Tank Bot:

- `main`: stable submission/demo branch.
- `dev`: integration branch for task work before finalizing.
- Optional task branches: `task/<task-id>-short-summary` when a change is risky or large.

For this small coursework project, work may happen directly on `main` during setup tasks, but each backlog Task ID should still end with a clear commit.

## Commit Convention

Use short, task-based commit messages:

```text
<Task ID>: <short summary>
```

Examples:

```text
AIO-S1-004: initialize git workflow
AIO-S2-003: persist Chroma vector store
```

## Folder Rules

- Do not modify baseline/reference code unless the selected Task ID explicitly requires it.
- Put technical improvements in `upgrade/` or the current physical equivalent.
- Put planning and documentation in `docs/`.
- Do not commit `.env`, tokens, local vector databases, cache folders or local installers.

## Pre-Commit Checklist

- Run the relevant test or validation command.
- Check `git status --short`.
- Confirm no secrets or local database files are staged.
- Confirm the commit scope matches exactly one Task ID.
