# Contributing

Thanks for helping improve the AI Knowledge Assistant. This guide gets you from fork
to merged pull request.

## Ground rules

- Be kind and constructive in issues and reviews.
- Keep pull requests focused: one bug fix or feature per PR.
- Never commit secrets or personal data. `.env` and `private/` are git-ignored; use `.env.example` for new variables.
- Retrieval or prompt changes need benchmark numbers (see below) — "feels better" isn't enough.

## Getting set up

1. **Fork** the repository on GitHub and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/<repo>.git
   cd <repo>
   git remote add upstream https://github.com/OWNER/REPO.git
   ```

2. **Create a virtual environment and install dev dependencies** (Python 3.11+):

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
   pip install -r requirements-dev.txt
   ```

3. **Configure** (only needed to run the server, not the tests):

   ```bash
   cp .env.example .env               # then fill in your own keys
   ```

## Before you push

Run the same checks CI runs:

```bash
ruff check .            # lint: PEP 8, imports, type hints, Google-style docstrings
ruff format .           # auto-format (CI runs `ruff format --check .`)
pytest                  # offline suite: no API keys, no model downloads
```

If you touched embeddings, retrieval, ranking or the knowledge-base pipeline, also run
the real-model tests (downloads ~80 MB once):

```bash
RUN_MODEL_TESTS=1 pytest                         # PowerShell: $env:RUN_MODEL_TESTS="1"; pytest
python -m benchmarks.evaluate --configs production
```

Paste the before/after benchmark line into your PR description.

## Code style

- **Type hints everywhere**, using modern syntax (`list[str]`, `dict[str, Any]`, `str | None`).
- **Google-style docstrings** for every public module, class and function (`Args:`, `Returns:`, `Raises:`). Tests are exempt.
- **Logging, not `print()`**, in application code — and never log secrets, question text or retrieved content.
- **External calls raise `ExternalServiceError` subclasses** (see `app/core/exceptions.py`); the central handler turns them into `503` + `Retry-After`. Don't add ad-hoc `try/except` in routers.
- **New settings** go in `app/core/config.py`, `.env.example` and the README configuration table.
- **New columns** go in `app/models/knowledge.py` (the admin grid adds them automatically) and `supabase/schema.sql`.
- **Admin UI** (`app/web/`) stays build-free: plain ES module, render user data with `textContent`, never `innerHTML`.
- Prefer small, boring code. Add an abstraction only when a second real use appears.

## Tests

- Put tests in the matching file: `tests/test_api_chat.py` / `tests/test_api_sync.py` / `tests/test_admin.py` for endpoints,
  `tests/test_services.py` for service logic (add to that service's section), `tests/test_benchmarks.py`
  for metrics and the quality gate. See [CODEBASE_MAP.md](CODEBASE_MAP.md) for where code belongs.
- Use the `fake_models` / `indexed_client` fixtures from `tests/conftest.py` for anything that embeds or searches — they keep tests fast and offline.
- A bug fix needs a test that fails without the fix.

## Submitting a pull request

1. Sync with upstream and branch off `main`:

   ```bash
   git fetch upstream && git checkout -b fix/short-description upstream/main
   ```

2. Commit with clear messages (e.g. `fix: reject non-ASCII webhook secrets with 401`).
3. Push to your fork and open a PR against `main`. Describe **what** changed and **why**,
   how you tested it, and benchmark numbers if retrieval behaviour changed.
4. CI must be green (lint, format, tests). A maintainer will review — please respond to
   feedback by pushing new commits rather than force-pushing over review history.

## Reporting bugs and security issues

- **Bugs:** open an issue with steps to reproduce, expected vs actual behaviour, and the
  `X-Request-ID` of a failing request if you have one.
- **Security vulnerabilities:** please don't open a public issue — contact the maintainer
  privately (see the GitHub profile) so it can be fixed before disclosure.
