Last Edit: Gemini CLI - 2026-03-08 - Motive: Initial audit for AGENTS.md compliance.

# ovos-chromadb-embeddings-plugin — Audit Report

## Documentation Status
- [ ] AGENTS.md Header Format
- [ ] QUICK_FACTS.md (Moved from docs/)
- [ ] FAQ.md (Moved from docs/)
- [ ] MAINTENANCE_REPORT.md
- [x] AUDIT.md
- [ ] SUGGESTIONS.md
- [ ] docs/index.md

## Technical Debt & Issues
- `[MAJOR]` **legal**: Missing LICENSE file
- `[MAJOR]` **tests**: No unit tests found
- `[MINOR]` **ci**: Action `pypa/gh-action-pypi-publish` pinned to `@master` (should be `@release/v1`)
- `[MINOR]` **ci**: Action `ad-m/github-push-action` pinned to `@master` (should be `@pinned ref`)
- `[INFO]` **packaging**: Uses setup.py (consider migrating to pyproject.toml)

## Next Steps
- Add Apache-2.0 LICENSE file
- Pin `pypa/gh-action-pypi-publish` to `@release/v1` instead of `@master`
- Pin `ad-m/github-push-action` to `@pinned ref` instead of `@master`
- Add unit tests in test/unittests/
- Migrate from setup.py to pyproject.toml
