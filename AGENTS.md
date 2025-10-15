# AGENTS.md

## Agent Setup & Environment

- The agent must look for `AGENTS.md` in the project root and ingest it before executing tasks.
- If `AGENTS.md` is missing, the agent should **auto-create** it (with default guardrails) and commit it.
- All Codex agents must run in an environment where `WRDS_USERNAME` and `WRDS_PASSWORD` are set as environment variables.

## WRDS Access Guardrails

- Agents must attempt a **probe query** via the `wrds` Python package to verify connection; if that fails or returns empty, abort.
- Use chunked queries (quarterly or monthly) for large tables; do not request entire TRACE in one SQL.
- Log failure reasons (JSON) with stage, error, details, timestamp under `_output/.../failure_log.json`.

## Plotting / Binary Output Constraints

- Agents must not import or call plotting libraries (`matplotlib`, `seaborn`, `plotly`, etc.).
- Any attempt to call `.savefig`, `.show`, or similar should abort with structured failure JSON.
- Agents may only produce CSV, JSON, or Markdown outputs.

## Event Ingestion Rules

- Agents fetch from official RSS/JSON/press release endpoints first; fallback to Google News RSS.
- Agents must normalize event dates, deduplicate, score confidence, and only include events with confidence ≥ 0.5.
- Map events to types (SLR, QE/QT, issuance, buyback, capital rules, etc.) and assign expected effect directions.

## Analysis & Modeling Rules

- Always compute:
  1. Mean/variance shift pre/post event.
  2. AR(1) persistence and implied half-life before/after.
  3. Breakpoint (Chow/Bai-Perron) alignment with events.
- Agents must also contextualize liquidity (ATS / capacity / futures flags) around events when data is available.

## Failure Contract

On any failure (WRDS access, HTTP fetch, parsing error, plot attempt), agent must return a JSON object structured:

```json
{
  "stage": "...",
  "error": "...",
  "details": { ... },
  "timestamp": "YYYY-MM-DDTHH:MM:SSZ"
}
