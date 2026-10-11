# Runbook: run a campaign

```
python scripts/validate_config.py configs/<campaign>.yaml
uv run finswarm campaign configs/<campaign>.yaml --dry-run      # projected calls and USD
uv run finswarm campaign configs/<campaign>.yaml --provider mock --workers 2   # rehearsal
uv run finswarm campaign configs/<campaign>.yaml --workers 4     # real run
```

Registered (pre-registered) campaigns additionally need:

- `python scripts/check_prereg.py --strict` passes (`FROZEN` present, hash matches);
- clean working tree and the commit pushed (the manifest records the git sha);
- the config is inside the pre-registered grid.

Checks after the run: every expected run id has a manifest in `results/manifests/`, spend in
`results/cost_ledger.jsonl` matches the dry-run projection (within 20 percent), failed runs listed.
Failures: see `resume_after_crash.md` or run the `campaign-triage` workflow.

## API keys and lane health

Keys live only in the root `.env` file (never in chat, commits or manifests) and are read by
`src/finswarm/llm/keys.py`:

| Variable | Lane(s) | Notes |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | `anthropic_haiku`, `anthropic_sonnet` | Prefer a key created **inside a workspace** whose spend limit equals the cap (200 USD): the cap is then enforced server-side. |
| `ANTHROPIC_WORKSPACE_ID` | same | Optional. Only needed when the key is *not* scoped to a workspace: the API then answers 400 unless the `anthropic-workspace-id` header is sent (the client adds it from this variable). |
| `MISTRAL_API_KEY` | `mistral_free` | Free "Experiment" tier. Only the ministral family has a non-zero per-minute limit on this tier (ministral-8b-latest 188 req/min, ministral-14b 30, ministral-3b 750; mistral-small/medium/large and magistral answer 429 with a 0 req/min limit): lane model `ministral-8b` (alias), rpm 150. |
| `GEMINI_API_KEY` | `gemini_free` | OpenAI-compatible endpoint. |
| `OPENROUTER_API_KEY` | `openrouter_free`, `openrouter_llama` | `:free` slugs change often (check `GET /api/v1/models`; 50 req/day unfunded, 1,000 once 10 USD of credits are bought). `openrouter_llama` is PAID: meta-llama/llama-3.3-70b-instruct at 0.22 / 0.50 USD per MTok, lane budget 10 USD inside the 200 USD cap; it is the judge for Sonnet-arm runs and the third family. |

Health check (one tiny call per lane, no prompt content, keys redacted in the output):

```
uv run finswarm lanes
```

