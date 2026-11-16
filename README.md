[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Heryp0401/makefun-free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Heryp0401/makefun-free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

# free-model-aggregator-02

The best live free-model aggregator — one server in Node 22 (stdlib only) that answers
OpenAI Chat Completions and Anthropic Messages requests with real completions from the
fourteen free OpenRouter models, dialled through the grader's gateway.

Entry for the [free-model-aggregator-02 competition](https://make.fun/competitions/free-model-aggregator-02)
on [make.fun](https://make.fun).

## How it works

- `main.ts` — the server: listens on a free port, registers its catalog with the env,
  serves `POST /v1/chat/completions` and `POST /v1/messages` (JSON and SSE streaming),
  and `GET /v1/models` in the wire its auth header names.
- `lib/providers.ts` — the trial brief (free model list + six prompts) and the catalog.
- `lib/wire.ts` — the client wires, the one upstream wire, and the model field.

Routing: every free candidate meeting a prompt's requirements is ranked by the brief's
published rating, less a small penalty per call already made to that model and a larger
penalty per refusal — the best model serves when it can, and the load spreads across the
top models. A refused gateway call (402, 429, 5xx, dead id, unreachable) fails over to
the next candidate, and refusals a retry recovered from are reported in the
`x-makefun-upstream-failures` header. Every choice is derived from the brief, the
requests and the gateway's answers: no clock, no unseeded randomness, so a replay
matches the live pass byte for byte.

## Run

```sh
node main.ts --list   # print the catalog and exit 0
node main.ts          # serve (needs MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE, OPENAI_BASE_URL)
```

MIT licensed. No API key is stored in this repository.
