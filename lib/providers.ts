// lib/providers.ts — brief parsing, catalog, adaptive router, gateway dial.
//
// The entry reads one brief (MAKEFUN_PROMPT_FILE): the house's sealed copy of
// OpenRouter's fourteen live :free models with their quality ratings and
// capability flags. Every routing decision below derives from that brief, the
// incoming request, and the gateway answers seen so far in this trial —
// nothing else. No wall clock, no unseeded randomness: a replay must make
// exactly the same calls and leave exactly the same state export.

export interface FreeModel {
  id: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface Catalog {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: string;
    models: CatalogModel[];
  }>;
}

export interface PromptRequirements {
  json: boolean;
  tools: boolean;
  vision: boolean;
}

export interface TrailRow {
  status: number;
  code: string;
  model: string | null;
}

// Routing penalties. A model is chosen by its quality rating first; every
// call it already served costs 1 milli (load spread), every refusal costs 15
// milli, and a model a refusal hit is cooled by 200 milli for the rest of
// the trial — enough to drop the 1475-rated model below the 1303-rated one
// after one refusal, so a gated model is tried once and then left alone.
export const CALL_PENALTY = 1;
export const FAILURE_PENALTY = 15;
export const COOLDOWN_MILLI = 200;

// The gateway refuses any request without an output cap, and any cap above
// 8192 tokens.
export const MAX_OUTPUT_TOKENS = 8192;
export const DEFAULT_MAX_TOKENS = 1024;

export function clampMaxTokens(value: unknown): number {
  const n = typeof value === "number" && Number.isFinite(value) ? Math.floor(value) : DEFAULT_MAX_TOKENS;
  if (n < 1) return 1;
  if (n > MAX_OUTPUT_TOKENS) return MAX_OUTPUT_TOKENS;
  return n;
}

// Extract the models table from the brief text. The brief carries it as one
// JSON object whose schema field reads "free-model-aggregator-02/models/v1".
// The scan is string-aware so braces inside quoted text cannot fool it.
export function parseBrief(text: string): FreeModel[] {
  const marker = '"schema":"free-model-aggregator-02/models/v1"';
  const at = text.indexOf(marker);
  if (at < 0) throw new Error("brief carries no free-model table");
  const start = text.lastIndexOf("{", at);
  if (start < 0) throw new Error("brief models table has no opening brace");
  let depth = 0;
  let inString = false;
  let escaped = false;
  let end = -1;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        end = i + 1;
        break;
      }
    }
  }
  if (end < 0) throw new Error("brief models table is not balanced JSON");
  const doc = JSON.parse(text.slice(start, end)) as { models?: unknown };
  if (!Array.isArray(doc.models)) throw new Error("brief models table has no models array");
  const out: FreeModel[] = [];
  for (const row of doc.models as Array<Record<string, unknown>>) {
    if (typeof row.id !== "string") continue;
    out.push({
      id: row.id,
      rating_milli: typeof row.rating_milli === "number" ? row.rating_milli : 1150,
      context_length: typeof row.context_length === "number" ? row.context_length : null,
      tools: row.tools === true,
      vision: row.vision === true,
      json: row.json === true,
    });
  }
  if (out.length === 0) throw new Error("brief models table is empty");
  return out;
}

// The catalog document: exactly the brief's free models, one provider,
// OpenRouter's public endpoint (never the gateway's trial address — the
// catalog is part of the replay-compared export). --list prints this same
// document, built from the same object, so the bytes match the register body.
export function buildCatalog(models: FreeModel[]): Catalog {
  return {
    providers: [
      {
        id: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models: models.map((m) => ({ id: m.id, free: true, context_length: m.context_length })),
      },
    ],
  };
}

export function catalogJson(models: FreeModel[]): string {
  return JSON.stringify(buildCatalog(models));
}

// Per-trial router memory. It is deliberately small: calls served, refusals,
// and which models a refusal cooled. All three only ever grow inside one
// trial, and the trial's prompts arrive one at a time, so the ranking a
// prompt sees is a pure function of what happened before it.
export interface RouterState {
  calls: Map<string, number>;
  failures: Map<string, number>;
  cooled: Set<string>;
}

export function newRouterState(): RouterState {
  return { calls: new Map(), failures: new Map(), cooled: new Set() };
}

export function routerScore(model: FreeModel, state: RouterState): number {
  const calls = state.calls.get(model.id) ?? 0;
  const failures = state.failures.get(model.id) ?? 0;
  const cooled = state.cooled.has(model.id) ? COOLDOWN_MILLI : 0;
  return model.rating_milli - CALL_PENALTY * calls - FAILURE_PENALTY * failures - cooled;
}

export function modelFits(model: FreeModel, reqs: PromptRequirements): boolean {
  if (reqs.json && !model.json) return false;
  if (reqs.tools && !model.tools) return false;
  if (reqs.vision && !model.vision) return false;
  return true;
}

// Candidates for one prompt, best first. Array sort is stable, so equal
// scores keep the brief's order — the ranking is fully deterministic.
export function rankCandidates(models: FreeModel[], reqs: PromptRequirements, state: RouterState): FreeModel[] {
  return models
    .filter((m) => modelFits(m, reqs))
    .sort((a, b) => routerScore(b, state) - routerScore(a, state));
}

export function recordServed(state: RouterState, modelId: string): void {
  state.calls.set(modelId, (state.calls.get(modelId) ?? 0) + 1);
}

export function recordRefusal(state: RouterState, modelId: string): void {
  state.failures.set(modelId, (state.failures.get(modelId) ?? 0) + 1);
  state.cooled.add(modelId);
}

export function trailCodeForStatus(status: number): string {
  return `http_${status}`;
}

// One model call through the grader's gateway, always on Chat Completions.
// The gateway swaps the trial's dummy key for the entrant's pushed
// OpenRouter key; the real key never enters this process. A transport
// failure (the gateway unreachable) comes back as status 0 so the caller
// can fail over exactly like a refused call.
export interface GatewayReply {
  status: number;
  body: string;
}

export async function dialGateway(
  openAiBaseUrl: string,
  apiKey: string,
  body: Record<string, unknown>,
): Promise<GatewayReply> {
  const url = `${openAiBaseUrl.replace(/\/$/, "")}/chat/completions`;
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: {
        authorization: `Bearer ${apiKey}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(28000),
    });
  } catch {
    return { status: 0, body: "" };
  }
  const text = await res.text();
  return { status: res.status, body: text };
}
