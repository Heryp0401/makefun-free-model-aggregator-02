// main.ts — one server, two API wires, one live free model behind every
// answer.
//
// The trial sets MAKEFUN_PROMPT_FILE (the brief: the fourteen free models
// with the house's ratings), MAKEFUN_ENV_URL (register + work), and the
// gateway's OPENAI_BASE_URL / ANTHROPIC_BASE_URL with the trial's dummy
// keys. The server listens on 0.0.0.0 at a free port it picks, registers
// {protocol_version: 1, port, catalog}, serves the env's six prompts one at
// a time — each answered by a real completion from a free OpenRouter model
// dialled through the gateway — and exits 0 when GET /work says done.
//
// `node main.ts --list` prints the catalog document and exits 0.
//
// Determinism: every choice derives from the brief, the requests and the
// gateway's answers. No wall clock and no unseeded randomness anywhere, so
// a replay leaves the same state export byte for byte.

import { readFileSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import {
  buildCatalog,
  catalogJson,
  dialGateway,
  newRouterState,
  parseBrief,
  rankCandidates,
  recordRefusal,
  recordServed,
  trailCodeForStatus,
  type FreeModel,
  type RouterState,
  type TrailRow,
} from "./lib/providers.ts";
import {
  anthropicErrorBody,
  anthropicJsonResponse,
  anthropicStreamResponse,
  errorMessageFromGateway,
  openAIErrorBody,
  openAIJsonResponse,
  openAIStreamResponse,
  parseProviderJson,
  parseProviderStream,
  requirementsOfAnthropic,
  requirementsOfOpenAI,
  toChatCompletions,
  type ClientWire,
  type ProviderAnswer,
} from "./lib/wire.ts";

const MAX_REQUEST_BYTES = 8 * 1024 * 1024;
const POLL_MS = 250;

function env(name: string): string | undefined {
  const value = process.env[name];
  return value === undefined || value === "" ? undefined : value;
}

function log(message: string): void {
  console.error(`[aggregator] ${message}`);
}

interface Served {
  status: number;
  body: string;
  stream: boolean;
  failures: TrailRow[];
}

// Serve one prompt: pick the best-rated free model that meets the request's
// requirements, dial it on the gateway, and translate the answer back into
// the client's wire. A refusal fails over to the next candidate (a 401 stops
// immediately — the key is missing or bad, and no model would answer), and
// every recovered refusal is reported in the x-makefun-upstream-failures
// header, the only way the failover column can see it.
async function servePrompt(
  wire: ClientWire,
  req: Record<string, unknown>,
  models: FreeModel[],
  state: RouterState,
  openAiBaseUrl: string,
  apiKey: string,
): Promise<Served> {
  const reqs = wire === "openai" ? requirementsOfOpenAI(req) : requirementsOfAnthropic(req);
  const stream = req.stream === true;
  const candidates = rankCandidates(models, reqs, state);
  const failures: TrailRow[] = [];
  for (const candidate of candidates) {
    const gatewayBody = toChatCompletions(wire, req, candidate.id, stream);
    const reply = await dialGateway(openAiBaseUrl, apiKey, gatewayBody);
    if (reply.status === 401) {
      failures.push({ status: 401, code: trailCodeForStatus(401), model: candidate.id });
      const message = errorMessageFromGateway(401, reply.body);
      return {
        status: 401,
        body: wire === "openai" ? openAIErrorBody(401, message) : anthropicErrorBody(401, message),
        stream: false,
        failures,
      };
    }
    if (reply.status < 200 || reply.status >= 300) {
      const status = reply.status === 0 ? 502 : reply.status;
      failures.push({ status, code: trailCodeForStatus(status), model: candidate.id });
      recordRefusal(state, candidate.id);
      continue;
    }
    const answer: ProviderAnswer | null = stream
      ? parseProviderStream(reply.body)
      : parseProviderJson(reply.body);
    if (answer === null) {
      failures.push({ status: 502, code: trailCodeForStatus(502), model: candidate.id });
      recordRefusal(state, candidate.id);
      continue;
    }
    recordServed(state, candidate.id);
    let body: string;
    if (wire === "openai") {
      body = stream ? openAIStreamResponse(answer, candidate.id) : openAIJsonResponse(answer, candidate.id);
    } else {
      body = stream ? anthropicStreamResponse(answer, candidate.id) : anthropicJsonResponse(answer, candidate.id);
    }
    log(`served ${wire}${stream ? " stream" : ""} via ${candidate.id}`);
    return { status: 200, body, stream, failures };
  }
  const message = "every free model refused this prompt";
  return {
    status: 502,
    body: wire === "openai" ? openAIErrorBody(502, message) : anthropicErrorBody(502, message),
    stream: false,
    failures,
  };
}

function modelsListOpenAI(models: FreeModel[]): string {
  return JSON.stringify({
    object: "list",
    data: models.map((m) => ({ id: m.id, object: "model", created: 0, owned_by: "openrouter" })),
  });
}

function modelsListAnthropic(models: FreeModel[]): string {
  return JSON.stringify({
    data: models.map((m) => ({
      id: m.id,
      type: "model",
      display_name: m.id,
      created_at: "2026-01-01T00:00:00Z",
    })),
    has_more: false,
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null,
  });
}

async function readJsonBody(req: IncomingMessage): Promise<Record<string, unknown> | null> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    const buf = chunk as Buffer;
    total += buf.length;
    if (total > MAX_REQUEST_BYTES) return null;
    chunks.push(buf);
  }
  const text = Buffer.concat(chunks).toString("utf8");
  if (text.trim() === "") return {};
  try {
    const value: unknown = JSON.parse(text);
    return typeof value === "object" && value !== null && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

function send(
  res: ServerResponse,
  status: number,
  body: string,
  stream: boolean,
  failures: TrailRow[],
): void {
  res.writeHead(status, {
    "content-type": stream ? "text/event-stream" : "application/json",
    "cache-control": "no-cache",
    ...(failures.length > 0
      ? { "x-makefun-upstream-failures": JSON.stringify(failures.slice(0, 8)) }
      : {}),
  });
  res.end(body);
}

async function postJson(url: string, payload: unknown): Promise<Record<string, unknown> | null> {
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(10000),
    });
    const text = await res.text();
    try {
      const value: unknown = JSON.parse(text);
      return typeof value === "object" && value !== null && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : null;
    } catch {
      return null;
    }
  } catch {
    return null;
  }
}

async function getJson(url: string): Promise<Record<string, unknown> | null> {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(10000) });
    const text = await res.text();
    try {
      const value: unknown = JSON.parse(text);
      return typeof value === "object" && value !== null && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : null;
    } catch {
      return null;
    }
  } catch {
    return null;
  }
}

async function main(): Promise<void> {
  if (process.argv.includes("--list")) {
    const briefPath = env("MAKEFUN_PROMPT_FILE");
    if (briefPath !== undefined) {
      const models = parseBrief(readFileSync(briefPath, "utf8"));
      process.stdout.write(`${catalogJson(models)}\n`);
    } else {
      // No brief, no catalog to print: an empty document, exit 0.
      process.stdout.write(`${JSON.stringify({ providers: [] })}\n`);
    }
    return;
  }

  const briefPath = env("MAKEFUN_PROMPT_FILE");
  const envUrl = env("MAKEFUN_ENV_URL");
  if (briefPath === undefined) throw new Error("MAKEFUN_PROMPT_FILE is not set");
  if (envUrl === undefined) throw new Error("MAKEFUN_ENV_URL is not set");
  const openAiBaseUrl = env("OPENAI_BASE_URL") ?? "https://openrouter.ai/api/v1";
  const apiKey = env("OPENAI_API_KEY") ?? "";

  const models = parseBrief(readFileSync(briefPath, "utf8"));
  const catalog = buildCatalog(models);
  const state = newRouterState();
  log(`brief parsed: ${models.length} free models`);

  const server = createServer((req: IncomingMessage, res: ServerResponse) => {
    void (async () => {
      try {
        const url = new URL(req.url ?? "/", "http://127.0.0.1");
        if (req.method === "GET" && url.pathname === "/v1/models") {
          const anthropic = req.headers["x-api-key"] !== undefined;
          const body = anthropic ? modelsListAnthropic(models) : modelsListOpenAI(models);
          send(res, 200, body, false, []);
          return;
        }
        if (
          req.method === "POST" &&
          (url.pathname === "/v1/chat/completions" || url.pathname === "/v1/messages")
        ) {
          const wire: ClientWire = url.pathname === "/v1/messages" ? "anthropic" : "openai";
          const body = await readJsonBody(req);
          if (body === null) {
            const message = "the request body is not a JSON object";
            send(
              res,
              400,
              wire === "openai" ? openAIErrorBody(400, message) : anthropicErrorBody(400, message),
              false,
              [],
            );
            return;
          }
          const served = await servePrompt(wire, body, models, state, openAiBaseUrl, apiKey);
          send(res, served.status, served.body, served.stream, served.failures);
          return;
        }
        send(res, 404, openAIErrorBody(404, "not found"), false, []);
      } catch (error) {
        const message = error instanceof Error ? error.message : "internal error";
        try {
          send(res, 500, openAIErrorBody(500, message), false, []);
        } catch {
          // The socket is already gone; nothing left to do.
        }
      }
    })();
  });

  await new Promise<void>((resolve) => {
    server.listen(0, "0.0.0.0", () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("the server did not bind a port");
  const port = address.port;
  log(`listening on 0.0.0.0:${port}`);

  const registerUrl = `${envUrl.replace(/\/$/, "")}/register`;
  let registered = false;
  for (let attempt = 0; attempt < 120 && !registered; attempt += 1) {
    const answer = await postJson(registerUrl, { protocol_version: 1, port, catalog });
    if (answer !== null && answer.ok === true) {
      registered = true;
      log(`registered (requests: ${String(answer.requests ?? "?")})`);
    } else if (attempt < 119) {
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  if (!registered) throw new Error("the env never accepted the register call");

  const workUrl = `${envUrl.replace(/\/$/, "")}/work`;
  await new Promise<void>((resolve) => {
    const timer = setInterval(() => {
      void (async () => {
        const work = await getJson(workUrl);
        if (work !== null && work.done === true) {
          clearInterval(timer);
          log("work done, exiting 0");
          server.close(() => resolve());
        }
      })();
    }, POLL_MS);
  });
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`[aggregator] fatal: ${message}`);
  process.exit(1);
});
