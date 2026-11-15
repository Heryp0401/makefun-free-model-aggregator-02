// free-model-aggregator-02: one server, two client wires, a real free model behind every answer.
//
// This entry reads the trial's brief, builds one catalog of the fourteen free OpenRouter models,
// then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//      x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//      /v1/messages, each with a JSON body or with server-sent events;
//   4. answers each prompt by dialling a free model through the grader's gateway on Chat Completions
//      (the only route these ids have), translating Anthropic-shaped requests first and the answer
//      back, and never answering with its own text;
//   5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//      process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
// body's catalog must be that same document. Build it once, and use the one object.
//
// THE ROUTING (what wins the score)
//   - quality (60%): every free candidate meeting the request's requirements is ranked by the brief's
//     rating_milli, so the best model serves when it can. The rating gaps at the top are small, so a
//     little load-spreading costs almost no quality and buys served prompts.
//   - load (served 15% + failover 10%): each candidate's score is its rating less a penalty per call
//     already made to that model (1 milli) and a larger penalty per call that model refused
//     (15 milli). A model that just failed drops down the order, and a model already used several times
//     yields to a fresh one of nearly the same rating, so the trial's six prompts spread across the
//     top models instead of draining one until its budget is spent. A model that answers 429 is
//     cooling down: OpenRouter's free rate limit lasts about a minute, longer than the trial's
//     remaining prompts, so a 429'd model takes a further 200-milli penalty and the next prompt goes
//     straight to a fresh model of nearly the same rating instead of spending a call re-proving the
//     limit — the cooled model stays in the list as a late fallback, in case the limit already
//     cleared.
//   - failover (10%): a gateway call answered 402, 403, 429, 5xx, model_not_allowed or unreachable is
//     retried on the next candidate in score order, until one answers 2xx or the candidates run out.
//     A 403 (or a 400 model_not_allowed) hard-excludes that model for the rest of the trial: a gated
//     model never recovers within a trial, and a penalty too small to demote the top-rated model
//     would just re-dial it on the next prompt. The loop stops at the first success. Every refusal a retry recovered from is reported in the
//     x-makefun-upstream-failures header, the only way the failover column can see it. A 401 means no
//     key was pushed: retrying would burn the trial's call budget for nothing, so the prompt is lost
//     at once.
//   - the wires: every field the scorer echoes — system prompt, user text, tools, max_tokens, stop
//     sequences, image, JSON mode — is forwarded to the provider, because a dropped field fails the
//     wire checks. Streaming always asks for the usage chunk (stream_options.include_usage), which the
//     scorer's usage check needs. JSON-mode requests never route to a model without JSON support:
//     the manifest's requirements say so, and the router enforces it.
//   - the catalog (5%): the catalog is exactly the brief's fourteen free ids, and GET /v1/models
//     answers with that same set on both auth headers.
//
// Everything here is deterministic: no clock, no unseeded randomness, so a live pass and its replay
// make the same gateway calls in the same order and leave the same state. The trial's rules are in
// the kit's RULES.txt.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  chatJsonWithModel,
  chatStreamWithModel,
  chatToMessages,
  messagesEventsFromChat,
  messagesToChat,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** Milli-points taken off a candidate per call already made to that model: spreads the load. */
const LOAD_PENALTY_MILLI = 1;
/** Milli-points taken off a candidate per call that model refused: steers away from an outage. */
const FAILURE_PENALTY_MILLI = 15;
/** Milli-points taken off a candidate that answered 429: its rate limit outlasts the trial. */
const RATE_LIMIT_PENALTY_MILLI = 200;
/** How long one gateway call may take before it counts as unreachable and fails over. */
const GATEWAY_TIMEOUT_MS = 20000;
/** The output cap sent when a client request names none: the gateway refuses requests without one. */
const DEFAULT_MAX_TOKENS = 1024;
/** The gateway's output cap ceiling: a request over it is refused. */
const MAX_OUTPUT_TOKENS = 8192;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Calls made to each model so far (successful and refused), by dialable id. */
const callsByModel = new Map<string, number>();
/** Calls each model refused so far, by dialable id. */
const failuresByModel = new Map<string, number>();
/** Models that answered 429 this trial: their free rate limit outlasts the trial's remaining prompts. */
const rateLimited = new Set<string>();
/**
 * Models hard-excluded for the rest of the trial after a per-model refusal (HTTP 403 — the model is
 * gated to agentic harnesses and never recovers within a trial — or 400 model_not_allowed). A small
 * penalty cannot demote the top-rated model (1475) below the next-best (~1303): the gap is ~172
 * milli, so exclusion, not a penalty, is the only fix. Routing-only: the model stays in the
 * catalog and in GET /v1/models, which must list all 14 ids.
 */
const excluded = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

interface Candidate {
  model: ModelRow;
  rating: number;
  score: number;
}

/**
 * The free candidates meeting one request's requirements, best first. Each candidate's score is its
 * brief rating less the load penalty (calls already made to that model), the failure penalty
 * (calls that model refused) and the rate-limit penalty (a 429'd model is cooling down for the rest
 * of the trial). A model that just failed drops down the order, and a model already used several
 * times yields to a fresh one of nearly the same rating.
 */
function candidatesFor(requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const model of freeModels(brief)) {
    if (excluded.has(model.id)) continue;
    if (!meetsRequirements(requires, capabilitiesOf(model))) continue;
    const rating = model.rating_milli;
    const calls = callsByModel.get(model.id) ?? 0;
    const failures = failuresByModel.get(model.id) ?? 0;
    const cooled = rateLimited.has(model.id) ? RATE_LIMIT_PENALTY_MILLI : 0;
    out.push({ model, rating, score: rating - LOAD_PENALTY_MILLI * calls - FAILURE_PENALTY_MILLI * failures - cooled });
  }
  out.sort((a, b) => b.score - a.score || b.rating - a.rating || (a.model.id < b.model.id ? -1 : 1));
  return out;
}

/** The requirements of the request now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The brief's prompt list is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One refused gateway call, as a row of the x-makefun-upstream-failures trail. */
interface TrailRow {
  status: number;
  code: string;
  model: string | null;
}

function trailHeader(rows: TrailRow[]): Record<string, string> {
  if (rows.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(rows.slice(0, 8)) };
}

/** The gateway's error code for a refused call, read from its error body when it parses. */
async function errorCodeOf(res: Response): Promise<string> {
  try {
    const text = await res.text();
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      if (typeof parsed.error.code === 'string' && parsed.error.code !== '') return parsed.error.code;
      if (typeof parsed.error.type === 'string' && parsed.error.type !== '') return parsed.error.type;
    }
    return `http_${res.status}`;
  } catch {
    return `http_${res.status}`;
  }
}

/** Whether a refusal is worth retrying on another model: 402, 403, 429, 5xx, unreachable, or a dead id. */
function retryable(status: number, code: string): boolean {
  if (status === 429 || status === 402 || status === 403) return true;
  if (status >= 500) return true;
  if (code === 'model_not_allowed') return true;
  if (code === 'upstream_unreachable') return true;
  return false;
}

/**
 * The upstream Chat Completions body for one candidate: the client's OpenAI body carried through, or
 * the Anthropic request rewritten first. The output cap is always present (the gateway refuses a
 * request without one) and streaming always asks for the usage chunk (the scorer's usage check needs
 * it).
 */
function upstreamBody(candidate: ModelRow, body: Json, clientApi: 'openai' | 'anthropic'): Json {
  const request: Json = clientApi === 'openai' ? { ...body, model: candidate.id } : messagesToChat(body, candidate.id);
  if (typeof request.max_tokens !== 'number' && typeof request.max_completion_tokens !== 'number') {
    request.max_tokens = DEFAULT_MAX_TOKENS;
  }
  if (typeof request.max_tokens === 'number' && request.max_tokens > MAX_OUTPUT_TOKENS) request.max_tokens = MAX_OUTPUT_TOKENS;
  if (body.stream === true) {
    request.stream = true;
    request.stream_options = { ...(isRecord(request.stream_options) ? request.stream_options : {}), include_usage: true };
  }
  return request;
}

/** One gateway call: the response, or null when the call itself failed (network, timeout). */
async function callOnce(candidate: ModelRow, body: Json, clientApi: 'openai' | 'anthropic'): Promise<Response | null> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  try {
    return await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody(candidate, body, clientApi)),
      signal: AbortSignal.timeout(GATEWAY_TIMEOUT_MS),
    });
  } catch {
    return null;
  }
}

function recordCall(modelId: string, ok: boolean): void {
  callsByModel.set(modelId, (callsByModel.get(modelId) ?? 0) + 1);
  if (!ok) failuresByModel.set(modelId, (failuresByModel.get(modelId) ?? 0) + 1);
}

/**
 * Serve one client request (either API, JSON or streaming) by trying the ranked candidates in order
 * until one answers 2xx. A refusal moves to the next candidate; the loop stops at the first success,
 * so exactly one gateway call answers the request. Refusals a retry recovered from are reported in
 * the x-makefun-upstream-failures header, the only way the failover column can see them.
 */
async function serve(req: IncomingMessage, res: ServerResponse, body: Json, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const stream = body.stream === true;
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) {
    sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model in the brief meets this request');
    return;
  }
  const trail: TrailRow[] = [];
  let lastStatus = 502;
  let lastDetail = 'no model answered';
  for (const candidate of candidates) {
    const upstream = await callOnce(candidate.model, body, clientApi);
    if (upstream === null) {
      recordCall(candidate.model.id, false);
      trail.push({ status: 502, code: 'upstream_unreachable', model: candidate.model.id });
      lastStatus = 502;
      lastDetail = `the gateway did not answer for ${candidate.model.id}`;
      continue;
    }
    if (!upstream.ok || upstream.body === null) {
      recordCall(candidate.model.id, false);
      const code = await errorCodeOf(upstream);
      trail.push({ status: upstream.status, code, model: candidate.model.id });
      // A 403 names the dialled model as gated to agentic harnesses: that model will refuse every
      // later prompt the same way, so exclude it for the rest of the trial instead of re-dialling
      // it. Same for a 400 model_not_allowed: the id is dead, exclude it so the next prompt does
      // not re-pick it on rating alone. (A 401 is different — no key was pushed — and stops below.)
      if (upstream.status === 403) excluded.add(candidate.model.id);
      if (upstream.status === 400 && code === 'model_not_allowed') excluded.add(candidate.model.id);
      // A 429 is a per-minute rate limit, not a one-off: the model stays cooling down for the rest
      // of the trial, so later prompts go straight to a fresh model instead of re-proving it.
      if (upstream.status === 429) rateLimited.add(candidate.model.id);
      lastStatus = upstream.status;
      lastDetail = `the gateway answered HTTP ${upstream.status} (${code}) for ${candidate.model.id}`;
      // A 401 means no key was pushed: every further model would answer the same, so stop at once
      // instead of burning the trial's call budget. Other 4xx that are not retryable end the prompt
      // too; only a refusal another model could survive is retried.
      if (upstream.status === 401 || !retryable(upstream.status, code)) break;
      continue;
    }
    recordCall(candidate.model.id, true);
    const headers = trailHeader(trail);
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
      try {
        if (clientApi === 'openai') {
          for await (const frame of chatStreamWithModel(upstream.body, candidate.model.id)) res.write(frame);
        } else {
          for await (const event of messagesEventsFromChat(upstream.body, candidate.model.id)) res.write(event);
        }
      } finally {
        res.end();
      }
      return;
    }
    let answer: unknown;
    try {
      answer = await upstream.json();
    } catch {
      sendError(res, clientApi, 502, 'api_error', `the gateway answered ${candidate.model.id} with a body that is not JSON`);
      return;
    }
    if (!isRecord(answer)) {
      sendError(res, clientApi, 502, 'api_error', `the gateway answered ${candidate.model.id} with a body that is not a JSON object`);
      return;
    }
    if (clientApi === 'openai') sendJson(res, 200, chatJsonWithModel(answer, candidate.model.id), headers);
    else sendJson(res, 200, chatToMessages(answer, candidate.model.id), headers);
    return;
  }
  sendError(res, clientApi, lastStatus === 402 || lastStatus === 429 ? lastStatus : 502, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', `no model answered: ${lastDetail}`);
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, body, 'openai');
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, body, 'anthropic');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
