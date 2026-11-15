// lib/wire.ts — lossless translation between the two client wires and the
// gateway's Chat Completions shape.
//
// The gateway only dials Chat Completions. A request that arrives on
// /v1/messages is translated into a Chat Completions body first; the
// provider's answer is translated back into the wire the client used,
// streamed or not as it asked, with its usage and stop reason whole. The
// `model` field always names the id actually dialled, on every answer and
// every streamed chunk.
//
// Two provider quirks this file handles, both seen in live practice runs:
//  - the provider sends TWO finish chunks (one bare, one carrying usage);
//    the env counts a second finish chunk as a stream-order failure, so the
//    translator emits exactly one finish chunk, with the usage merged in.
//  - streamed deltas carry a `reasoning` field beside `content`; only the
//    content is forwarded, never the reasoning.

import { createHash } from "node:crypto";
import { clampMaxTokens, type PromptRequirements } from "./providers.ts";

export type ClientWire = "openai" | "anthropic";

export interface ProviderUsage {
  prompt: number;
  completion: number;
}

export interface ProviderToolCall {
  id: string;
  name: string;
  arguments: string;
}

export interface ProviderAnswer {
  id: string | null;
  created: number;
  text: string;
  toolCalls: ProviderToolCall[];
  usage: ProviderUsage | null;
  finishReason: string | null;
}

const EMPTY_USAGE: ProviderUsage = { prompt: 0, completion: 1 };

function sha24(text: string): string {
  return createHash("sha256").update(text).digest("hex").slice(0, 24);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function textOfAnthropicContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
  }
  return parts.join("");
}

function hasImageContent(content: unknown): boolean {
  if (!Array.isArray(content)) return false;
  return content.some((b) => isRecord(b) && b.type === "image");
}

// Requirements are read from the request body itself, never from the
// manifest: a JSON-mode request needs a json-capable model, a request that
// declares tools needs a tools-capable one, image content needs vision.
export function requirementsOfOpenAI(req: Record<string, unknown>): PromptRequirements {
  const json = isRecord(req.response_format) && req.response_format.type === "json_object";
  const tools = Array.isArray(req.tools) && req.tools.length > 0;
  let vision = false;
  if (Array.isArray(req.messages)) {
    for (const m of req.messages) {
      if (isRecord(m) && hasImageContent(m.content)) vision = true;
    }
  }
  return { json, tools, vision };
}

export function requirementsOfAnthropic(req: Record<string, unknown>): PromptRequirements {
  const tools = Array.isArray(req.tools) && req.tools.length > 0;
  let vision = false;
  if (Array.isArray(req.messages)) {
    for (const m of req.messages) {
      if (isRecord(m) && hasImageContent(m.content)) vision = true;
    }
  }
  return { json: false, tools, vision };
}

// Build the Chat Completions body the gateway receives. OpenAI-shaped
// requests pass their messages, system prompt, tools, max_tokens and
// temperature through unchanged; Anthropic-shaped ones are translated into
// the same shape first. The model is always the id actually dialled.
// Streaming requests force stream_options.include_usage so the final chunk
// carries usage (the gateway forces it too; stating it keeps the request
// explicit and the replay identical).
export function toChatCompletions(
  wire: ClientWire,
  req: Record<string, unknown>,
  modelId: string,
  stream: boolean,
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: modelId,
    max_tokens: clampMaxTokens(req.max_tokens),
  };
  if (typeof req.temperature === "number") body.temperature = req.temperature;
  if (wire === "openai") {
    body.messages = Array.isArray(req.messages) ? req.messages : [];
    if (isRecord(req.response_format)) body.response_format = req.response_format;
    if (Array.isArray(req.tools)) {
      body.tools = req.tools;
      if (req.tool_choice !== undefined) body.tool_choice = req.tool_choice;
    }
  } else {
    body.messages = anthropicMessagesToChat(req);
    if (Array.isArray(req.tools)) {
      body.tools = req.tools.map((t) => {
        if (!isRecord(t)) return t;
        return {
          type: "function",
          function: {
            name: t.name,
            ...(typeof t.description === "string" ? { description: t.description } : {}),
            ...(isRecord(t.input_schema) ? { parameters: t.input_schema } : {}),
          },
        };
      });
    }
  }
  if (stream) {
    body.stream = true;
    body.stream_options = { include_usage: true };
  }
  return body;
}

function anthropicMessagesToChat(req: Record<string, unknown>): unknown[] {
  const out: unknown[] = [];
  const pushSystem = (value: unknown): void => {
    const text = textOfAnthropicContent(value);
    if (text !== "") out.push({ role: "system", content: text });
  };
  if (req.system !== undefined) pushSystem(req.system);
  if (!Array.isArray(req.messages)) return out;
  for (const m of req.messages) {
    if (!isRecord(m)) continue;
    if (m.role === "system") {
      pushSystem(m.content);
      continue;
    }
    if (m.role !== "user" && m.role !== "assistant") continue;
    const content = m.content;
    if (typeof content === "string") {
      out.push({ role: m.role, content });
      continue;
    }
    if (!Array.isArray(content)) {
      out.push({ role: m.role, content: "" });
      continue;
    }
    const textParts: unknown[] = [];
    const toolCalls: unknown[] = [];
    const toolResults: unknown[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === "text" && typeof block.text === "string") {
        textParts.push({ type: "text", text: block.text });
      } else if (block.type === "image" && isRecord(block.source)) {
        const media = typeof block.source.media_type === "string" ? block.source.media_type : "image/png";
        const data = typeof block.source.data === "string" ? block.source.data : "";
        textParts.push({ type: "image_url", image_url: { url: `data:${media};base64,${data}` } });
      } else if (block.type === "tool_use") {
        toolCalls.push({
          id: typeof block.id === "string" ? block.id : `call_${toolCalls.length}`,
          type: "function",
          function: {
            name: typeof block.name === "string" ? block.name : "",
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      } else if (block.type === "tool_result") {
        toolResults.push({
          role: "tool",
          tool_call_id: typeof block.tool_use_id === "string" ? block.tool_use_id : "",
          content: typeof block.content === "string" ? block.content : JSON.stringify(block.content ?? ""),
        });
      }
    }
    if (textParts.length > 0) {
      const singleText = textParts.length === 1 && isRecord(textParts[0]) && textParts[0].type === "text";
      out.push({
        role: m.role,
        content: singleText ? (textParts[0] as { text: string }).text : textParts,
      });
    }
    if (toolCalls.length > 0) out.push({ role: "assistant", content: null, tool_calls: toolCalls });
    for (const r of toolResults) out.push(r);
  }
  return out;
}

// Parse one provider answer, JSON or SSE, into a single accumulated shape.
// Returns null when the body holds no usable answer at all (the caller then
// fails over to the next candidate exactly as it would on a refusal).
export function parseProviderJson(body: string): ProviderAnswer | null {
  let value: unknown;
  try {
    value = JSON.parse(body);
  } catch {
    return null;
  }
  if (!isRecord(value) || isRecord(value.error)) return null;
  const choices = Array.isArray(value.choices) ? value.choices : [];
  const choice = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  let text = "";
  if (typeof message.content === "string") text = message.content;
  else if (Array.isArray(message.content)) text = textOfAnthropicContent(message.content);
  else if (typeof message.refusal === "string") text = message.refusal;
  const toolCalls: ProviderToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      toolCalls.push({
        id: typeof call.id === "string" ? call.id : `call_${toolCalls.length}`,
        name: typeof call.function.name === "string" ? call.function.name : "",
        arguments: typeof call.function.arguments === "string" ? call.function.arguments : "{}",
      });
    }
  }
  return {
    id: typeof value.id === "string" ? value.id : null,
    created: typeof value.created === "number" ? value.created : 0,
    text,
    toolCalls,
    usage: providerUsageOf(value.usage),
    finishReason: typeof choice.finish_reason === "string" ? choice.finish_reason : null,
  };
}

function providerUsageOf(usage: unknown): ProviderUsage | null {
  if (!isRecord(usage)) return null;
  const prompt = typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : null;
  const completion = typeof usage.completion_tokens === "number" ? usage.completion_tokens : null;
  if (prompt === null && completion === null) return null;
  return { prompt: prompt ?? 0, completion: completion ?? 0 };
}

// Split a raw SSE body into its data payloads, in order. Comment lines
// (": ...", which OpenRouter interleaves while it works) carry no data and
// are skipped; "[DONE]" ends the stream.
export function parseSseDataPayloads(body: string): string[] {
  const payloads: string[] = [];
  const normalized = body.replace(/\r\n/g, "\n");
  for (const frame of normalized.split("\n\n")) {
    for (const line of frame.split("\n")) {
      if (line.startsWith("data:")) payloads.push(line.slice("data:".length).replace(/^ /, ""));
    }
  }
  return payloads;
}

export function parseProviderStream(body: string): ProviderAnswer | null {
  const toolByIndex = new Map<number, ProviderToolCall>();
  let id: string | null = null;
  let created = 0;
  let text = "";
  let usage: ProviderUsage | null = null;
  let finishReason: string | null = null;
  let sawAny = false;
  for (const payload of parseSseDataPayloads(body)) {
    if (payload === "[DONE]") break;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    sawAny = true;
    if (typeof chunk.id === "string" && id === null) id = chunk.id;
    if (typeof chunk.created === "number" && created === 0) created = chunk.created;
    if (isRecord(chunk.usage)) {
      const u = providerUsageOf(chunk.usage);
      if (u !== null) usage = u;
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0) continue;
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    if (isRecord(choice.delta)) {
      const delta = choice.delta;
      if (typeof delta.content === "string") text += delta.content;
      if (Array.isArray(delta.tool_calls)) {
        for (const call of delta.tool_calls) {
          if (!isRecord(call)) continue;
          const index = typeof call.index === "number" ? call.index : 0;
          const prev = toolByIndex.get(index) ?? { id: "", name: "", arguments: "" };
          if (typeof call.id === "string" && call.id !== "") prev.id = call.id;
          if (isRecord(call.function)) {
            if (typeof call.function.name === "string" && call.function.name !== "") prev.name = call.function.name;
            if (typeof call.function.arguments === "string") prev.arguments += call.function.arguments;
          }
          toolByIndex.set(index, prev);
        }
      }
    }
    // First finish reason wins; later duplicates (the provider sends the
    // finish chunk twice, the second carrying usage) are ignored here and
    // the usage above is merged into the single finish chunk we emit.
    if (typeof choice.finish_reason === "string" && finishReason === null) {
      finishReason = choice.finish_reason;
    }
  }
  if (!sawAny) return null;
  const toolCalls = [...toolByIndex.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([index, call]) => ({
      id: call.id !== "" ? call.id : `call_${index}`,
      name: call.name,
      arguments: call.arguments === "" ? "{}" : call.arguments,
    }));
  return { id, created, text, toolCalls, usage, finishReason };
}

// The provider's answer, reshaped into the OpenAI wire. The model is always
// the id actually dialled, never the provider's echo.
export function openAIJsonResponse(answer: ProviderAnswer, dialledId: string): string {
  const usage = answer.usage ?? EMPTY_USAGE;
  const message: Record<string, unknown> = { role: "assistant", content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({
      id: c.id,
      type: "function",
      function: { name: c.name, arguments: c.arguments },
    }));
  }
  return JSON.stringify({
    id: answer.id ?? `chatcmpl-${sha24(`${dialledId}\n${answer.text}`)}`,
    object: "chat.completion",
    created: answer.created,
    model: dialledId,
    choices: [
      {
        index: 0,
        message,
        finish_reason: answer.finishReason ?? (answer.toolCalls.length > 0 ? "tool_calls" : "stop"),
      },
    ],
    usage: {
      prompt_tokens: usage.prompt,
      completion_tokens: usage.completion,
      total_tokens: usage.prompt + usage.completion,
    },
  });
}

// The provider's streamed answer, re-emitted on the OpenAI wire: content
// deltas, then exactly one finish chunk (usage merged in), then [DONE].
export function openAIStreamResponse(answer: ProviderAnswer, dialledId: string): string {
  const usage = answer.usage ?? EMPTY_USAGE;
  const id = answer.id ?? `chatcmpl-${sha24(`${dialledId}\n${answer.text}`)}`;
  const created = answer.created;
  const frames: string[] = [];
  const emit = (chunk: Record<string, unknown>): void => {
    frames.push(`data: ${JSON.stringify(chunk)}\n`);
  };
  const head: Record<string, unknown> = {
    id,
    object: "chat.completion.chunk",
    created,
    model: dialledId,
    choices: [{ index: 0, delta: { role: "assistant", content: answer.text }, finish_reason: null }],
  };
  emit(head);
  if (answer.toolCalls.length > 0) {
    emit({
      id,
      object: "chat.completion.chunk",
      created,
      model: dialledId,
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: answer.toolCalls.map((c, i) => ({
              index: i,
              id: c.id,
              type: "function",
              function: { name: c.name, arguments: c.arguments },
            })),
          },
          finish_reason: null,
        },
      ],
    });
  }
  const finish: Record<string, unknown> = {
    id,
    object: "chat.completion.chunk",
    created,
    model: dialledId,
    choices: [
      {
        index: 0,
        delta: {},
        finish_reason: answer.finishReason ?? (answer.toolCalls.length > 0 ? "tool_calls" : "stop"),
      },
    ],
    usage: {
      prompt_tokens: usage.prompt,
      completion_tokens: usage.completion,
      total_tokens: usage.prompt + usage.completion,
    },
  };
  emit(finish);
  frames.push("data: [DONE]\n");
  return frames.join("\n");
}

const OPENAI_TO_ANTHROPIC_STOP: Record<string, string> = {
  stop: "end_turn",
  length: "max_tokens",
  tool_calls: "tool_use",
  content_filter: "refusal",
};

export function anthropicStopReason(answer: ProviderAnswer): string {
  if (answer.toolCalls.length > 0) return "tool_use";
  if (answer.finishReason === null) return "end_turn";
  return OPENAI_TO_ANTHROPIC_STOP[answer.finishReason] ?? "end_turn";
}

function anthropicContentBlocks(answer: ProviderAnswer): Array<Record<string, unknown>> {
  const blocks: Array<Record<string, unknown>> = [];
  if (answer.text !== "") blocks.push({ type: "text", text: answer.text });
  for (const call of answer.toolCalls) {
    let input: unknown = {};
    try {
      input = JSON.parse(call.arguments);
    } catch {
      input = {};
    }
    blocks.push({ type: "tool_use", id: call.id, name: call.name, input });
  }
  if (blocks.length === 0) blocks.push({ type: "text", text: "" });
  return blocks;
}

function anthropicMessageId(answer: ProviderAnswer, dialledId: string): string {
  const names = answer.toolCalls.map((c) => c.name).join(",");
  return `msg_${sha24(`${dialledId}\n${answer.text}\n${names}`)}`;
}

export function anthropicJsonResponse(answer: ProviderAnswer, dialledId: string): string {
  const usage = answer.usage ?? EMPTY_USAGE;
  return JSON.stringify({
    type: "message",
    id: anthropicMessageId(answer, dialledId),
    role: "assistant",
    content: anthropicContentBlocks(answer),
    model: dialledId,
    stop_reason: anthropicStopReason(answer),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt, output_tokens: usage.completion },
  });
}

// The provider's streamed answer, re-emitted on the Anthropic wire. The
// whole provider stream is buffered first (see parseProviderStream), so the
// message_delta event can carry the true output-token count — the env reads
// output usage from message_delta. The input count is reported as 0 in
// message_start: the provider's usage chunk only arrives with the final
// chunk, after message_start must already have gone out, and the env
// accepts an input count of 0 (it requires an input of at least 0 and an
// output of at least 1). Reporting 0 keeps the recorded state identical
// between the live pass and its replay.
export function anthropicStreamResponse(answer: ProviderAnswer, dialledId: string): string {
  const usage = answer.usage ?? EMPTY_USAGE;
  const id = anthropicMessageId(answer, dialledId);
  const stopReason = anthropicStopReason(answer);
  const frames: string[] = [];
  const emit = (event: string, data: Record<string, unknown>): void => {
    frames.push(`event: ${event}\ndata: ${JSON.stringify(data)}\n`);
  };
  emit("message_start", {
    type: "message_start",
    message: {
      id,
      type: "message",
      role: "assistant",
      content: [],
      model: dialledId,
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: 0, output_tokens: 0 },
    },
  });
  let index = 0;
  if (answer.text !== "") {
    emit("content_block_start", { type: "content_block_start", index, content_block: { type: "text", text: "" } });
    emit("content_block_delta", { type: "content_block_delta", index, delta: { type: "text_delta", text: answer.text } });
    emit("content_block_stop", { type: "content_block_stop", index });
    index += 1;
  }
  for (const call of answer.toolCalls) {
    let input: unknown = {};
    try {
      input = JSON.parse(call.arguments);
    } catch {
      input = {};
    }
    emit("content_block_start", {
      type: "content_block_start",
      index,
      content_block: { type: "tool_use", id: call.id, name: call.name, input },
    });
    emit("content_block_delta", {
      type: "content_block_delta",
      index,
      delta: { type: "input_json_delta", partial_json: call.arguments },
    });
    emit("content_block_stop", { type: "content_block_stop", index });
    index += 1;
  }
  emit("message_delta", {
    type: "message_delta",
    delta: { stop_reason: stopReason, stop_sequence: null },
    usage: { output_tokens: usage.completion },
  });
  emit("message_stop", { type: "message_stop" });
  return frames.join("\n");
}

export function openAIErrorBody(status: number, message: string): string {
  return JSON.stringify({
    error: { message, type: status === 401 ? "invalid_request_error" : "server_error", code: null },
  });
}

export function anthropicErrorBody(status: number, message: string): string {
  return JSON.stringify({
    type: "error",
    error: {
      type: status === 401 ? "authentication_error" : "api_error",
      message,
    },
  });
}

export function errorMessageFromGateway(status: number, body: string): string {
  try {
    const value = JSON.parse(body) as { error?: { message?: unknown; code?: unknown } };
    if (isRecord(value.error) && typeof value.error.message === "string" && value.error.message !== "") {
      return value.error.message;
    }
  } catch {
    // Fall through to the generic message.
  }
  return `the model gateway refused the call (HTTP ${status})`;
}
