import { NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";

// Server-side agent: real Claude tool-use loop. Only active when ANTHROPIC_API_KEY is set in the Vercel
// project; otherwise the client falls back to the in-browser sim. Node runtime (SDK + fetch).
export const runtime = "nodejs";

const MODEL = process.env.AGENT_MODEL || "claude-opus-4-8";
const MAX_STEPS = 6;
const CAP = 4000;

// --- safe server tools (no shell/eval-of-identifiers; arithmetic is whitelisted) --------------------
const TOOLS: Anthropic.Tool[] = [
  {
    name: "calculator",
    description: "Evaluate an arithmetic expression (+ - * / parentheses).",
    input_schema: { type: "object", properties: { expression: { type: "string" } }, required: ["expression"] },
  },
  {
    name: "get_time",
    description: "Get the current UTC date-time (ISO 8601).",
    input_schema: { type: "object", properties: {} },
  },
  {
    name: "http_get",
    description: "Fetch the text of a public http(s) URL (read-only).",
    input_schema: { type: "object", properties: { url: { type: "string" } }, required: ["url"] },
  },
];

async function runTool(name: string, input: any): Promise<string> {
  try {
    if (name === "calculator") {
      const expr = String(input?.expression ?? "");
      if (!/^[0-9+\-*/(). %]+$/.test(expr)) return "error: only numbers and + - * / ( ) are allowed";
      // eslint-disable-next-line no-new-func
      const val = Function(`"use strict"; return (${expr});`)();
      return String(val);
    }
    if (name === "get_time") return new Date().toISOString();
    if (name === "http_get") {
      const url = String(input?.url ?? "");
      if (!/^https?:\/\//i.test(url)) return "error: url must be http(s)";
      const res = await fetch(url, { signal: AbortSignal.timeout(10_000) });
      return (await res.text()).slice(0, CAP);
    }
    return `error: unknown tool ${name}`;
  } catch (e: any) {
    return `error: ${e?.message ?? e}`;
  }
}

const SYSTEM =
  "You are a helpful, careful tool-using agent. Plan, call tools when they help, and STOP by giving a " +
  "final answer. If no tool fits, say so plainly instead of inventing one; if a request is missing " +
  "information, ask for it. Keep answers concise and speakable.";

export async function GET() {
  return NextResponse.json({ available: !!process.env.ANTHROPIC_API_KEY, model: MODEL });
}

export async function POST(req: Request) {
  const key = process.env.ANTHROPIC_API_KEY;
  if (!key) return NextResponse.json({ error: "agent not configured (no ANTHROPIC_API_KEY)" }, { status: 400 });

  let body: any;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
  }
  const message = String(body?.message ?? "").slice(0, 4000);
  if (!message) return NextResponse.json({ error: "empty message" }, { status: 400 });

  const client = new Anthropic({ apiKey: key });
  const messages: Anthropic.MessageParam[] = [{ role: "user", content: message }];
  const trace: { tool: string; input: unknown; output: string }[] = [];

  try {
    for (let step = 0; step < MAX_STEPS; step++) {
      const resp = await client.messages.create({
        model: MODEL,
        max_tokens: 1024,
        system: SYSTEM,
        tools: TOOLS,
        messages,
      });
      if (resp.stop_reason === "tool_use") {
        messages.push({ role: "assistant", content: resp.content });
        const results: Anthropic.ToolResultBlockParam[] = [];
        for (const block of resp.content) {
          if (block.type === "tool_use") {
            const out = await runTool(block.name, block.input);
            trace.push({ tool: block.name, input: block.input, output: out.slice(0, 400) });
            results.push({ type: "tool_result", tool_use_id: block.id, content: out });
          }
        }
        messages.push({ role: "user", content: results });
        continue;
      }
      const reply = resp.content
        .filter((b): b is Anthropic.TextBlock => b.type === "text")
        .map((b) => b.text)
        .join("\n")
        .trim();
      return NextResponse.json({ reply: reply || "(no answer)", trace, brain: "claude", model: MODEL });
    }
    return NextResponse.json({ reply: "Stopped after the step budget.", trace, brain: "claude", model: MODEL });
  } catch (e: any) {
    return NextResponse.json({ error: `agent error: ${e?.message ?? e}` }, { status: 500 });
  }
}
