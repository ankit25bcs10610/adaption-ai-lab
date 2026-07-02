/**
 * Deterministic tool-calling engine for the in-browser playground.
 *
 * This mirrors the DECISION LOGIC the fine-tuned model is trained to produce (call / refuse / clarify)
 * without downloading a 1 GB model — so the demo is instant, works on every judge's browser, and the
 * "toggle a tool off → CALL becomes REFUSE" cause/effect is real. It is a faithful simulation of the
 * behavior, not the weights; the page says so.
 */

export interface ToolParam {
  type?: string;
  description?: string;
  enum?: (string | number)[];
}
export interface Tool {
  name: string;
  description?: string;
  parameters?: {
    type?: string;
    properties?: Record<string, ToolParam>;
    required?: string[];
  };
  enabled?: boolean;
}

export type Outcome = "call" | "refuse" | "clarify";

export interface SimResult {
  action: Outcome;
  calls?: { name: string; arguments: Record<string, unknown> }[];
  message?: string;
  rationale: string;
}

const STOP = new Set([
  "the", "a", "an", "to", "for", "of", "in", "on", "me", "my", "please", "can", "you", "i", "is",
  "it", "and", "with", "get", "what", "whats", "s", "do", "now", "right", "give", "tell",
]);

function tokens(s: string): string[] {
  return (s.toLowerCase().match(/[a-z0-9]+/g) || []).filter((t) => t.length > 1 && !STOP.has(t));
}

function scoreTool(tool: Tool, qtok: Set<string>): number {
  const ttok = new Set(tokens(`${tool.name} ${tool.description ?? ""}`.replace(/_/g, " ")));
  let score = 0;
  ttok.forEach((t) => {
    if (qtok.has(t)) score += 1;
  });
  return score;
}

/** Try to fill a required arg's value from the query. Returns undefined if not found. */
function extractArg(name: string, spec: ToolParam, query: string, qtok: Set<string>): unknown {
  const q = query.toLowerCase();
  if (spec.enum && spec.enum.length) {
    const hit = spec.enum.find((e) => q.includes(String(e).toLowerCase()));
    if (hit !== undefined) return hit;
    return undefined;
  }
  const type = spec.type ?? "string";
  if (type === "integer" || type === "number") {
    const m = query.match(/-?\d+(\.\d+)?/);
    return m ? Number(m[0]) : undefined;
  }
  if (type === "boolean") {
    if (/\b(true|yes|enable|on)\b/i.test(query)) return true;
    if (/\b(false|no|disable|off)\b/i.test(query)) return false;
    return undefined;
  }
  // string: quoted value, else a Capitalized proper noun, else a salient query token
  const quoted = query.match(/"([^"]+)"|'([^']+)'/);
  if (quoted) return quoted[1] ?? quoted[2];
  const proper = query.match(/\b([A-Z][a-zA-Z]{2,})\b/);
  if (proper && !/^(please|book|delete|write|what)$/i.test(proper[1])) return proper[1];
  // fall back to a token that also appears near the arg name
  if (qtok.has(name.toLowerCase())) return name;
  return undefined;
}

export function simulate(tools: Tool[], query: string): SimResult {
  const active = tools.filter((t) => t.enabled !== false);
  const qtok = new Set(tokens(query));

  if (!query.trim()) {
    return { action: "clarify", message: "What would you like to do?", rationale: "Empty request." };
  }
  if (active.length === 0) {
    return {
      action: "refuse",
      message: "No tools are available, so I can't act on this request.",
      rationale: "No enabled tools.",
    };
  }

  const scored = active
    .map((t) => ({ t, s: scoreTool(t, qtok) }))
    .sort((a, b) => b.s - a.s);
  const best = scored[0];

  // No relevant tool → refuse (this is the moat: don't hallucinate a call)
  if (best.s === 0) {
    return {
      action: "refuse",
      message: "None of the available tools can handle this request.",
      rationale: "No tool's name/description overlaps with the request.",
    };
  }

  // Two tools tie on relevance → clarify which one
  if (scored.length > 1 && scored[1].s === best.s && best.s > 0) {
    return {
      action: "clarify",
      message: `Did you mean to use \`${best.t.name}\` or \`${scored[1].t.name}\`? Let me know which.`,
      rationale: "Two tools match the request equally well.",
    };
  }

  // Fill required args; if any required arg is missing → clarify
  const tool = best.t;
  const props = tool.parameters?.properties ?? {};
  const required = tool.parameters?.required ?? Object.keys(props);
  const args: Record<string, unknown> = {};
  const missing: string[] = [];
  for (const name of required) {
    const val = extractArg(name, props[name] ?? {}, query, qtok);
    if (val === undefined) missing.push(name);
    else args[name] = val;
  }
  if (missing.length) {
    return {
      action: "clarify",
      message: `I can use \`${tool.name}\`, but I need the required ${missing
        .map((m) => `\`${m}\``)
        .join(", ")} before I can proceed. Could you provide ${missing.length > 1 ? "them" : "it"}?`,
      rationale: `Missing required argument(s): ${missing.join(", ")}.`,
    };
  }

  return {
    action: "call",
    calls: [{ name: tool.name, arguments: args }],
    rationale: `\`${tool.name}\` matches and all required arguments are present.`,
  };
}

// ---- Preset tools + scenarios -------------------------------------------------
export const PRESET_TOOLS: Tool[] = [
  {
    name: "get_weather",
    description: "Get the current weather for a city",
    parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
    enabled: true,
  },
  {
    name: "book_flight",
    description: "Book a flight between two cities on a date",
    parameters: {
      type: "object",
      properties: { origin: { type: "string" }, destination: { type: "string" }, date: { type: "string" } },
      required: ["origin", "destination", "date"],
    },
    enabled: true,
  },
  {
    name: "convert_currency",
    description: "Convert an amount from one currency to another",
    parameters: {
      type: "object",
      properties: { amount: { type: "number" }, from: { type: "string" }, to: { type: "string" } },
      required: ["amount", "from", "to"],
    },
    enabled: true,
  },
];

export interface Scenario {
  label: string;
  outcome: Outcome;
  query: string;
}

export const SCENARIOS: Scenario[] = [
  { label: "A valid call", outcome: "call", query: 'What is the weather in "Mumbai"?' },
  { label: "Missing argument", outcome: "clarify", query: "Book me a flight to Delhi." },
  { label: "No tool fits", outcome: "refuse", query: "Write me a poem about the monsoon." },
];
