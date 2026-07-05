/**
 * In-browser agent turn — the offline "brain" for the on-site agent.
 *
 * Reuses the deterministic tool-caller simulator (toolcaller-sim.ts, which mirrors the trained model's
 * call / refuse / clarify decision) and adds client-side tool EXECUTORS so a prompt actually produces a
 * result — no server, no key. Executors are clearly demo stubs (a static site can't book a real flight);
 * the real Claude-powered path (/api/agent) does live reasoning + real tools when a key is configured.
 */
import { PRESET_TOOLS, simulate, type SimResult, type Tool } from "./toolcaller-sim";

export type { Tool };
export const AGENT_TOOLS: Tool[] = PRESET_TOOLS;

/** Client-side, side-effect-free demo executors keyed by tool name. */
const EXECUTORS: Record<string, (a: Record<string, unknown>) => string> = {
  get_weather: (a) => `(demo) ${a.city ?? "there"}: 29°C, light clouds, gentle breeze.`,
  book_flight: (a) =>
    `(demo) Reserved ${a.origin ?? "?"} → ${a.destination ?? "?"} on ${a.date ?? "?"}. Confirmation #DL${String(
      Math.abs(hash(`${a.origin}${a.destination}${a.date}`)) % 9000 + 1000,
    )}.`,
  convert_currency: (a) => {
    const amt = Number(a.amount) || 0;
    const rate = 83.1; // demo USD→INR
    return `(demo) ${amt} ${a.from ?? "USD"} ≈ ${(amt * rate).toFixed(2)} ${a.to ?? "INR"} (rate ${rate}).`;
  },
};

function hash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h << 5) - h + s.charCodeAt(i);
  return h;
}

export interface AgentTurn {
  action: SimResult["action"];
  reply: string;
  calls?: { name: string; arguments: Record<string, unknown> }[];
  observation?: string;
  rationale: string;
}

/** One conversational turn using the offline sim brain. */
export function runAgentTurnSim(query: string, tools: Tool[] = AGENT_TOOLS): AgentTurn {
  const r = simulate(tools, query);
  if (r.action === "call" && r.calls?.length) {
    const observation = r.calls
      .map((c) => {
        const exec = EXECUTORS[c.name] ?? (() => "done (demo)");
        return `${c.name} → ${exec(c.arguments)}`;
      })
      .join("  ·  ");
    const names = r.calls.map((c) => `\`${c.name}\``).join(", ");
    return { action: "call", calls: r.calls, observation, rationale: r.rationale,
             reply: `Done — I called ${names}. ${observation}` };
  }
  // refuse / clarify are first-class outcomes (the moat)
  return { action: r.action, reply: r.message ?? "…", rationale: r.rationale };
}
