"use client";

import { useEffect, useRef, useState } from "react";
import { Bot, Loader2, Mic, Send, Square, User, Volume2, VolumeX, Wrench } from "lucide-react";
import { runAgentTurnSim } from "@/lib/agent-sim";
import { useVoice } from "@/lib/use-voice";
import { cn } from "@/lib/utils";

type Msg = { role: "user" | "agent" | "tool"; text: string; tag?: string };

const SUGGESTIONS = [
  "What's the weather in Mumbai?",
  "Book me a flight to Goa.",
  "Write me a poem about the monsoon.",
  "Convert 100 USD to INR.",
];

const HELLO =
  "Hi! Give me a goal — type it or tap the mic 🎤. I'll call the right tool, or **refuse** / **ask** " +
  "when I shouldn't guess. Try a suggestion below.";

export function AgentConsole() {
  const [msgs, setMsgs] = useState<Msg[]>([{ role: "agent", text: HELLO }]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [speakOn, setSpeakOn] = useState(false);
  const [brain, setBrain] = useState<"loading" | "in-browser" | "claude">("loading");
  const { supported, listening, listen, stopListening, speak } = useVoice();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/agent")
      .then((r) => r.json())
      .then((d) => setBrain(d?.available ? "claude" : "in-browser"))
      .catch(() => setBrain("in-browser"));
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [msgs, busy]);

  async function send(text: string) {
    const q = text.trim();
    if (!q || busy) return;
    setInput("");
    setMsgs((m) => [...m, { role: "user", text: q }]);
    setBusy(true);
    try {
      if (brain === "claude") {
        const r = await fetch("/api/agent", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ message: q }),
        }).then((res) => res.json());
        if (r?.error) throw new Error(r.error);
        const trace = (r.trace || []) as { tool: string; output: string }[];
        setMsgs((m) => [
          ...m,
          ...trace.map((t) => ({ role: "tool" as const, text: `${t.tool} → ${t.output}` })),
          { role: "agent", text: r.reply, tag: "Claude" },
        ]);
        if (speakOn) speak(r.reply);
      } else {
        const r = runAgentTurnSim(q);
        setMsgs((m) => [
          ...m,
          ...(r.observation ? [{ role: "tool" as const, text: r.observation }] : []),
          { role: "agent", text: r.reply, tag: r.action },
        ]);
        if (speakOn) speak(r.reply);
      }
    } catch (e: any) {
      setMsgs((m) => [...m, { role: "agent", text: `⚠️ ${e?.message || "something went wrong"}` }]);
    } finally {
      setBusy(false);
    }
  }

  function onMic() {
    if (listening) {
      stopListening();
      return;
    }
    listen((text) => {
      setInput(text);
      void send(text);
    });
  }

  const brainLabel = brain === "claude" ? "Claude · live" : brain === "in-browser" ? "in-browser" : "…";

  return (
    <div className="mx-auto flex h-[540px] max-w-2xl flex-col overflow-hidden rounded-2xl glass">
      {/* header */}
      <div className="flex items-center justify-between border-b border-border/50 px-4 py-3">
        <span className="flex items-center gap-2 font-display text-sm font-semibold">
          <Bot className="h-4 w-4 text-run" /> Agent
          <span className="rounded-full border border-run/40 bg-run/10 px-2 py-0.5 text-[10px] font-medium text-run">
            {brainLabel}
          </span>
        </span>
        <button
          type="button"
          onClick={() => setSpeakOn((v) => !v)}
          aria-pressed={speakOn}
          aria-label={speakOn ? "Turn voice replies off" : "Turn voice replies on"}
          className={cn("grid h-8 w-8 cursor-pointer place-items-center rounded-lg border transition-colors",
            speakOn ? "border-run/50 bg-run/10 text-run" : "border-border/60 text-muted-foreground hover:text-foreground")}
          title={supported.tts ? "Spoken replies" : "Speech synthesis unavailable in this browser"}
          disabled={!supported.tts}
        >
          {speakOn ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
        </button>
      </div>

      {/* messages */}
      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {msgs.map((m, i) => (
          <Bubble key={i} m={m} />
        ))}
        {busy && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin text-run" /> thinking…
          </div>
        )}
      </div>

      {/* suggestions */}
      <div className="flex flex-wrap gap-2 px-4 pb-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => send(s)}
            disabled={busy}
            className="cursor-pointer rounded-full border border-border/60 px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:border-cyan/50 hover:text-foreground disabled:opacity-50"
          >
            {s}
          </button>
        ))}
      </div>

      {/* input */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
        className="flex items-center gap-2 border-t border-border/50 p-3"
      >
        {supported.stt && (
          <button
            type="button"
            onClick={onMic}
            aria-label={listening ? "Stop listening" : "Speak your request"}
            className={cn("grid h-10 w-10 shrink-0 cursor-pointer place-items-center rounded-xl border transition-colors",
              listening ? "border-run/60 bg-run/15 text-run" : "border-border/60 text-muted-foreground hover:text-foreground")}
          >
            {listening ? <Square className="h-4 w-4 animate-pulse" /> : <Mic className="h-4 w-4" />}
          </button>
        )}
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={listening ? "Listening…" : "Ask the agent to do something…"}
          aria-label="Message the agent"
          className="min-w-0 flex-1 rounded-xl border border-border/60 bg-black/10 px-3.5 py-2.5 text-sm text-foreground outline-none focus:border-cyan/50"
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          aria-label="Send"
          className="grid h-10 w-10 shrink-0 cursor-pointer place-items-center rounded-xl bg-run text-slate-950 transition-colors hover:bg-run/90 disabled:opacity-50"
        >
          <Send className="h-4 w-4" />
        </button>
      </form>
    </div>
  );
}

function Bubble({ m }: { m: Msg }) {
  if (m.role === "tool") {
    return (
      <div className="flex items-start gap-2 pl-1 text-xs text-muted-foreground">
        <Wrench className="mt-0.5 h-3.5 w-3.5 shrink-0 text-cyan" />
        <code className="whitespace-pre-wrap break-words">{m.text}</code>
      </div>
    );
  }
  const isUser = m.role === "user";
  return (
    <div className={cn("flex gap-2.5", isUser && "flex-row-reverse")}>
      <span
        className={cn("grid h-7 w-7 shrink-0 place-items-center rounded-lg border",
          isUser ? "border-cyan/40 bg-cyan/10 text-cyan" : "border-run/40 bg-run/10 text-run")}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </span>
      <div
        className={cn("max-w-[80%] rounded-2xl px-3.5 py-2 text-sm",
          isUser ? "bg-cyan/10 text-foreground" : "inset-well text-foreground/90")}
      >
        <span className="whitespace-pre-wrap break-words">{m.text}</span>
        {m.tag && !isUser && (
          <span className="ml-2 align-middle text-[10px] uppercase tracking-wide text-muted-foreground">· {m.tag}</span>
        )}
      </div>
    </div>
  );
}
