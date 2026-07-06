import { Mic, Volume2 } from "lucide-react";
import { Section, SectionHeader } from "@/components/section";
import { AgentConsole } from "@/components/agent-console";

export function AgentSection() {
  return (
    <Section id="agent">
      <SectionHeader
        eyebrow="try it · voice + agent"
        title={<>Tell the agent what to do — <span className="text-grad">by voice or text.</span></>}
        tone="run"
        className="mb-8"
      >
        A live agent that plans, calls the right tool, and <span className="text-foreground">refuses or asks</span>{" "}
        when it shouldn&rsquo;t guess — the exact discipline this dataset teaches. Runs in your browser (no key,
        no server); upgrades to a real <span className="text-foreground">Claude-powered</span> agent when an API
        key is configured. Tap the mic{" "}
        <Mic className="inline h-3.5 w-3.5 align-[-2px] text-run" aria-hidden /> to speak; toggle{" "}
        <Volume2 className="inline h-3.5 w-3.5 align-[-2px] text-run" aria-hidden /> for spoken replies.
      </SectionHeader>
      <AgentConsole />
    </Section>
  );
}
