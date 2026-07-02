import { Cpu } from "lucide-react";
import { cn } from "@/lib/utils";

/** Single brand lockup — used by nav + footer so the mark never drifts (matches app/icon.tsx). */
export function Logo({ className }: { className?: string }) {
  return (
    <span className={cn("flex items-center gap-2.5 font-display text-lg font-bold tracking-tight", className)}>
      <span className="grid h-8 w-8 place-items-center rounded-lg border border-run/40 bg-run/15">
        <Cpu className="h-4 w-4 text-run" aria-hidden />
      </span>
      AutoScientist<span className="text-run">·</span>ToolCaller
    </span>
  );
}
