import { Nav } from "@/components/nav";
import { Hero } from "@/components/hero";
import { Marquee } from "@/components/marquee";
import { Behaviors } from "@/components/behaviors";
import { Playground } from "@/components/playground";
import { AgentSection } from "@/components/agent-section";
import { Pipeline } from "@/components/pipeline";
import { Benchmarks } from "@/components/benchmarks";
import { DataViz } from "@/components/dataviz";
import { CTA } from "@/components/cta";
import { Footer } from "@/components/footer";
import { ScrollProgress } from "@/components/ui/scroll-progress";

export default function Page() {
  return (
    <>
      <a href="#top" className="skip-link rounded-lg bg-run px-4 py-2 text-sm font-semibold text-slate-950">
        Skip to content
      </a>
      <ScrollProgress />
      <div className="aurora" aria-hidden />
      <div className="grid-fade" aria-hidden />
      <div className="grain" aria-hidden />
      <Nav />
      <main className="relative min-h-screen overflow-x-hidden">
        <Hero />
      <Marquee />
      <Behaviors />
      <Playground />
      <AgentSection />
      <Pipeline />
      <Benchmarks />
      <DataViz />
      <CTA />
      </main>
      <Footer />
    </>
  );
}
