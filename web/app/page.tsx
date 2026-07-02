import { Nav } from "@/components/nav";
import { Hero } from "@/components/hero";
import { Marquee } from "@/components/marquee";
import { Behaviors } from "@/components/behaviors";
import { Playground } from "@/components/playground";
import { Pipeline } from "@/components/pipeline";
import { Benchmarks } from "@/components/benchmarks";
import { DataViz } from "@/components/dataviz";
import { CTA } from "@/components/cta";
import { Footer } from "@/components/footer";
import { ScrollProgress } from "@/components/ui/scroll-progress";

export default function Page() {
  return (
    <main className="relative min-h-screen overflow-x-hidden">
      <ScrollProgress />
      <div className="aurora" aria-hidden />
      <div className="grid-fade" aria-hidden />
      <Nav />
      <Hero />
      <Marquee />
      <Behaviors />
      <Playground />
      <Pipeline />
      <Benchmarks />
      <DataViz />
      <CTA />
      <Footer />
    </main>
  );
}
