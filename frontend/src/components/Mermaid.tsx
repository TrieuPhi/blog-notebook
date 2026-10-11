import { useEffect, useRef, useState } from "react";

let mermaidInstance: any = null;

export default function Mermaid({ code }: { code: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!mermaidInstance) {
        const mod = await import("mermaid");
        mermaidInstance = mod.default;
        mermaidInstance.initialize({ startOnLoad: false, theme: "dark" });
      }
      try {
        const id = "m" + Math.random().toString(36).slice(2);
        const { svg } = await mermaidInstance.render(id, code);
        if (!cancelled) setSvg(svg);
      } catch {
        if (!cancelled) setSvg("<pre>Diagram failed to render</pre>");
      }
    })();
    return () => { cancelled = true; };
  }, [code]);

  return <div ref={ref} className="my-4 bg-[#101210] rounded-lg p-4 border border-line overflow-auto" dangerouslySetInnerHTML={{ __html: svg }} />;
}
