export default function About() {
  return (
    <div className="grid sm:grid-cols-[150px_1fr] gap-7 items-start">
      <div className="w-[120px] h-[120px] rounded bg-gradient-to-br from-accent to-amber flex items-center justify-center font-mono text-[28px] font-bold text-[color:var(--accent-fg)]">AI</div>
      <div>
        <div className="font-mono text-[11.5px] text-fgdim mb-2">ABOUT</div>
        <h2 className="text-[26px] font-bold">Hi, I'm the person behind set.log</h2>
        <p className="text-fgdim">
          I'm an AI Engineer. I spend most of my time building and evaluating retrieval-augmented systems, fine-tuning
          small language models, and shipping the infrastructure that lets them run reliably. This site is my working
          notebook — part study notes, part experiment log, part reference I go back to.
        </p>
        <h4 className="font-mono text-[11px] text-fgdim mt-6 mb-2">What I work on</h4>
        <p>RAG pipelines, evaluation frameworks (RAGAS and custom judges), SLM fine-tuning, and the MLOps plumbing — serving, monitoring, deployment — that keeps models usable outside a notebook.</p>
        <h4 className="font-mono text-[11px] text-fgdim mt-6 mb-2">Technologies</h4>
        <div className="flex gap-1.5 flex-wrap">
          {["Python", "PyTorch", "vLLM", "RAGAS", "LangChain", "FastAPI", "Docker", "Kubernetes"].map((t) => (
            <span key={t} className="font-mono text-[10.5px] border border-line rounded-full px-2 py-0.5 text-fgdim">{t}</span>
          ))}
        </div>
        <h4 className="font-mono text-[11px] text-fgdim mt-6 mb-2">Currently learning</h4>
        <p>Evaluation methodology for open-domain RAG, efficient fine-tuning of small models, and better ways to keep inference infra boring and predictable.</p>
        <h4 className="font-mono text-[11px] text-fgdim mt-6 mb-2">Contact</h4>
        <div className="flex gap-1.5 flex-wrap">
          <a href="https://github.com" target="_blank" className="font-mono text-[10.5px] border border-line rounded-full px-2 py-0.5 text-fgdim">GitHub ↗</a>
          <a href="https://linkedin.com" target="_blank" className="font-mono text-[10.5px] border border-line rounded-full px-2 py-0.5 text-fgdim">LinkedIn ↗</a>
        </div>
      </div>
    </div>
  );
}
