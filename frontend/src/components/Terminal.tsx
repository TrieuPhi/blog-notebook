export default function Terminal() {
  return (
    <div className="bg-[#101210] text-[#c9d6c9] rounded-lg px-5 py-4 font-mono text-[13px] border border-[#2a2e2a] shadow-sm">
      <div className="my-1"><span className="text-accent">$</span> whoami</div>
      <div className="my-1">AI Engineer</div>
      <div className="my-1"><span className="text-accent">$</span> interests</div>
      <div className="my-1">AI / LLM / RAG / MLOps / Systems</div>
      <div className="my-1"><span className="text-accent">$</span> currently_learning</div>
      <div className="my-1">Evaluation · SLM · Fine-tuning · Infra<span className="inline-block w-[7px] h-[14px] bg-accent align-[-2px] blink-cursor" /></div>
    </div>
  );
}
