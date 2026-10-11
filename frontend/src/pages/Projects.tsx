import { Link } from "react-router-dom";
import { PROJECTS, POSTS } from "../lib/posts";

export default function Projects() {
  return (
    <div>
      <div className="font-mono text-[11.5px] text-fgdim mb-2">PROJECTS / ENGINEERING LOG</div>
      <h2 className="text-[26px] font-bold mb-5">Projects</h2>
      <div className="grid sm:grid-cols-2 gap-4">
        {PROJECTS.map((pr) => (
          <div key={pr.name} className="border border-line bg-bg2 rounded p-4.5 shadow-sm p-[18px]">
            <div className="flex justify-between items-center">
              <h3 className="m-0">{pr.name}</h3>
              <span className="font-mono text-[10.5px] border border-accent text-accent rounded-full px-2 py-0.5">{pr.status}</span>
            </div>
            <p className="text-fgdim text-sm my-2.5">{pr.desc}</p>
            <div className="font-mono text-xs text-fgdim mt-2.5">{pr.stack}</div>
            <div className="flex gap-1.5 flex-wrap mt-2.5">
              {pr.posts.map((s) => {
                const p = POSTS.find((x) => x.slug === s);
                return p ? <Link key={s} to={`/article/${s}`} className="font-mono text-[10.5px] border border-line rounded-full px-2 py-0.5 text-accent">{p.title}</Link> : null;
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
