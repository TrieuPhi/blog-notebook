import { NavLink } from "react-router-dom";
import { useState } from "react";
import ThemeToggle from "./ThemeToggle";

const LINKS = [
  { to: "/", label: "Home", end: true },
  { to: "/blog", label: "Blog" },
  { to: "/topics", label: "Topics" },
  { to: "/graph", label: "Graph" },
  { to: "/projects", label: "Projects" },
  { to: "/about", label: "About" },
];

export default function Nav({ onSearch }: { onSearch: () => void }) {
  const [open, setOpen] = useState(false);
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `font-mono text-[13.5px] px-2.5 py-1.5 rounded ${isActive ? "text-fg bg-bg2 border border-line" : "text-fgdim hover:text-fg"}`;

  return (
    <header className="sticky top-0 z-40 border-b border-line backdrop-blur bg-bg/90">
      <div className="max-w-[1160px] mx-auto flex items-center gap-5 px-5 py-3">
        <div className="font-mono font-bold text-[15px] flex items-center gap-1.5">
          <span className="w-[7px] h-[7px] rounded-full bg-accent blink-cursor" /> set.log
        </div>
        <nav className="hidden md:flex gap-0.5 ml-2">
          {LINKS.map((l) => (
            <NavLink key={l.to} to={l.to} end={l.end} className={linkClass}>{l.label}</NavLink>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <button onClick={onSearch} className="hidden md:inline-flex border border-line bg-bg2 rounded px-3 py-1.5 font-mono text-xs text-fgdim hover:text-fg">
            ⌘K search
          </button>
          <button onClick={() => setOpen((o) => !o)} className="md:hidden border border-line bg-bg2 rounded px-3 py-1.5 text-fgdim">☰</button>
          <ThemeToggle />
          <a href="https://github.com" target="_blank" rel="noopener" className="hidden md:inline-flex border border-line bg-bg2 rounded px-3 py-1.5 font-mono text-xs text-fgdim hover:text-fg">GitHub</a>
          <a href="https://linkedin.com" target="_blank" rel="noopener" className="hidden md:inline-flex border border-line bg-bg2 rounded px-3 py-1.5 font-mono text-xs text-fgdim hover:text-fg">LinkedIn</a>
        </div>
      </div>
      {open && (
        <div className="md:hidden border-t border-line bg-bg2">
          {LINKS.map((l) => (
            <NavLink key={l.to} to={l.to} end={l.end} onClick={() => setOpen(false)} className="block px-5 py-3 font-mono text-sm border-t border-line first:border-t-0">
              {l.label}
            </NavLink>
          ))}
          <button onClick={() => { setOpen(false); onSearch(); }} className="block w-full text-left px-5 py-3 font-mono text-sm border-t border-line">
            Search (⌘K)
          </button>
        </div>
      )}
    </header>
  );
}
