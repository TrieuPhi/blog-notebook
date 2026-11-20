import { Link } from "react-router-dom";
import type { Post } from "../lib/posts";

interface PostCardProps {
  p: Post;
  showPhaseBadge?: boolean;
}

function artifactIcon(artifact: string) {
  if (artifact.includes("Code")) return "💻";
  if (artifact.includes("Checklist")) return "📋";
  if (artifact.includes("Tool")) return "🛠️";
  if (artifact.includes("Prompt")) return "💬";
  if (artifact.includes("Cheatsheet")) return "📐";
  return "⚡";
}

export default function PostCard({ p, showPhaseBadge = true }: PostCardProps) {
  return (
    <Link
      to={`/article/${p.slug}`}
      className="group flex flex-col gap-2.5 bg-bg2 border border-line rounded-lg p-4 shadow-sm hover:border-accent hover:-translate-y-0.5 transition duration-150"
    >
      {/* Top Metadata Row */}
      <div className="font-mono text-[11px] text-fgdim flex gap-2 flex-wrap items-center">
        <span className="text-accent font-semibold">{p.category}</span>
        <span>•</span>
        <span>{p.date ? p.date.replace(/-/g, ".") : "Draft"}</span>
        <span>•</span>
        <span>{p.readingTime || 10} min read</span>

        {showPhaseBadge && p.phaseNumber && (
          <span className="font-mono text-[10px] border border-accent/40 bg-accent/10 text-accent rounded-full px-2 py-0.5 font-medium ml-auto sm:ml-0">
            Phase {p.phaseNumber}
          </span>
        )}

        {p.status && (
          <span className="font-mono text-[10px] border border-line text-fgdim rounded-full px-2 py-0.5 ml-auto">
            {p.status}
          </span>
        )}
      </div>

      {/* Title */}
      <h3 className="text-[16px] font-semibold leading-snug m-0 group-hover:text-accent transition-colors">
        {p.title}
      </h3>

      {/* Core Motto */}
      {p.motto && (
        <div className="border-l-2 border-accent bg-bg/50 px-3 py-1.5 rounded-r my-0.5">
          <p className="text-[12.5px] italic text-fg/85 m-0 leading-relaxed font-sans">
            &ldquo;{p.motto}&rdquo;
          </p>
        </div>
      )}

      {/* Description */}
      <p className="text-[13.5px] text-fgdim flex-1 m-0 leading-relaxed">
        {p.description}
      </p>

      {/* Actionable Artifacts Badges */}
      {p.artifacts && p.artifacts.length > 0 && (
        <div className="flex gap-1.5 flex-wrap items-center pt-1 border-t border-line/50">
          <span className="font-mono text-[10px] text-fgdim uppercase tracking-wider mr-1">
            Artifacts:
          </span>
          {p.artifacts.map((art) => (
            <span
              key={art}
              className="font-mono text-[10.5px] border border-amber/35 bg-amber/10 text-amber rounded px-2 py-0.5 inline-flex items-center gap-1 font-medium"
            >
              <span>{artifactIcon(art)}</span>
              <span>{art}</span>
            </span>
          ))}
        </div>
      )}

      {/* Tags */}
      {p.tags && p.tags.length > 0 && (
        <div className="flex gap-1.5 flex-wrap pt-0.5">
          {p.tags.map((t) => (
            <span
              key={t}
              className="font-mono text-[10.5px] border border-line bg-bg/40 rounded-full px-2 py-0.5 text-fgdim"
            >
              #{t}
            </span>
          ))}
        </div>
      )}
    </Link>
  );
}
