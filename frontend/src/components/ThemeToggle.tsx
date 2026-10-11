import { useEffect, useState } from "react";

const THEMES = ["light", "dark", "warm"] as const;
type Theme = (typeof THEMES)[number];

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => (localStorage.getItem("slog-theme") as Theme) || "light");
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme === "light" ? "" : theme);
    localStorage.setItem("slog-theme", theme);
  }, [theme]);
  return { theme, setTheme };
}

export default function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  return (
    <button
      className="border border-line bg-bg2 rounded px-3 py-1.5 font-mono text-xs text-fgdim hover:text-fg"
      onClick={() => setTheme(THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length])}
    >
      theme: {theme}
    </button>
  );
}
