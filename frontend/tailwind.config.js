/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["class"],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        bg2: "var(--bg2)",
        fg: "var(--fg)",
        fgdim: "var(--fg-dim)",
        line: "var(--line)",
        accent: "var(--accent)",
        amber: "var(--amber)",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      borderRadius: { DEFAULT: "6px" },
    },
  },
  plugins: [],
};
