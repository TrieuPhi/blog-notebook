import { useEffect, useState } from "react";
import { Routes, Route, useLocation } from "react-router-dom";
import Nav from "./components/Nav";
import SearchModal from "./components/SearchModal";
import Home from "./pages/Home";
import Blog from "./pages/Blog";
import Article from "./pages/Article";
import Topics from "./pages/Topics";
import GraphPage from "./pages/GraphPage";
import Projects from "./pages/Projects";
import About from "./pages/About";

export default function App() {
  const [searchOpen, setSearchOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); setSearchOpen((o) => !o); }
      if (e.key === "Escape") setSearchOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => { window.scrollTo({ top: 0 }); }, [location.pathname]);

  return (
    <>
      <Nav onSearch={() => setSearchOpen(true)} />
      <main className="max-w-[1160px] mx-auto px-5 pb-20">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/blog" element={<Blog />} />
          <Route path="/article/:slug" element={<Article />} />
          <Route path="/topics" element={<Topics />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="/projects" element={<Projects />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
      <footer className="border-t border-line mt-10">
        <div className="max-w-[1160px] mx-auto px-5 py-5 flex justify-between font-mono text-[11.5px] text-fgdim flex-wrap gap-2">
          <span>set.log — personal knowledge base</span>
          <span>built with React · Vite · Markdown</span>
        </div>
      </footer>
      <SearchModal open={searchOpen} onClose={() => setSearchOpen(false)} />
    </>
  );
}
