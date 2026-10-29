import fs from "fs";
import path from "path";
import { pathToFileURL } from "url";

const katexPath = path.resolve("c:/project/blog/blog-notebook/frontend/node_modules/katex/dist/katex.mjs");
const katexModule = await import(pathToFileURL(katexPath).href);
const katex = katexModule.default || katexModule;

const postsDir = path.resolve("c:/project/blog/blog-notebook/frontend/posts");
const files = fs.readdirSync(postsDir).filter(f => f.endsWith(".md"));

console.log("======================================================================");
console.log("       ADVERSARIAL REAL KATEX COMPILER VERIFICATION HARNESS");
console.log("======================================================================");

let totalFormulas = 0;
let failedFormulas = 0;

for (const file of files) {
  const filePath = path.join(postsDir, file);
  const content = fs.readFileSync(filePath, "utf-8");

  // Remove frontmatter
  const fmMatch = content.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
  const body = fmMatch ? fmMatch[2] : content;

  // Mask code blocks
  const bodyNoCode = body.replace(/```[\s\S]*?```/g, m => "\n".repeat((m.match(/\n/g) || []).length));
  // Mask inline code
  const bodyNoInlineCode = bodyNoCode.replace(/`[^`\n]+`/g, m => " ".repeat(m.length));

  console.log(`\nAuditing KaTeX formulas in: ${file}`);
  let fileFormulas = 0;
  let fileFailures = 0;

  // 1. Display math $$...$$
  const displayRegex = /\$\$([\s\S]*?)\$\$/g;
  let match;
  while ((match = displayRegex.exec(bodyNoInlineCode)) !== null) {
    const rawFormula = match[1].trim();
    if (!rawFormula) continue;
    fileFormulas++;
    totalFormulas++;
    try {
      katex.renderToString(rawFormula, { displayMode: true, throwOnError: true });
    } catch (err) {
      fileFailures++;
      failedFormulas++;
      console.error(`  [DISPLAY MATH ERROR] in ${file}:`);
      console.error(`  Formula: ${rawFormula.slice(0, 80)}...`);
      console.error(`  Error: ${err.message}`);
    }
  }

  // Mask display math before inline math
  const bodyNoDisplay = bodyNoInlineCode.replace(/\$\$[\s\S]*?\$\$/g, m => " ".repeat(m.length));

  // 2. Inline math $...$
  const lines = bodyNoDisplay.split("\n");
  for (let lineNo = 1; lineNo <= lines.length; lineNo++) {
    const line = lines[lineNo - 1];
    let i = 0;
    while (i < line.length) {
      if (line[i] === "\\" && i + 1 < line.length && line[i + 1] === "$") {
        i += 2;
        continue;
      }
      if (line[i] === "$") {
        let closing = -1;
        for (let j = i + 1; j < line.length; j++) {
          if (line[j] === "\\" && j + 1 < line.length && line[j + 1] === "$") {
            j++;
            continue;
          }
          if (line[j] === "$") {
            closing = j;
            break;
          }
        }
        if (closing !== -1) {
          const formula = line.slice(i + 1, closing);
          if (formula.trim()) {
            fileFormulas++;
            totalFormulas++;
            try {
              katex.renderToString(formula, { displayMode: false, throwOnError: true });
            } catch (err) {
              fileFailures++;
              failedFormulas++;
              console.error(`  [INLINE MATH ERROR] line ${lineNo} in ${file}:`);
              console.error(`  Formula: $${formula}$`);
              console.error(`  Error: ${err.message}`);
            }
          }
          i = closing + 1;
          continue;
        }
      }
      i++;
    }
  }

  if (fileFailures === 0) {
    console.log(`  [PASS] All ${fileFormulas} formulas compiled cleanly with KaTeX engine.`);
  } else {
    console.log(`  [FAIL] ${fileFailures} / ${fileFormulas} formulas failed KaTeX compilation.`);
  }
}

console.log("\n----------------------------------------------------------------------");
console.log(`Total formulas verified: ${totalFormulas} | Failures: ${failedFormulas}`);
console.log("----------------------------------------------------------------------");

if (failedFormulas > 0) {
  process.exit(1);
} else {
  console.log("VERDICT: REAL KATEX COMPILATION PASSED 100%");
  process.exit(0);
}
