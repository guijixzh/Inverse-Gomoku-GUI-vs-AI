import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(here, "..");
const repoRoot = path.resolve(webRoot, "..");
const publicDir = path.join(webRoot, "public");
const pyodideSrc = path.join(webRoot, "node_modules", "pyodide");
const pyodideDst = path.join(publicDir, "pyodide");
const wheelDst = path.join(publicDir, "antifive.whl");

fs.mkdirSync(publicDir, { recursive: true });
fs.mkdirSync(pyodideDst, { recursive: true });

const pkg = JSON.parse(fs.readFileSync(path.join(pyodideSrc, "package.json"), "utf8"));
const version = pkg.version;
const CDN = `https://cdn.jsdelivr.net/pyodide/v${version}/full/`;

for (const name of [
  "pyodide.mjs",
  "pyodide.asm.mjs",
  "pyodide.asm.wasm",
  "python_stdlib.zip",
  "pyodide-lock.json",
]) {
  fs.copyFileSync(path.join(pyodideSrc, name), path.join(pyodideDst, name));
}

const lock = JSON.parse(fs.readFileSync(path.join(pyodideSrc, "pyodide-lock.json"), "utf8"));
const numpyFile = lock.packages.numpy.file_name;
const numpyDst = path.join(pyodideDst, numpyFile);
if (!fs.existsSync(numpyDst)) {
  const cached = path.join(pyodideSrc, numpyFile);
  if (fs.existsSync(cached)) {
    fs.copyFileSync(cached, numpyDst);
  } else {
    const resp = await fetch(CDN + numpyFile);
    if (!resp.ok) throw new Error(`download numpy failed: ${resp.status}`);
    fs.writeFileSync(numpyDst, Buffer.from(await resp.arrayBuffer()));
  }
}
console.log(`pyodide ${version} + ${numpyFile} ready in public/pyodide/`);

if (process.env.ANTIFIVE_SKIP_WHEEL !== "1") {
  const srcDir = path.join(repoRoot, "src", "antifive");
  const newest = (dir) => {
    let latest = 0;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) latest = Math.max(latest, newest(full));
      else if (entry.name.endsWith(".py")) latest = Math.max(latest, fs.statSync(full).mtimeMs);
    }
    return latest;
  };
  const stale = process.env.ANTIFIVE_FORCE_WHEEL === "1"
    || !fs.existsSync(wheelDst) || fs.statSync(wheelDst).mtimeMs < newest(srcDir);
  if (stale) {
    const python = process.env.ANTIFIVE_PYTHON ?? "python";
    const tmp = path.join(webRoot, ".wheel-tmp");
    fs.rmSync(tmp, { recursive: true, force: true });
    fs.mkdirSync(tmp, { recursive: true });
    const r = spawnSync(python, ["-m", "pip", "wheel", repoRoot, "--no-deps", "-w", tmp], {
      stdio: "inherit",
    });
    const whl = r.status === 0 ? fs.readdirSync(tmp).find((f) => f.endsWith(".whl")) : null;
    if (!whl) {
      if (fs.existsSync(wheelDst)) {
        console.warn("wheel build failed; using existing public/antifive.whl");
      } else {
        throw new Error("wheel build failed; set ANTIFIVE_PYTHON or build the wheel manually");
      }
    } else {
      fs.copyFileSync(path.join(tmp, whl), wheelDst);
      console.log(`antifive wheel ready: public/antifive.whl (${whl})`);
    }
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}
