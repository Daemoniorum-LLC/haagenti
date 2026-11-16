// Copied from scriptorium tomes/bigint/tests/ at 5da585b (LARES-932): the estate's wasm runner pattern.
// Run a `sigil wasm` module under node against qliphoth's runtime/sigil_runtime.js, as
// sigil-lang's parser/tests/wasm_differential/run-wasm.mjs does at 6fbf5c455 (its
// program-lines.mjs inlined): through the runtime's own instantiateWasm, which hands the
// runtime the module's memory.
//
// usage: node run-wasm.mjs <sigil_runtime.js> <file.wasm>
//
// stdout: what the program printed, one line per print (a `[sigil]` tag, from an older
//         runtime, is dropped), then `main returned <n>`.
// exit:   0 ran and main gave no status or 0; 1 ran and main gave a nonzero status (a `main` that
//         returns Err gives 1, and run and AOT exit 1: SIGIL-2036); 2 could not run (a missing
//         import, no `main`); 3 trapped. tests/run-wasm-check.mjs checks the first two.
import fs from 'fs';
import path from 'path';

const [runtimePath, wasmPath] = process.argv.slice(2);
const rt = await import(path.resolve(runtimePath));
const bytes = fs.readFileSync(wasmPath);
const imports = rt.createImports();
const missing = WebAssembly.Module.imports(new WebAssembly.Module(bytes))
  .filter(i => !(imports[i.module] && i.name in imports[i.module]));
if (missing.length) {
  console.error('missing imports:', missing.length, missing.map(i => `${i.module}.${i.name}`).join(' '));
  process.exit(2);
}

const out = [];
const log = console.log;
console.log = (...args) => {
  if (args[0] === '[sigil]') out.push(args.slice(1).map(String).join(' '));
  else out.push(args.map(String).join(' '));
};
const flush = () => { console.log = log; for (const line of out) log(line); };
const flushRuntime = () => { if (typeof rt.flushConsole === 'function') rt.flushConsole(); };

let instance;
try {
  instance = rt.instantiateWasm(bytes);
} catch (e) {
  flush();
  console.error('instantiate:', e && e.message ? e.message : String(e));
  process.exit(2);
}
if (typeof instance.exports.main !== 'function') {
  flush();
  console.error('no exported main');
  process.exit(2);
}
let ret;
try {
  ret = instance.exports.main();
} catch (e) {
  flushRuntime();
  flush();
  console.error('trap:', e && e.message ? e.message : String(e));
  process.exit(3);
}
flushRuntime();
flush();
log(`main returned ${ret === undefined ? '()' : ret}`);
// A nonzero status is a failure, as on run and AOT: never exit 0 on it (SIGIL-2036).
if ((typeof ret === 'number' || typeof ret === 'bigint') && Number(ret) !== 0) {
  console.error(`main returned ${ret}: exit 1, as run and AOT exit on an Err (SIGIL-2036)`);
  process.exit(1);
}
