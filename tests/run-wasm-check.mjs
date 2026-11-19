// Copied from scriptorium tomes/bigint/tests/ at 5da585b (LARES-932): the estate's wasm runner pattern.
// Check that tests/run-wasm.mjs exits with main's status (SIGIL-2036): 1 when main returns a nonzero
// status (a Sigil `main` that returns Err gives 1 on wasm; run and AOT exit 1), 0 otherwise. A failing
// program must never read as a pass. No compiler and no qliphoth: a stub runtime and hand-made modules.
//
// usage: node run-wasm-check.mjs [<run-wasm.mjs>]   (default: the one beside this file)
// stdout: one `ok` or `WRONG` line per case. exit: 0 every case right, 1 otherwise.
import fs from 'fs';
import os from 'os';
import path from 'path';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';

const here = path.dirname(fileURLToPath(import.meta.url));
const runner = path.resolve(process.argv[2] || path.join(here, 'run-wasm.mjs'));
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'run-wasm-check-'));

// A runtime with no imports: run-wasm.mjs needs only createImports and instantiateWasm.
fs.writeFileSync(path.join(dir, 'stub-runtime.mjs'),
  'export function createImports() { return {}; }\n' +
  'export function instantiateWasm(bytes) { return new WebAssembly.Instance(new WebAssembly.Module(bytes), {}); }\n');

const head = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
const exportMain = [0x03, 0x02, 0x01, 0x00, 0x07, 0x08, 0x01, 0x04, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00];
// main: () → ()
const unit = [...head, 0x01, 0x04, 0x01, 0x60, 0x00, 0x00, ...exportMain, 0x0a, 0x04, 0x01, 0x02, 0x00, 0x0b];
// main: () → i32, returning n (0 ≤ n < 64: one byte of signed LEB128)
const status = n => [...head, 0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f, ...exportMain,
  0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, n, 0x0b];

const cases = [
  ['main returns nothing', unit, 0],
  ['main returns 0 (Ok)', status(0), 0],
  ['main returns 1 (Err): the must-fail', status(1), 1],
  ['main returns 7', status(7), 1],
];
let wrong = 0;
for (const [label, bytes, want] of cases) {
  const wasm = path.join(dir, 'm.wasm');
  fs.writeFileSync(wasm, Buffer.from(bytes));
  const r = spawnSync(process.execPath, [runner, path.join(dir, 'stub-runtime.mjs'), wasm], { encoding: 'utf8' });
  const got = r.status;
  if (got === want) console.log(`ok    exit ${got}  ${label}`);
  else { console.log(`WRONG exit ${got} (wanted ${want})  ${label}: ${(r.stderr || '').split('\n')[0]}`); wrong = 1; }
}
fs.rmSync(dir, { recursive: true, force: true });
process.exit(wrong);
