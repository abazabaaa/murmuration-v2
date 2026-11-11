#!/usr/bin/env node
// Refuse personal names and home-directory paths in anything that would be published.
//
//   node scripts/privacy-check.js              every tracked file in the working tree
//   node scripts/privacy-check.js --staged     the index (pre-commit hook)
//   node scripts/privacy-check.js --push A B   every blob, message and author in B, not in A or a remote (pre-push hook)
//   node scripts/privacy-check.js --history    every blob, message and author reachable from any ref
'use strict';
const fs = require('node:fs'), os = require('node:os'), path = require('node:path'), zlib = require('node:zlib');
const cp = require('node:child_process');

const root = cp.execFileSync('git', ['rev-parse', '--show-toplevel'], { encoding: 'utf8' }).trim();
const git = (args, opts = {}) => cp.execFileSync('git', args, { cwd: root, maxBuffer: 1 << 30, ...opts });

const escape = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const words = new Set();
const login = os.userInfo().username;
if (login && !['runner', 'root', 'user'].includes(login)) words.add(login);
const wordFile = path.join(root, '.private-words');
if (fs.existsSync(wordFile)) {
  for (const w of fs.readFileSync(wordFile, 'utf8').split('\n')) {
    if (w.trim() && !w.startsWith('#')) words.add(w.trim());
  }
}
for (const w of (process.env.PRIVATE_WORDS || '').split(/[\n,]/)) {
  if (w.trim()) words.add(w.trim());
}
const home = String.raw`(?:/Users/|/home/|[A-Za-z]:\\Users\\|-Users-|-home-)(?!(?:Shared|user|redactedname)\b)[A-Za-z0-9._-]+`;
const pattern = new RegExp([home, ...[...words].map(w => escape(w).replace(/\\? /g, '[\\s._-]?'))].join('|'), 'gi');

function zstdAll(buf) {
  if (buf.length > 9 && buf.readUInt32LE(buf.length - 4) === 0x8f92eab1) {
    const n = buf.readUInt32LE(buf.length - 9), size = buf[buf.length - 5] & 0x80 ? 12 : 8, table = buf.length - 9 - n * size;
    const frames = [];
    for (let i = 0, at = 0; i < n; i++) {
      const c = buf.readUInt32LE(table + i * size);
      frames.push(zlib.zstdDecompressSync(buf.subarray(at, at + c)));
      at += c;
    }
    return Buffer.concat(frames);
  }
  return cp.execFileSync('zstd', ['-dcq'], { input: buf, maxBuffer: 1 << 30 });
}

function texts(buf) {
  const out = [buf];
  try {
    if (buf[0] === 0x28 && buf[1] === 0xb5 && buf[2] === 0x2f && buf[3] === 0xfd) out.push(zstdAll(buf));
    else if (buf[0] === 0x1f && buf[1] === 0x8b) out.push(zlib.gunzipSync(buf));
    else if (buf.length >= 8 && buf.readUInt32BE(0) === 0x89504e47) {
      for (let i = 8; i + 12 <= buf.length;) {
        const n = buf.readUInt32BE(i), type = buf.toString('latin1', i + 4, i + 8), data = buf.subarray(i + 8, i + 8 + n);
        if (type === 'zTXt') out.push(zlib.inflateSync(data.subarray(data.indexOf(0) + 2)));
        if (type === 'iTXt') {
          const k = data.indexOf(0);
          if (data[k + 1] === 1) {
            let j = k + 3;
            j = data.indexOf(0, j) + 1;
            j = data.indexOf(0, j) + 1;
            out.push(zlib.inflateSync(data.subarray(j)));
          }
        }
        i += 12 + n;
      }
    }
  } catch (e) {
    out.push(Buffer.from(`unreadable compressed data: ${e.message}`));
  }
  return out;
}

const hits = [];
function scan(label, buf) {
  for (const t of texts(buf)) {
    const s = t.toString('latin1');
    for (const m of s.matchAll(pattern)) {
      hits.push(`${label}: ${m[0]}`);
      if (hits.length > 200) return;
    }
  }
}

function scanObjects(revArgs) {
  const list = git(['rev-list', '--objects', ...revArgs], { encoding: 'utf8' }).split('\n').filter(Boolean)
    .map(l => { const i = l.indexOf(' '); return i < 0 ? null : [l.slice(0, i), l.slice(i + 1)]; }).filter(Boolean);
  scanBlobs(list);
  for (const line of git(['log', '--format=%H%x00%an <%ae>%x00%cn <%ce>%x00%B%x01', ...revArgs], { encoding: 'utf8' }).split('\x01')) {
    const [sha, ...rest] = line.trim().split('\0');
    if (sha) scan(`commit ${sha.slice(0, 12)}`, Buffer.from(rest.join('\n')));
  }
}

function scanBlobs(list) {
  const seen = new Set(), want = list.filter(([sha]) => !seen.has(sha) && seen.add(sha));
  if (!want.length) return;
  const types = git(['cat-file', '--batch-check=%(objecttype)'], { input: want.map(w => w[0]).join('\n') + '\n', encoding: 'utf8' }).split('\n');
  const blobs = want.filter((_, i) => types[i] === 'blob');
  if (!blobs.length) return;
  const out = git(['cat-file', '--batch'], { input: blobs.map(b => b[0]).join('\n') + '\n' });
  let at = 0;
  for (const [sha, file] of blobs) {
    const nl = out.indexOf(10, at), size = Number(out.toString('latin1', at, nl).split(' ')[2]);
    scan(`${file} (${sha.slice(0, 12)})`, out.subarray(nl + 1, nl + 1 + size));
    at = nl + 2 + size;
  }
}

const args = process.argv.slice(2);
if (args[0] === '--staged') {
  const list = git(['ls-files', '-s', '-z'], { encoding: 'utf8' }).split('\0').filter(Boolean)
    .map(l => { const [meta, file] = l.split('\t'); return [meta.split(' ')[1], file]; });
  scanBlobs(list);
} else if (args[0] === '--push') {
  const [base, tip] = args.slice(1);
  const zero = /^0+$/;
  if (!tip || zero.test(tip)) process.exit(0);
  scanObjects([tip, '--not', '--remotes', ...(base && !zero.test(base) ? [base] : [])]);
} else if (args[0] === '--history') {
  const hasCommits = git(['rev-parse', '--branches', '--tags'], { encoding: 'utf8' }).trim();
  if (hasCommits) scanObjects(['--all']);
} else {
  for (const file of git(['ls-files', '-z'], { encoding: 'utf8' }).split('\0').filter(Boolean)) {
    const full = path.join(root, file);
    if (fs.existsSync(full) && fs.statSync(full).isFile()) scan(file, fs.readFileSync(full));
  }
}

if (hits.length) {
  console.error(`privacy-check: ${hits.length} personal name or home path found${hits.length > 200 ? ' (first 200)' : ''}:`);
  for (const h of hits) console.error('  ' + h);
  console.error('Write paths relative to the repository or with ~.');
  process.exit(1);
}
console.log(`privacy-check: clean (${words.size} private word${words.size === 1 ? '' : 's'} + home-path pattern)`);
