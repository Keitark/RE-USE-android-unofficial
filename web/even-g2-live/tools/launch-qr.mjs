import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import QRCode from 'qrcode';
import { launchConfiguration } from '../src/capture-session.js';

const args = process.argv.slice(2);
const option = name => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : null; };
const url = option('--url');
if (!url) throw new Error('Usage: npm run qr -- --url "http://127.0.0.1:8765/#token=<companion token>" [--out launch-qr.svg]');
launchConfiguration(url);
const out = resolve(option('--out') ?? 'launch-qr.svg');
await writeFile(out, await QRCode.toString(url, { type: 'svg', errorCorrectionLevel: 'M', margin: 4 }));
console.log(`QR written to ${out}`);
console.log('The QR payload is the unmodified launch URL. Scan it inside the official Even Developer Hub, or paste that same URL into Prototype mode.');
