import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// Source data is a local, verified snapshot; no model calls or network access.
const dir = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const sharp = require(process.env.CHART_SHARP_PATH || 'sharp');
const dataFile = resolve(dir, 'benchmark-charts.data.json');
const raw = readFileSync(dataFile);
const data = JSON.parse(raw);
const out = resolve(dir, '../final');
mkdirSync(out, { recursive: true });

const WIDTH = 1600;
const C = {
  paper: '#ffffff', ink: '#26333d', muted: '#66737c', grid: '#e8edf0',
  baseline: '#bbc3c9', jev: '#347f83', laya: '#bc8958', other: '#8190a1',
};
const order = ['jev', 'laya', 'bge', 'qwen06', 'qwen8'];
const labels = {
  baseline: 'Без реранжирования', jev: 'Jev 1.13', laya: 'Laya Multilingual',
  bge: 'BGE-reranker-v2-m3', qwen06: 'Qwen3-Reranker-0.6B', qwen8: 'Qwen3-Reranker-8B',
};
const datasets = [
  { id: 'scifact', label: 'SciFact', n: 300 },
  { id: 'fiqa', label: 'FiQA', n: 648 },
  { id: 'nfcorpus', label: 'NFCorpus', n: 323 },
];
const color = id => C[id] ?? C.other;
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const num = (value, digits = 0) => {
  const [integer, fraction] = Number(value).toFixed(digits).split('.');
  const grouped = integer.replace(/\B(?=(\d{3})+(?!\d))/g, '\u202f');
  return fraction === undefined ? grouped : `${grouped},${fraction}`;
};
if (num(.69617, 5) !== '0,69617' || num(1751.2, 1) !== '1\u202f751,2') {
  throw new Error('Invalid number formatting');
}
const charts = [];
let parts;
let chart;

function start(name, title, height, description) {
  chart = { name, title, width: WIDTH, height, observations: [] };
  parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${height}" viewBox="0 0 ${WIDTH} ${height}" role="img" aria-labelledby="title desc">`,
    `<title id="title">${escape(title)}</title><desc id="desc">${escape(description)}</desc>`,
    `<rect width="${WIDTH}" height="${height}" fill="${C.paper}"/>`,
    `<g font-family="Noto Sans, DejaVu Sans, sans-serif" fill="${C.ink}">`,
  ];
}
function text(x, y, value, size = 26, { anchor = 'start', fill = C.ink, weight = 400, halo = false } = {}) {
  parts.push(`<text x="${x}" y="${y}" font-size="${size}" text-anchor="${anchor}" fill="${fill}" font-weight="${weight}"${halo ? ` paint-order="stroke" stroke="${C.paper}" stroke-width="8" stroke-linejoin="round"` : ''}>${escape(value)}</text>`);
}
function line(x1, y1, x2, y2, stroke = C.grid, width = 1.5, dashed = false) {
  parts.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${width}"${dashed ? ' stroke-dasharray="5 7"' : ''}/>`);
}
function rect(x, y, width, height, fill, extra = '') {
  parts.push(`<rect x="${x}" y="${y}" width="${width}" height="${height}" fill="${fill}" ${extra}/>`);
}
function axis({ x, y, width, max, ticks, top, label, digits = 0 }) {
  for (const v of ticks) {
    const px = x + v / max * width;
    line(px, top, px, y);
    text(px, y + 36, num(v, digits), 24, { anchor: 'middle', fill: C.muted });
  }
  if (label) text(x + width / 2, y + 78, label, 25, { anchor: 'middle', fill: C.muted });
}
function observe(record) { chart.observations.push(record); }
function finite(value, label, lo = 0, hi = Infinity) {
  if (!Number.isFinite(value) || value < lo || value > hi) throw new Error(`Invalid ${label}: ${value}`);
  return value;
}
async function save() {
  parts.push('</g></svg>');
  const svg = parts.join('\n');
  const target = resolve(out, chart.name);
  writeFileSync(`${target}.svg`, svg + '\n');
  await sharp(Buffer.from(svg), { density: 144 }).png().toFile(`${target}.png`);
  charts.push(chart);
}

async function quality() {
  start('quality-rrf', 'Изменение NDCG@10 после реранжирования кандидатов RRF', 582,
    'Разница NDCG@10 между моделью и исходным гибридным поиском. Ноль означает качество исходного поиска. Влево - снижение, вправо - рост. Это абсолютная разница, не процентное изменение.');
  const ys = order.map((_, i) => 127 + i * 82);
  order.forEach((id, i) => text(56, ys[i] + 10, labels[id], 28,
    { weight: ['jev', 'laya'].includes(id) ? 600 : 400 }));
  for (const [j, dataset] of datasets.entries()) {
    const x = 409 + j * 394, width = 304;
    const min = -.30, max = .20;
    const scale = value => x + (value - min) / (max - min) * width;
    text(x + width / 2, 55, dataset.label, 30, { anchor: 'middle', weight: 600 });
    for (const tick of [-.2, 0, .2]) {
      const px = scale(tick);
      line(px, 94, px, 499, tick === 0 ? '#bac3c9' : C.grid, tick === 0 ? 2 : 1);
      text(px, 539, tick === 0 ? '0' : `${tick > 0 ? '+' : ''}${num(tick, 1)}`, 24,
        { anchor: 'middle', fill: C.muted });
    }
    const rows = qualityRows('rrf', dataset.id);
    const baseline = finite(rows.baseline, 'baseline', 0, 1);
    order.forEach((id, i) => {
      const value = finite(rows[id], id, 0, 1);
      const delta = value - baseline;
      finite(delta, 'quality delta', min, max);
      const a = scale(0), b = scale(delta);
      rect(Math.min(a, b), ys[i] - 15, Math.abs(b - a), 30, color(id));
      text(b + (delta < 0 ? -12 : 12), ys[i] + 9,
        `${delta > 0 ? '+' : ''}${num(delta, 3)}`, 25,
        { anchor: delta < 0 ? 'end' : 'start', halo: true });
      observe({ dataset: dataset.id, method: 'rrf', model: id, metric: 'ndcg_at_10_delta', value: delta,
        baseline_ndcg10: baseline, model_ndcg10: value });
    });
  }
  await save();
}

async function latency() {
  start('latency', 'Время реранжирования 20 документов', 672,
    'У каждой модели две полосы на общей шкале: сверху бирюзовая медиана P50, снизу песочная P95. Время полного списка, 2542 измерения на модель. У Jev учитывается сеть и API OpenRouter; четыре локальные модели работают на RTX 5090. Первичный поиск не включен.');
  const x = 432, max = 2200, width = 960, ys = [144, 248, 352, 456, 560];
  const series = [
    { field: 'p50_ms', label: 'Медиана (P50)', offset: -17, fill: C.jev, legendX: x },
    { field: 'p95_ms', label: 'P95', offset: 17, fill: C.laya, legendX: x + 345 },
  ];
  text(56, 55, '20 документов · мс', 26, { fill: C.muted });
  for (const item of series) {
    rect(item.legendX, 37, 26, 18, item.fill);
    text(item.legendX + 40, 55, item.label, 26);
  }
  axis({ x, y: 610, width, max, ticks: [0, 500, 1000, 1500, 2000], top: 94 });
  order.forEach((id, i) => {
    const row = latencyRow(id);
    const p50 = finite(row.p50_ms, `${id}/p50`), p95 = finite(row.p95_ms, `${id}/p95`);
    if (p95 < p50 || p95 > max || row.n_lists !== 2542) throw new Error(`Invalid latency ${id}`);
    text(56, ys[i] + 9, labels[id] + (id === 'jev' ? ' · API' : ''), 28,
      { weight: ['jev', 'laya'].includes(id) ? 600 : 400 });
    for (const item of series) {
      const end = x + row[item.field] / max * width;
      const y = ys[i] + item.offset;
      rect(x, y - 12, end - x, 24, item.fill);
      text(end + 12, y + 9, num(row[item.field], 1), 25, { halo: true });
    }
    observe({ model: id, metric: 'list_latency_ms', p50, p95, n: row.n_lists });
  });
  await save();
}

async function throughput() {
  start('throughput', 'Скорость обработки потока', 480,
    'Отдельный замер четырех локальных моделей на другой RTX 5090: три прохода по 480 пар. Jev не измерялся.');
  const x = 432, width = 960, max = 200, ys = [127, 209, 291, 373];
  text(56, 55, 'Пар/с', 26, { fill: C.muted });
  axis({ x, y: 417, width, max, ticks: [0, 50, 100, 150, 200], top: 94 });
  order.filter(id => id !== 'jev').forEach((id, i) => {
    const row = throughputRow(id);
    const value = finite(row.pairs_per_second, id, 0, max);
    if (row.measured_pairs !== 1440) throw new Error('Unexpected throughput sample');
    text(56, ys[i] + 9, labels[id], 27, { weight: id === 'laya' ? 600 : 400 });
    rect(x, ys[i] - 15, value / max * width, 30, color(id));
    text(x + value / max * width + 16, ys[i] + 10, num(value, 1), 29);
    observe({ model: id, metric: 'pairs_per_second', value, measured_pairs: row.measured_pairs });
  });
  await save();
}

// Normalize only names, never values. The snapshot is shared by all plots.
const sourceAliases = {
  jev: ['jev', 'jev-openrouter', 'typesafe/jev-1.13'],
  laya: ['laya', 'laya-multi', 'convaiinnovations/laya-multilingual'],
  bge: ['bge', 'BAAI/bge-reranker-v2-m3'],
  qwen06: ['qwen06', 'qwen-0.6b', 'Qwen/Qwen3-Reranker-0.6B'],
  qwen8: ['qwen8', 'qwen-8b', 'Qwen/Qwen3-Reranker-8B'],
  baseline: ['baseline', 'retrieval_baseline'],
};
function find(rows, id) {
  const found = rows.find(r => sourceAliases[id].includes(r.model));
  if (!found) throw new Error(`Missing ${id}`);
  return found;
}
function qualityRows(method, dataset) {
  const row = data.quality.find(r => r.method === method && r.dataset === dataset);
  if (!row) throw new Error(`Missing quality ${dataset}/${method}`);
  const rows = Object.entries(row.models).map(([model, metrics]) => ({ model, ...metrics }));
  return { baseline: row.baseline_ndcg10, ...Object.fromEntries(order.map(id => [id, find(rows, id).ndcg10])) };
}
function latencyRow(id) { return find(data.latency, id); }
function throughputRow(id) { return find(data.throughput, id); }
function memoryRow(id) { return find(data.memory, id); }

await quality();
await latency();
await throughput();

const tables = ['# Дополнительные результаты', '',
  'Значения из того же снимка данных, что и графики. Округлены только подписи.', '',
  'NDCG@10 учитывает релевантность и положение документов в первой десятке. Значения от 0 до 1, больше - лучше. SciFact: 300 запросов, FiQA: 648, NFCorpus: 323. Внутри каждого варианта поиска реранкеры получают одинаковые 20 кандидатов.', ''];
for (const method of ['rrf', 'bm25']) {
  tables.push(`## NDCG@10 после ${method.toUpperCase()}`, '',
    '| Модель | SciFact | FiQA | NFCorpus |', '|---|---:|---:|---:|');
  for (const id of ['baseline', ...order]) {
    tables.push(`| ${labels[id]} | ${datasets.map(d => num(qualityRows(method, d.id)[id], 5)).join(' | ')} |`);
  }
  tables.push('');
}
tables.push('С кандидатами BM25 и RRF направление изменений совпало: Laya снизила NDCG@10 на всех трех наборах, остальные модели повысили. Величины изменений различаются. Например, на FiQA потеря Laya составила 0,028 после BM25 и 0,085 после RRF.', '');
tables.push('## Пиковая видеопамять', '', '| Модель | Allocated, GiB | Reserved, GiB |', '|---|---:|---:|');
for (const id of order.filter(id => id !== 'jev')) {
  const row = memoryRow(id);
  tables.push(`| ${labels[id]} | ${num(row.allocated_gib, 2)} | ${num(row.reserved_gib, 2)} |`);
}
tables.push('', 'Allocated - память тензоров через CUDA allocator; reserved включает кеш и не является минимальным требованием к VRAM. Пики основного теста на RTX 5090. Laya - FP32 с TF32, BGE - FP16, Qwen - BF16. Память Jev через API не измерялась.', '');
writeFileSync(resolve(dir, '../chart-tables.md'), tables.join('\n'));

const sha256 = buffer => createHash('sha256').update(buffer).digest('hex');
const manifest = {
  schema_version: 1,
  data_file: 'benchmark-charts.data.json', data_sha256: sha256(raw),
  renderer: { name: 'SVG + sharp', sharp: sharp.versions.sharp, font: 'Noto Sans', png_scale: 2 },
  charts: charts.map(c => ({ ...c,
    svg_sha256: sha256(readFileSync(resolve(out, `${c.name}.svg`))),
    png_sha256: sha256(readFileSync(resolve(out, `${c.name}.png`))),
  })),
};
writeFileSync(resolve(dir, 'charts-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');

const thumbWidth = 800;
let y = 0;
const tiles = [];
for (const c of charts) {
  const buffer = await sharp(resolve(out, `${c.name}.png`)).resize({ width: thumbWidth }).png().toBuffer();
  const meta = await sharp(buffer).metadata();
  tiles.push({ input: buffer, left: 0, top: y });
  y += meta.height + 24;
}
await sharp({ create: { width: thumbWidth, height: y - 24, channels: 3, background: '#f2f4f5' } })
  .composite(tiles).png().toFile(resolve(out, 'charts-preview.png'));
console.log(`Rendered ${charts.length} charts, SVG + 2x PNG; saved data/output hashes and ${manifest.charts.reduce((n, c) => n + c.observations.length, 0)} observations.`);
