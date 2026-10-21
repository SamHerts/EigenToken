/* Render-path tests for the web UI, run in Node against real API responses.
 *
 * app.js is written for a browser, so this supplies a DOM small enough to
 * build and serialise a tree.  It catches the failures that matter in a
 * template-free UI -- a bad field name, an unbalanced call, a null deref --
 * without needing a headless browser.
 *
 * Usage: node tests/render.js [base-url]   (default: the server's own port 8077)
 */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const BASE = process.argv[2] || 'http://127.0.0.1:8077';
const APP = path.join(__dirname, '..', 'src', 'eigentoken', 'web', 'app.js');

/* ------------------------------------------------------------ tiny DOM */

const VOID = new Set(['input', 'br', 'hr', 'img']);

class Node {
  constructor(tag, ns) {
    this.tagName = tag;
    this.ns = ns || null;
    this.attrs = {};
    this.childNodes = [];
    this.listeners = {};
    this._text = null;
    this.nodeType = 1;
    this.style = {};
    const self = this;
    this.classList = {
      add: (c) => { self.attrs.class = ((self.attrs.class || '') + ' ' + c).trim(); },
      remove: (c) => {
        self.attrs.class = (self.attrs.class || '').split(/\s+/).filter((x) => x && x !== c).join(' ');
      },
      toggle: (c, on) => (on ? self.classList.add(c) : self.classList.remove(c)),
      contains: (c) => (self.attrs.class || '').split(/\s+/).includes(c),
    };
  }
  set className(v) { this.attrs.class = v; }
  get className() { return this.attrs.class || ''; }
  set innerHTML(v) { this._text = v; }
  set textContent(v) { this._text = v; this.childNodes = []; }
  get textContent() {
    if (this._text !== null) return this._text;
    return this.childNodes.map((c) => c.textContent).join('');
  }
  get children() { return this.childNodes.filter((c) => c.nodeType === 1); }
  setAttribute(k, v) { this.attrs[k] = v; }
  getAttribute(k) { return this.attrs[k]; }
  addEventListener(k, fn) { (this.listeners[k] ||= []).push(fn); }
  removeEventListener() {}
  append(...kids) { for (const k of kids) if (k != null) this.childNodes.push(k); }
  appendChild(k) { this.childNodes.push(k); return k; }
  replaceChildren(...kids) { this.childNodes = kids.filter((k) => k != null); this._text = null; }
  querySelector(sel) {
    const want = sel.replace('.', '');
    for (const c of this.children) {
      if ((c.attrs.class || '').split(/\s+/).includes(want)) return c;
      const hit = c.querySelector(sel);
      if (hit) return hit;
    }
    return null;
  }
  getBoundingClientRect() { return { width: 900, height: 620, left: 0, top: 0 }; }
  setPointerCapture() {} releasePointerCapture() {} hasPointerCapture() { return false; }
  get offsetWidth() { return 100; } get offsetHeight() { return 40; }
}

class TextNode {
  constructor(t) { this.nodeType = 3; this._t = String(t); }
  get textContent() { return this._t; }
}

function serialize(n, depth = 0) {
  if (n.nodeType === 3) return n.textContent;
  const a = Object.entries(n.attrs).map(([k, v]) => ` ${k}="${v}"`).join('');
  const inner = n._text !== null ? n._text : n.childNodes.map((c) => serialize(c, depth + 1)).join('');
  if (VOID.has(n.tagName)) return `<${n.tagName}${a}>`;
  return `<${n.tagName}${a}>${inner}</${n.tagName}>`;
}

const stub = () => new Node('div');
const elements = {};
const document = {
  createElement: (t) => new Node(t),
  createElementNS: (ns, t) => new Node(t, ns),
  createTextNode: (t) => new TextNode(t),
  getElementById: (id) => (elements[id] ||= new Node('div')),
  querySelectorAll: () => [],
  addEventListener: () => {},
};

const sandbox = {
  document,
  console,
  localStorage: {
    _d: {},
    getItem(k) { return this._d[k] ?? null; },
    setItem(k, v) { this._d[k] = v; },
  },
  window: { addEventListener: () => {} },
  fetch: async () => { throw new Error('network disabled in render tests'); },
  setTimeout, setInterval: () => 0, clearInterval: () => {},
  requestAnimationFrame: () => 0, cancelAnimationFrame: () => {},
  Math, JSON, Date, Object, Array, String, Number, Set, Map, Error, isNaN,
};
sandbox.globalThis = sandbox;
sandbox.window.document = document;

const ctx = vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(APP, 'utf8'), ctx, { filename: 'app.js' });

const get = (name) => vm.runInContext(name, ctx);

/* ---------------------------------------------------------------- tests */

const failures = [];
function check(name, cond, detail = '') {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
  if (!cond) failures.push(name);
}

async function post(pathname, body) {
  for (let i = 0; i < 40; i++) {
    const res = await fetch(BASE + '/api' + pathname, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (res.status === 202) { await new Promise((s) => setTimeout(s, 1500)); continue; }
    if (!res.ok) throw new Error(`${pathname}: ${JSON.stringify(data)}`);
    return data;
  }
  throw new Error(`${pathname}: model never became ready`);
}

/* Render and assert the output contains the strings a human would look for. */
function renders(name, fn, data, mustContain) {
  let html;
  try {
    const out = fn(data);
    html = [].concat(out).map((n) => serialize(n)).join('');
  } catch (e) {
    check(name, false, e.message);
    return '';
  }
  const missing = mustContain.filter((s) => !html.includes(s));
  check(name, missing.length === 0,
    missing.length ? `missing ${JSON.stringify(missing)}` : `${html.length} chars`);
  return html;
}

const GPT2 = { model_id: 'gpt2', matrix: 'input', center: true, whiten: false };
const GPT2OUT = { model_id: 'gpt2', matrix: 'output', center: true, whiten: false };
const PYTHIA = { model_id: 'EleutherAI/pythia-160m', matrix: 'input', center: true, whiten: false };

(async () => {
  console.log(`app.js loaded against ${BASE}\n`);

  console.log('helpers');
  check('h() builds an element tree',
    serialize(get('h')('p', { class: 'x' }, 'hi')) === '<p class="x">hi</p>');
  check('splitWords handles mixed separators',
    JSON.stringify(get('splitWords')('a, b\nc  d;e')) === '["a","b","c","d","e"]');
  check('specLabel names a duplicated slot distinctly',
    get('specLabel')(GPT2) !== get('specLabel')(GPT2OUT),
    `${get('specLabel')(GPT2)} vs ${get('specLabel')(GPT2OUT)}`);

  console.log('\ncompare');
  const cmp = await post('/compare', {
    specs: [GPT2, GPT2OUT], words: ['king', 'queen', 'man', 'woman'], dims: 4,
  });
  renders('renderCompare', get('renderCompare'), cmp,
    ['king ~ queen', 'cosine matrix', 'dimensions: king vs queen', 'ties its embeddings']);

  console.log('\nneighbours');
  const nb = await post('/neighbors', { specs: [GPT2], words: ['king'], k: 8 });
  /* U+2423 marks the leading space; 'queen' and ' queen' are different rows. */
  renders('renderNeighbors', get('renderNeighbors'), nb, ['\u2423queen', 'cosine']);

  const nbFiltered = await post('/neighbors', {
    specs: [GPT2], words: ['king'], k: 8, exclude_variants: true,
  });
  const nbToks = nbFiltered.results[0].words[0].neighbors
    .map((n) => n.token.replace('\u2423', '').toLowerCase());
  check('exclude_variants drops king itself from its own neighbours',
    !nbToks.includes('king'), nbToks.join(' '));

  console.log('\nanalogy');
  const an = await post('/analogy', {
    specs: [GPT2], a: 'man', b: 'king', c: 'woman', expect: 'queen', k: 5,
  });
  renders('renderAnalogy', get('renderAnalogy'), an,
    ['offset parallelism', 'rank of expected', '#1']);

  console.log('\naxis');
  const ax = await post('/axis', {
    specs: [GPT2],
    pairs: ['man:woman', 'king:queen', 'father:mother', 'actor:actress'],
    probes: ['doctor', 'nurse'], extremes: 5,
  });
  renders('renderAxis', get('renderAxis'), ax,
    ['offset coherence', 'PC1 energy share', 'vocabulary extremes', 'usable']);

  console.log('\nbias');
  const bi = await post('/bias', { specs: [GPT2], preset: 'flowers_insects' });
  renders('renderBias', get('renderBias'), bi,
    ["effect size (Cohen's d)", 'p-value', 'per-word association', 'statistic']);

  console.log('\ncross-model');
  const xm = await post('/crossmodel', {
    specs: [GPT2, PYTHIA],
    words: ['king', 'queen', 'man', 'woman', 'paris', 'tokyo', 'dog', 'cat', 'red', 'blue'],
    neighbors: 10, single_token_only: true,
  });
  renders('renderCross', get('renderCross'), xm,
    ['RSA spearman', 'procrustes disparity', 'disagree about most', 'jaccard']);

  const same = await post('/crossmodel', {
    specs: [GPT2, GPT2OUT],
    words: ['king', 'queen', 'man', 'woman', 'paris', 'tokyo'],
    neighbors: 0, single_token_only: true,
  });
  renders('renderCross flags identical matrices', get('renderCross'), same,
    ['same underlying tensor']);

  console.log('\ntokens');
  const tk = await post('/tokens', {
    spec: GPT2, words: ['king', 'antidisestablishmentarianism'],
  });
  renders('renderTokens', get('renderTokens'), tk, ['pieces', 'variants tried']);

  console.log('\ncloud');
  const cl = await post('/cloud', {
    spec: GPT2, word: 'king', k: 80, clusters: 5, mode: 'radial',
  });
  const cloud = get('cloud');
  cloud.data = cl;
  cloud.hidden = new Set();
  get('fitView')();
  get('drawCloud')();
  const svg = elements.cloudSvg;
  const html = serialize(svg);
  const nodes = svg.childNodes.flatMap((g) => g.childNodes || []).length;
  check('cloud draws a node per point',
    html.split('class="node"').length - 1 >= cl.points.length * 0.9,
    `${html.split('class="node"').length - 1} of ${cl.points.length} in view`);
  check('cloud draws distance rings', html.includes('class="ring"') && html.includes('cos '));
  check('cloud labels the query at the centre',
    html.includes('center-node') && html.includes(cl.center.query));
  check('labels are placed without overlapping',
    html.includes('<text'), `${html.split('<text').length - 1} labels drawn`);

  get('renderCloudSide')();
  const legend = serialize(elements.cloudLegend);
  const list = serialize(elements.cloudList);
  check('legend summarises each group',
    legend.includes('legend-row') && legend.includes('true cosine distance'));
  check('side list has a row per neighbour',
    list.split('lst-row').length - 1 === cl.points.length,
    `${list.split('lst-row').length - 1} rows`);

  /* radius must track cosine distance, or the layout lies */
  const rad = cl.points.map((p) => [Math.hypot(p.x, p.y), 1 - p.cosine]);
  const bad = rad.filter(([r, d], i) => {
    const [r2, d2] = rad[(i + 1) % rad.length];
    return (r - r2) * (d - d2) < -1e-9;
  });
  check('radius is monotone in cosine distance', bad.length === 0, `${bad.length} inversions`);

  const pcaCloud = await post('/cloud', {
    spec: GPT2, word: 'king', k: 40, clusters: 0, mode: 'pca',
  });
  cloud.data = pcaCloud;
  get('fitView')(); get('drawCloud')(); get('renderCloudSide')();
  check('pca mode renders without rings',
    !serialize(elements.cloudSvg).includes('class="ring"'));
  check('pca legend reports explained variance',
    serialize(elements.cloudLegend).includes('PC1'));

  console.log();
  if (failures.length) {
    console.log(`${failures.length} FAILED: ${failures.join(', ')}`);
    process.exit(1);
  }
  console.log('all render checks passed');
})();
