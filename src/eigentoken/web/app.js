/* eigentoken web app — vanilla JS, no build step. */
'use strict';

const $ = (id) => document.getElementById(id);
const h = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined && v !== false) e.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return e;
};
const svgEl = (tag, attrs = {}) => {
  const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  return e;
};

const n4 = (x) => (x === null || x === undefined || Number.isNaN(x) ? '—' : x.toFixed(4));
const n3 = (x) => (x === null || x === undefined || Number.isNaN(x) ? '—' : x.toFixed(3));
const n2 = (x) => (x === null || x === undefined || Number.isNaN(x) ? '—' : x.toFixed(2));

/* Probagraph's exit colours, less its blue -- that is the accent, which marks the
   query token -- plus enough to reach twelve. All read on light and dark. */
const PALETTE = ['#ea580c', '#16a34a', '#8b5cf6', '#0d9488', '#e11d48', '#ca8a04',
                 '#db2777', '#0891b2', '#65a30d', '#64748b', '#a16207', '#c026d3'];

const fmtBytes = (n) => {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (n >= 1000 && i < units.length - 1) { n /= 1000; i++; }
  return `${i < 2 ? n.toFixed(0) : n.toFixed(1)} ${units[i]}`;
};

const splitWords = (s) => s.split(/[\s,;]+/).map((w) => w.trim()).filter(Boolean);

/* ------------------------------------------------------------------ api */

/* Where the API lives. '' means this same origin -- the page as `eigentoken serve`
   serves it. A static copy (GitHub Pages) has no API of its own, so it takes one from
   ?api=URL (remembered for next visit), the eigentoken-api meta tag, or the last one
   used, in that order. */
function resolveApiBase() {
  const clean = (u) => (u || '').trim().replace(/\/+$/, '');
  let fromQuery = '';
  try { fromQuery = new URLSearchParams(location.search).get('api') || ''; } catch (_) {}
  if (fromQuery) {
    try { localStorage.setItem('eigentoken.api', clean(fromQuery)); } catch (_) {}
    return clean(fromQuery);
  }
  try {
    const meta = document.querySelector('meta[name="eigentoken-api"]');
    if (meta && meta.content.trim()) return clean(meta.content);
  } catch (_) {}
  try { return clean(localStorage.getItem('eigentoken.api')); } catch (_) { return ''; }
}

const API_BASE = resolveApiBase();

async function api(path, body) {
  const res = await fetch(API_BASE + '/api' + path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
  if (res.status === 202) return { pending: data };
  if (!res.ok) {
    const detail = (data && (data.detail || data.message)) || res.statusText;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return { data };
}

/* Run a request, waiting out any model loads and reporting progress. */
async function run(path, body, outId, render, btn) {
  const out = $(outId);
  if (btn) btn.disabled = true;
  try {
    for (;;) {
      let r;
      try {
        r = await api(path, body);
      } catch (e) {
        out.replaceChildren(h('div', { class: 'msg err' }, e.message));
        return;
      }
      if (r.pending) {
        out.replaceChildren(h('div', { class: 'msg load' },
          h('span', { class: 'spin' }),
          'loading ' + r.pending.waiting.map((w) =>
            `${w.spec.model_id} (${w.spec.matrix}, ${w.note ? w.note + ', ' : ''}${w.elapsed}s)`).join(', ') +
          ' — first use of a model downloads and resides it; later requests are instant'));
        refreshLoaded();
        await new Promise((s) => setTimeout(s, 1200));
        continue;
      }
      out.replaceChildren(...[].concat(render(r.data)));
      refreshLoaded();
      return;
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

/* --------------------------------------------------------- model specs */

let specs = [];
let nextSpecId = 1;

function defaultSpec() {
  return { uid: nextSpecId++, model_id: 'gpt2', matrix: 'input', center: true, whiten: false };
}

function loadSpecs() {
  try {
    const saved = JSON.parse(localStorage.getItem('eigentoken.specs') || 'null');
    if (Array.isArray(saved) && saved.length) {
      specs = saved.map((s) => ({ ...s, uid: nextSpecId++ }));
      return;
    }
  } catch (_) { /* fall through to the default */ }
  specs = [defaultSpec()];
}

function saveSpecs() {
  const plain = specs.map(({ uid, ...rest }) => rest);
  try { localStorage.setItem('eigentoken.specs', JSON.stringify(plain)); } catch (_) {}
}

const specPayload = (s) => ({
  model_id: s.model_id, matrix: s.matrix, center: !!s.center, whiten: !!s.whiten,
});
const specKey = (s) => [s.model_id, s.matrix, s.center ? 'centered' : null,
                        s.whiten ? 'whitened' : null].filter(Boolean).join('|');
const specLabel = (s) => [s.model_id, s.matrix, s.center ? 'centered' : null,
                          s.whiten ? 'whitened' : null].filter(Boolean).join(' · ');

/* Two slots naming the same model must still be distinguishable in output. */
function specDisplay(s, i) {
  const dupes = specs.filter((o) => o.model_id === s.model_id).length;
  return dupes > 1 ? `${specLabel(s)}  [slot ${i + 1}]` : specLabel(s);
}

function renderSpecs() {
  const list = $('specList');
  list.replaceChildren(...specs.map((s, i) => {
    const meta = h('div', { class: 'spec-meta' }, statusText(s));
    const dot = h('span', { class: 'dot ' + (s._status || '') });
    const row = h('div', { class: 'spec' },
      h('div', { class: 'spec-head' },
        dot,
        h('input', {
          value: s.model_id, list: 'suggested', placeholder: 'org/model-name',
          onchange: (e) => { s.model_id = e.target.value.trim(); commit(); },
        }),
        h('button', { class: 'mini', title: 'Duplicate this slot',
          onclick: () => { specs.splice(i + 1, 0, { ...s, uid: nextSpecId++, _status: null }); commit(); } }, '⧉'),
        h('button', { class: 'mini', title: 'Remove',
          onclick: () => { specs.splice(i, 1); if (!specs.length) specs = [defaultSpec()]; commit(); } }, '×'),
      ),
      h('div', { class: 'spec-opts' },
        h('select', { onchange: (e) => { s.matrix = e.target.value; commit(); } },
          h('option', { value: 'input', selected: s.matrix === 'input' }, 'input'),
          h('option', { value: 'output', selected: s.matrix === 'output' }, 'output')),
        h('label', {},
          h('input', { type: 'checkbox', checked: s.center,
            onchange: (e) => { s.center = e.target.checked; commit(); } }), 'center'),
        h('label', {},
          h('input', { type: 'checkbox', checked: s.whiten,
            onchange: (e) => { s.whiten = e.target.checked; commit(); } }), 'whiten'),
      ),
      meta,
    );
    if (s._status === 'error') meta.classList.add('err');
    return row;
  }));
  fillSpecSelect();
}

function statusText(s) {
  if (s._status === 'ready') {
    return `${s._vocab} × ${s._dim}${s._nlayers ? ` · ${s._nlayers} layers` : ''}`;
  }
  if (s._status === 'loading') return `loading… ${s._note ? s._note + ' · ' : ''}${s._elapsed || 0}s`;
  if (s._status === 'error') return s._error || 'load failed';
  return 'not loaded';
}

function commit() { saveSpecs(); renderSpecs(); }

function fillOneSpecSelect(id) {
  const sel = $(id);
  const prev = sel.value;
  sel.replaceChildren(...specs.map((s, i) =>
    h('option', { value: String(s.uid) }, specDisplay(s, i))));
  /* Keyed by uid, not array index -- so removing an earlier slot can't
     silently repoint the selection at a different model. */
  sel.value = specs.some((s) => String(s.uid) === prev) ? prev : String(specs[0]?.uid ?? '');
}

function fillSpecSelect() {
  fillOneSpecSelect('cloudSpec');
  fillOneSpecSelect('tkSpec');
}

function specForSelect(id) {
  const uid = Number($(id).value);
  return specs.find((sp) => sp.uid === uid) || specs[0];
}

/* Poll resident-model state so the sidebar reflects background loads. */
async function refreshLoaded() {
  let data;
  try { ({ data } = await api('/models')); } catch (_) { return; }
  const byKey = Object.fromEntries((data.loaded || []).map((e) => [e.key, e]));
  let changed = false;
  for (const s of specs) {
    const e = byKey[specKey(s)];
    const st = e ? e.status : null;
    if (s._status !== st) changed = true;
    s._status = st;
    if (e) {
      s._dim = e.dim; s._vocab = e.vocab_size; s._nlayers = e.n_layers;
      s._error = e.error; s._elapsed = e.elapsed; s._note = e.note;
    }
  }
  if (changed) renderSpecs(); else refreshSpecMeta();

  const mem = data.memory || {};
  $('memUse').textContent = mem.used
    ? fmtBytes(mem.used) + (mem.max ? ` of ${fmtBytes(mem.max)}` : '') : '';
  $('loadedList').replaceChildren(...(data.loaded || []).length
    ? data.loaded.map((e) => h('div', { class: 'loaded-row' },
        h('span', { class: 'dot ' + e.status }),
        h('span', { title: e.key + (e.bytes ? ' · ' + fmtBytes(e.bytes) : '') }, e.key),
        e.bytes ? h('em', {}, fmtBytes(e.bytes)) : null,
        e.status === 'error'
          ? h('button', { title: 'Retry this load',
              onclick: async () => {
                await api('/models/load', { ...e.spec, retry: true });
                refreshLoaded();
              } }, '↻')
          : null,
        data.can_unload
          ? h('button', { title: 'Free this matrix',
              onclick: async () => { await api('/models/unload', { key: e.key }); refreshLoaded(); } }, '×')
          : null))
    : [h('div', { class: 'hint' }, 'nothing resident yet')]);
}

function refreshSpecMeta() {
  const rows = $('specList').children;
  specs.forEach((s, i) => {
    const row = rows[i];
    if (!row) return;
    row.querySelector('.dot').className = 'dot ' + (s._status || '');
    const meta = row.querySelector('.spec-meta');
    meta.textContent = statusText(s);
    meta.classList.toggle('err', s._status === 'error');
  });
}

/* ------------------------------------------------------- shared renders */

function tiedNote(meta, spec) {
  if (!meta || !meta.tied || spec.matrix !== 'output') return null;
  return h('div', { class: 'msg warn' },
    'This model ties its embeddings: the output matrix is the same tensor as the '
    + 'input table, so these numbers are identical to the input run by construction.');
}

function warnRow(w) {
  return w.warning ? h('div', { class: 'msg warn' }, w.warning) : null;
}

function wordTable(words) {
  return h('div', { class: 'scroll' }, h('table', {},
    h('thead', {}, h('tr', {},
      ['word', 'single token', 'surface', 'pieces', 'ids', 'norm'].map((t) => h('th', {}, t)))),
    h('tbody', {}, words.map((w) => h('tr', {},
      h('td', {}, w.query),
      h('td', {}, w.single_token
        ? h('span', { class: 'pill good' }, 'yes')
        : h('span', { class: 'pill warn' }, `${w.token_ids.length} pieces`)),
      h('td', {}, JSON.stringify(w.surface)),
      h('td', {}, h('div', { class: 'tokchips' },
        w.tokens.map((t) => h('span', { class: 'tokchip' }, t)))),
      h('td', {}, w.token_ids.join(',')),
      h('td', {}, n3(w.norm)),
    ))),
  ));
}

const heatColor = (v) => {
  /* v in [-1, 1] → red for negative, blue for positive, mid-grey at zero. */
  const t = Math.max(-1, Math.min(1, v));
  const a = Math.abs(t) * 0.62;
  return t >= 0
    ? `color-mix(in srgb, var(--accent) ${(a * 100).toFixed(0)}%, transparent)`
    : `color-mix(in srgb, var(--bad) ${(a * 100).toFixed(0)}%, transparent)`;
};

function cosineMatrix(words, M) {
  return h('div', { class: 'scroll' }, h('table', { class: 'heat' },
    h('thead', {}, h('tr', {}, h('th', {}), words.map((w) => h('th', {}, w)))),
    h('tbody', {}, M.map((row, i) => h('tr', {},
      h('td', {}, words[i]),
      row.map((v, j) => h('td', {
        class: 'cell',
        style: i === j ? '' : `background:${heatColor(v)}`,
      }, n3(v))),
    ))),
  ));
}

function bar(value, max = 1) {
  const pct = Math.min(100, Math.abs(value) / Math.max(max, 1e-9) * 100);
  return h('div', { class: 'barcell' },
    h('span', {}, n4(value)),
    h('span', { class: 'track' },
      h('span', { class: 'fill' + (value < 0 ? ' neg' : ''), style: `width:${pct}%;left:0` })));
}

/* ------------------------------------------------------------- compare */

function renderCompare(d) {
  return d.results.map((r) => {
    const kids = [
      h('h3', {}, r.spec.label, h('span', { class: 'sub' }, `${r.vocab_size} × ${r.dim}`),
        h('span', { class: 'sub' }, r.meta.source)),
      tiedNote(r.meta, r.spec),
      ...r.words.map(warnRow).filter(Boolean),
      h('h4', {}, 'resolution'), wordTable(r.words),
      h('h4', {}, 'pairwise'),
      h('div', { class: 'scroll' }, h('table', {},
        h('thead', {}, h('tr', {},
          ['pair', 'cosine', 'angle°', 'euclid', 'unit euclid', '|a|', '|b|']
            .map((t) => h('th', {}, t)))),
        h('tbody', {}, r.pairs.map((p) => h('tr', {},
          h('td', {}, `${p.a} ~ ${p.b}`),
          h('td', {}, bar(p.cosine)),
          h('td', {}, p.angle_deg.toFixed(1)),
          h('td', {}, n3(p.euclidean)),
          h('td', {}, n3(p.unit_euclidean)),
          h('td', {}, n2(p.norm_a)),
          h('td', {}, n2(p.norm_b)),
        ))))),
    ];
    if (r.words.length > 2) {
      kids.push(h('h4', {}, 'cosine matrix'),
        cosineMatrix(r.words.map((w) => w.query), r.cosine_matrix));
    }
    for (const p of r.pairs) {
      if (!p.attribution) continue;
      const a = p.attribution;
      kids.push(
        h('h4', {}, `dimensions: ${p.a} vs ${p.b}`),
        h('div', { class: 'hint' },
          `the difference is spread over ~${a.effective_dims.toFixed(0)} of ${a.total_dims} `
          + `dimensions; the top 10 carry ${(a.top10_delta_mass * 100).toFixed(1)}% of it`),
        h('div', { class: 'scroll' }, h('table', {},
          h('thead', {}, h('tr', {},
            h('th', {}, 'dim'), h('th', {}, p.a), h('th', {}, p.b),
            h('th', {}, 'share of cosine'),
            h('th', {}, 'dim'), h('th', {}, p.a), h('th', {}, p.b),
            h('th', {}, 'delta'), h('th', {}, 'share of dist²'))),
          h('tbody', {}, a.aligned.map((al, i) => {
            const dv = a.divergent[i] || {};
            return h('tr', {},
              h('td', {}, al.index), h('td', {}, n3(al.a)), h('td', {}, n3(al.b)),
              h('td', {}, n4(al.cosine_share)),
              h('td', {}, dv.index ?? ''), h('td', {}, dv.a !== undefined ? n3(dv.a) : ''),
              h('td', {}, dv.b !== undefined ? n3(dv.b) : ''),
              h('td', {}, dv.delta !== undefined ? n3(dv.delta) : ''),
              h('td', {}, dv.delta_share !== undefined ? n4(dv.delta_share) : ''));
          })))),
      );
    }
    return h('div', { class: 'card' }, ...kids);
  });
}

/* ----------------------------------------------------------- neighbours */

function renderNeighbors(d) {
  return d.results.map((r) => h('div', { class: 'card' },
    h('h3', {}, r.spec.label, h('span', { class: 'sub' }, d.metric)),
    tiedNote(r.meta, r.spec),
    ...r.words.map(({ word, neighbors }) => [
      warnRow(word),
      h('h4', {}, `${word.query} → ${word.tokens.join(' ')} (${word.token_ids.join(',')})`),
      h('div', { class: 'scroll' }, h('table', {},
        h('thead', {}, h('tr', {}, ['#', 'token', 'id', d.metric].map((t) => h('th', {}, t)))),
        h('tbody', {}, neighbors.map((nb) => h('tr', {},
          h('td', {}, nb.rank), h('td', {}, nb.token), h('td', {}, nb.token_id),
          h('td', {}, bar(nb.score, Math.max(...neighbors.map((x) => Math.abs(x.score)))))))))),
    ]).flat().filter(Boolean),
  ));
}

/* -------------------------------------------------------------- analogy */

function renderAnalogy(d) {
  return d.results.map((r) => {
    const kids = [
      h('h3', {}, r.spec.label,
        h('span', { class: 'sub' }, `${d.a} : ${d.b} :: ${d.c} : ?`)),
      tiedNote(r.meta, r.spec),
      ...r.words.map(warnRow).filter(Boolean),
      h('div', { class: 'scroll' }, h('table', {},
        h('thead', {}, h('tr', {}, ['#', 'token', 'id', 'cosine'].map((t) => h('th', {}, t)))),
        h('tbody', {}, r.candidates.map((c) => h('tr', {},
          h('td', {}, c.rank), h('td', {}, c.token), h('td', {}, c.token_id),
          h('td', {}, bar(c.score))))))),
    ];
    if (r.expected) {
      const e = r.expected;
      const ok = e.rank === 1;
      kids.push(
        h('h4', {}, `against the expected answer "${e.word}"`),
        h('div', { class: 'scroll' }, h('table', {},
          h('thead', {}, h('tr', {},
            ['measure', 'value', 'reading'].map((t) => h('th', {}, t)))),
          h('tbody', {},
            h('tr', {},
              h('td', {}, 'offset parallelism'), h('td', {}, n4(e.offset_parallelism)),
              h('td', { class: 'note' },
                `cos(${d.b}−${d.a}, ${e.word}−${d.c}); 1 = identical offset`)),
            h('tr', {},
              h('td', {}, 'cosine to b−a+c'), h('td', {}, n4(e.cosine_to_target)),
              h('td', { class: 'note' }, 'how close the arithmetic lands')),
            h('tr', {},
              h('td', {}, 'rank of expected'),
              h('td', {}, h('span', { class: 'pill ' + (ok ? 'good' : 'warn') },
                e.rank === null ? 'not found' : `#${e.rank}`)),
              h('td', { class: 'note' }, '1 = the analogy resolves correctly'))))),
      );
      if (e.offset_parallelism < 0.2) {
        kids.push(h('div', { class: 'msg warn' },
          'Offset parallelism is low: this contrast is not a consistent direction '
          + 'in this model, so a good top-1 here would be luck.'));
      }
    }
    return h('div', { class: 'card' }, ...kids);
  });
}

/* ----------------------------------------------------------------- axis */

function renderAxis(d) {
  return d.results.map((r) => {
    const kids = [
      h('h3', {}, r.spec.label,
        h('span', { class: 'sub' },
          r.usable ? 'axis looks usable' : 'axis is weak — read with care')),
      tiedNote(r.meta, r.spec),
      h('h4', {}, 'defining pairs'),
      h('div', { class: 'scroll' }, h('table', {},
        h('thead', {}, h('tr', {},
          ['pair', 'offset norm', 'cos to axis'].map((t) => h('th', {}, t)))),
        h('tbody', {}, r.pairs.map((p) => h('tr', {},
          h('td', {}, p.label), h('td', {}, n3(p.norm)),
          h('td', {}, bar(p.cos_to_axis))))))),
      h('h4', {}, 'is this actually one direction?'),
      h('div', { class: 'scroll' }, h('table', {},
        h('thead', {}, h('tr', {},
          ['diagnostic', 'value', 'reading'].map((t) => h('th', {}, t)))),
        h('tbody', {},
          h('tr', {}, h('td', {}, 'offset coherence'), h('td', {}, n4(r.coherence)),
            h('td', { class: 'note' }, 'mean cosine between the defining offsets')),
          h('tr', {}, h('td', {}, 'PC1 energy share'), h('td', {}, n4(r.pc1_ratio)),
            h('td', { class: 'note' },
              `1.0 = perfectly rank-one; ${n2(r.chance_pc1)} = chance with this many pairs`)),
          h('tr', {}, h('td', {}, 'mean / PC1 alignment'), h('td', {}, n4(r.pc1_alignment)),
            h('td', { class: 'note' }, '1 = the mean is the main axis')),
          h('tr', {}, h('td', {}, 'verdict'),
            h('td', {}, h('span', { class: 'pill ' + (r.usable ? 'good' : 'bad') },
              r.usable ? 'usable' : 'weak')), h('td', { class: 'note' }, ''))))),
    ];
    if (!r.usable) {
      kids.push(h('div', { class: 'msg warn' },
        'The defining offsets disagree with each other, so this contrast is not a '
        + 'single linear direction in this model. Anything below is reported but '
        + 'should not be read as measuring the concept.'));
    }
    if (r.probes) {
      const mx = Math.max(...r.probes.map((p) => Math.abs(p.projection)), 0.05);
      kids.push(
        h('h4', {}, `probe words  (− toward "${r.low_label}", + toward "${r.high_label}")`),
        h('div', { class: 'scroll' }, h('table', {},
          h('thead', {}, h('tr', {}, ['word', 'projection'].map((t) => h('th', {}, t)))),
          h('tbody', {}, r.probes.map((p) => h('tr', {},
            h('td', {}, p.word), h('td', {}, bar(p.projection, mx))))))),
      );
    }
    if (r.extremes) {
      kids.push(h('h4', {}, 'vocabulary extremes'),
        h('div', { class: 'scroll' }, h('table', {},
          h('thead', {}, h('tr', {},
            h('th', {}, '#'), h('th', {}, `toward ${r.high_label}`), h('th', {}, 'score'),
            h('th', {}, `toward ${r.low_label}`), h('th', {}, 'score'))),
          h('tbody', {}, r.extremes.high.map((hi, i) => h('tr', {},
            h('td', {}, i + 1), h('td', {}, hi.token), h('td', {}, n3(hi.score)),
            h('td', {}, r.extremes.low[i].token), h('td', {}, n3(r.extremes.low[i].score))))))));
    }
    return h('div', { class: 'card' }, ...kids);
  });
}

/* --------------------------------------------------------------- bias */

function renderBias(d) {
  return d.results.map((r) => {
    const mx = Math.max(...r.word_scores.map((w) => Math.abs(w.score)), 0.05);
    return h('div', { class: 'card' },
      h('h3', {}, r.spec.label,
        h('span', { class: 'sub' },
          `d=${n3(r.effect_size)} (${r.magnitude})  ·  p=${n4(r.p_value)}`)),
      tiedNote(r.meta, r.spec),
      h('h4', {}, 'is the association significant?'),
      h('div', { class: 'scroll' }, h('table', {},
        h('thead', {}, h('tr', {},
          ['metric', 'value', 'reading'].map((t) => h('th', {}, t)))),
        h('tbody', {},
          h('tr', {}, h('td', {}, "effect size (Cohen's d)"), h('td', {}, n3(r.effect_size)),
            h('td', {},
              h('span', { class: 'pill ' + (Math.abs(r.effect_size) >= 0.5 ? 'good' : 'bad') },
                r.magnitude))),
          h('tr', {}, h('td', {}, 'p-value'), h('td', {}, n4(r.p_value)),
            h('td', { class: 'note' },
              `Monte Carlo estimate, ${r.n_permutations} permutations`)),
          h('tr', {}, h('td', {}, 'statistic'), h('td', {}, n4(r.statistic)),
            h('td', { class: 'note' }, 'raw s(X,Y,A,B)'))))),
      h('h4', {}, `per-word association  (+ toward "${r.a_label}", − toward "${r.b_label}")`),
      h('div', { class: 'scroll' }, h('table', {},
        h('thead', {}, h('tr', {},
          ['word', 'group', 's(word, A, B)'].map((t) => h('th', {}, t)))),
        h('tbody', {}, r.word_scores.map((w) => h('tr', {},
          h('td', {}, w.word), h('td', {}, w.group === 'x' ? r.x_label : r.y_label),
          h('td', {}, bar(w.score, mx))))))));
  });
}

/* ----------------------------------------------------------- crossmodel */

function renderCross(d) {
  const cards = [h('div', { class: 'card' },
    h('h3', {}, 'models', h('span', { class: 'sub' }, `${d.words.length} words compared`)),
    d.identical_matrices ? h('div', { class: 'msg warn' },
      'Two of the selected slots resolve to the same underlying tensor (this model '
      + 'ties its input and output embeddings), so they agree perfectly by '
      + 'construction rather than by finding anything.') : null,
    d.dropped.length ? h('div', { class: 'msg warn' },
      `dropped ${d.dropped.length} word(s) that are multi-token in at least one `
      + `model: ${d.dropped.join(', ')}`) : null,
    h('div', { class: 'scroll' }, h('table', {},
      h('thead', {}, h('tr', {},
        ['model', 'dim', 'vocab', 'multi-token words'].map((t) => h('th', {}, t)))),
      h('tbody', {}, d.models.map((m) => h('tr', {},
        h('td', {}, m.spec.label, h('span', { class: 'sub' }, ' ' + m.meta.source)),
        h('td', {}, m.dim),
        h('td', {}, m.vocab_size), h('td', {}, m.multi_token)))))),
  )];

  for (const c of d.comparisons) {
    const kids = [
      h('h3', {}, `${c.a.label}  vs  ${c.b.label}`),
      h('div', { class: 'scroll' }, h('table', {},
        h('thead', {}, h('tr', {}, ['measure', 'value', 'reading'].map((t) => h('th', {}, t)))),
        h('tbody', {},
          h('tr', {}, h('td', {}, 'RSA spearman'), h('td', {}, bar(c.rsa_spearman)),
            h('td', { class: 'note' }, 'rank agreement of all pairwise similarities')),
          h('tr', {}, h('td', {}, 'RSA pearson'), h('td', {}, bar(c.rsa_pearson)),
            h('td', { class: 'note' }, 'linear agreement')),
          h('tr', {}, h('td', {}, 'procrustes disparity'), h('td', {}, n4(c.procrustes_disparity)),
            h('td', { class: 'note' },
              `residual after the best rotation in ${c.procrustes_dim}d; 0 = same shape`
              + (c.underdetermined
                 ? ` — only ${d.words.length} words for a ${c.procrustes_dim}d rotation, so this`
                   + ' reflects the pairwise-distance mismatch alone. Add words for a stronger test.'
                 : '')))))),
      h('h4', {}, 'words the two models disagree about most'),
      h('div', { class: 'scroll' }, h('table', {},
        h('thead', {}, h('tr', {}, ['word', 'profile agreement'].map((t) => h('th', {}, t)))),
        h('tbody', {}, c.divergent_words.map((w) => h('tr', {},
          h('td', {}, w.word), h('td', {}, bar(w.agreement))))))),
    ];
    if (c.overlap) {
      const rows = c.overlap.per_word.slice().sort((x, y) => x.jaccard - y.jaccard);
      kids.push(
        h('h4', {}, `nearest-neighbour agreement (top ${c.overlap.k}) — `
          + `mean jaccard ${n3(c.overlap.mean_jaccard)}`),
        h('div', { class: 'scroll' }, h('table', {},
          h('thead', {}, h('tr', {},
            h('th', {}, 'word'), h('th', {}, 'jaccard'),
            h('th', {}, c.a.model_id), h('th', {}, c.b.model_id))),
          h('tbody', {}, rows.map((w) => h('tr', {},
            h('td', {}, w.word), h('td', {}, n2(w.jaccard)),
            h('td', { class: 'note' }, w.a.slice(0, 8).join(' ')),
            h('td', { class: 'note' }, w.b.slice(0, 8).join(' '))))))),
      );
    }
    cards.push(h('div', { class: 'card' }, ...kids));
  }
  return cards;
}

/* --------------------------------------------------------------- tokens */

function renderTokens(d) {
  return h('div', { class: 'card' },
    h('h3', {}, d.spec.label, h('span', { class: 'sub' }, d.meta.source)),
    wordTable(d.words),
    h('h4', {}, 'variants tried'),
    h('div', { class: 'scroll' }, h('table', {},
      h('thead', {}, h('tr', {}, ['word', 'forms (form → piece count)'].map((t) => h('th', {}, t)))),
      h('tbody', {}, d.words.map((w) => h('tr', {},
        h('td', {}, w.query),
        h('td', { class: 'note' }, h('div', { class: 'tokchips' },
          w.variants.map((v) => h('span', {
            class: 'tokchip',
            style: v.n_tokens === 1 ? 'color:var(--good);border-color:var(--good)' : '',
          }, `${JSON.stringify(v.form)} → ${v.n_tokens}`))))))))),
  );
}

/* ---------------------------------------------------------------- cloud */

const cloud = {
  data: null,
  view: { cx: 0, cy: 0, scale: 1 },
  hidden: new Set(),
  trail: [],
};

function cloudSize() {
  const svg = $('cloudSvg');
  const r = svg.getBoundingClientRect();
  return { w: r.width || 800, h: r.height || 620 };
}

function fitView() {
  const { w, h: hh } = cloudSize();
  cloud.view = { cx: 0, cy: 0, scale: Math.min(w, hh) / 2 * 0.82 };
}

const toScreen = (x, y) => {
  const { w, h: hh } = cloudSize();
  return [(x - cloud.view.cx) * cloud.view.scale + w / 2,
          (y - cloud.view.cy) * cloud.view.scale + hh / 2];
};

function drawCloud() {
  const svg = $('cloudSvg');
  const d = cloud.data;
  svg.replaceChildren();
  if (!d) return;
  const { w, h: hh } = cloudSize();
  svg.setAttribute('viewBox', `0 0 ${w} ${hh}`);

  const pts = d.points.filter((p) => !cloud.hidden.has(p.cluster));
  const sims = pts.map((p) => p.cosine);
  const lo = Math.min(...sims, 0), hi = Math.max(...sims, 1e-6);
  const norm = (s) => (hi === lo ? 1 : (s - lo) / (hi - lo));

  /* Rings: only meaningful when radius encodes true cosine distance. */
  if (d.mode === 'radial') {
    const maxDist = 1 - lo;
    const gring = svgEl('g');
    for (const t of [0.25, 0.5, 0.75, 1.0]) {
      const [cx, cy] = toScreen(0, 0);
      gring.append(svgEl('circle', {
        class: 'ring', cx, cy, r: t * cloud.view.scale,
      }));
      gring.append(Object.assign(svgEl('text', {
        class: 'ring-label', x: cx + 4, y: cy - t * cloud.view.scale - 3,
      }), { textContent: `cos ${(1 - t * maxDist).toFixed(2)}` }));
    }
    svg.append(gring);
  }

  /* Nodes, most similar first so labels go to the closest tokens. */
  const order = pts.slice().sort((a, b) => b.cosine - a.cosine);
  const placed = [];
  const gnodes = svgEl('g');

  for (const p of order) {
    const [x, y] = toScreen(p.x, p.y);
    if (x < -60 || y < -30 || x > w + 60 || y > hh + 30) continue;
    const t = norm(p.cosine);
    const rad = 2.5 + 4.5 * t;
    const fs = 9 + 4 * t;
    const colour = PALETTE[p.cluster % PALETTE.length];

    const g = svgEl('g', {
      class: 'node',
      onclick: () => recentre(p),
      onmouseenter: (ev) => showTip(ev, p),
      onmousemove: (ev) => showTip(ev, p),
      onmouseleave: hideTip,
    });
    g.append(svgEl('circle', { cx: x, cy: y, r: rad, fill: colour, 'fill-opacity': 0.85 }));

    /* Greedy label placement: skip anything that would overlap. Zooming in
       spreads the points out, so more labels appear as you explore. */
    const label = p.token;
    const wpx = label.length * fs * 0.58 + 6;
    const box = [x + rad + 3, y - fs / 2, x + rad + 3 + wpx, y + fs / 2];
    if (!placed.some((b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) {
      placed.push(box);
      const txt = svgEl('text', {
        x: x + rad + 3, y, 'font-size': fs, fill: 'var(--fg)',
        'fill-opacity': 0.55 + 0.45 * t,
      });
      txt.textContent = label;
      g.append(txt);
    }
    gnodes.append(g);
  }
  svg.append(gnodes);

  /* The query itself, always drawn on top. */
  const [qx, qy] = toScreen(0, 0);
  const cg = svgEl('g', { class: 'node center-node' });
  cg.append(svgEl('circle', { cx: qx, cy: qy, r: 6 }));
  const ct = svgEl('text', { x: qx + 10, y: qy, 'font-size': 13 });
  ct.textContent = d.center.tokens.join('') || d.center.query;
  cg.append(ct);
  svg.append(cg);
}

function showTip(ev, p) {
  const tip = $('tip');
  const stage = $('cloudSvg').getBoundingClientRect();
  tip.classList.remove('hidden');
  tip.replaceChildren(
    h('div', {}, h('b', {}, p.token), ` id ${p.token_id}`),
    h('div', {}, `cosine ${n4(p.cosine)} · norm ${n2(p.norm)} · group ${p.cluster + 1}`),
    h('div', { style: 'color:var(--fg-faint)' }, 'click to recentre'),
  );
  const x = ev.clientX - stage.left + 14;
  const y = ev.clientY - stage.top + 14;
  tip.style.left = Math.min(x, stage.width - tip.offsetWidth - 8) + 'px';
  tip.style.top = Math.min(y, stage.height - tip.offsetHeight - 8) + 'px';
}
const hideTip = () => $('tip').classList.add('hidden');

function renderCloudSide() {
  const d = cloud.data;
  if (!d) return;
  const groups = [...new Set(d.points.map((p) => p.cluster))].sort((a, b) => a - b);
  $('cloudLegend').replaceChildren(
    h('div', { class: 'hint', style: 'margin:0 0 6px' },
      d.mode === 'radial'
        ? 'radius = true cosine distance · angle = projection'
        : `PCA scatter · PC1 ${(d.explained_variance_ratio[0] * 100).toFixed(0)}%`
        + `, PC2 ${(d.explained_variance_ratio[1] * 100).toFixed(0)}%`),
    ...(groups.length > 1 ? groups.map((c) => {
      const members = d.points.filter((p) => p.cluster === c);
      const top = members.slice().sort((a, b) => b.cosine - a.cosine).slice(0, 3)
        .map((p) => p.token).join(' ');
      return h('div', {
        class: 'legend-row' + (cloud.hidden.has(c) ? ' muted' : ''),
        title: 'click to show or hide this group',
        onclick: () => {
          cloud.hidden.has(c) ? cloud.hidden.delete(c) : cloud.hidden.add(c);
          drawCloud(); renderCloudSide();
        },
      },
        h('span', { class: 'swatch', style: `background:${PALETTE[c % PALETTE.length]}` }),
        h('span', {}, `${members.length}`),
        h('span', { style: 'flex:1;overflow:hidden;text-overflow:ellipsis' }, top));
    }) : []),
  );

  $('cloudList').replaceChildren(...d.points.map((p) => h('div', {
    class: 'lst-row', onclick: () => recentre(p),
  },
    h('span', { class: 'swatch', style: `background:${PALETTE[p.cluster % PALETTE.length]}` }),
    h('span', { class: 'w' }, p.token),
    h('span', { class: 's' }, n3(p.cosine)))));
}

function renderCrumbs() {
  const c = $('crumbs');
  if (!cloud.trail.length) { c.replaceChildren(); return; }
  const kids = [];
  cloud.trail.forEach((step, i) => {
    if (i) kids.push(h('span', { class: 'sep' }, '›'));
    kids.push(h('button', {
      title: step.query,
      onclick: () => {
        cloud.trail = cloud.trail.slice(0, i);
        $('cloudWord').value = step.query;
        runCloud();
      },
    }, step.label));
  });
  c.replaceChildren(...kids);
}

/* Recentre on the exact row that was clicked.
 *
 * Not on its text: "water" and "␣water" are different tokens, and re-resolving
 * the text would silently jump to whichever one the resolver prefers. The id
 * names the row unambiguously. */
function recentre(point) {
  const d = cloud.data;
  if (d) {
    cloud.trail.push({
      label: d.center.tokens.join('') || d.center.query,
      query: $('cloudWord').value,
    });
  }
  $('cloudWord').value = 'id:' + point.token_id;
  runCloud();
}

async function runCloud() {
  const s = specForSelect('cloudSpec');
  if (!s) return;
  const body = {
    spec: specPayload(s),
    word: $('cloudWord').value.trim(),
    k: Number($('cloudK').value),
    clusters: Number($('cloudC').value),
    mode: $('cloudMode').value,
    pooling: $('pooling').value,
    exclude_variants: $('cloudExVar').checked,
  };
  await run('/cloud', body, 'out-cloud', (d) => {
    cloud.data = d;
    cloud.hidden = new Set();
    fitView();
    drawCloud();
    renderCloudSide();
    renderCrumbs();
    const msgs = [];
    if (d.center.warning) msgs.push(h('div', { class: 'msg warn' }, d.center.warning));
    msgs.push(h('div', { class: 'msg' },
      `${d.points.length} neighbours of ${d.center.tokens.join('')} `
      + `(id ${d.center.token_ids.join(',')}) in ${d.spec.label}`));
    const tn = tiedNote(d.meta, d.spec);
    if (tn) msgs.push(tn);
    return msgs;
  }, $('runCloud'));
}

function initCloudInteraction() {
  const svg = $('cloudSvg');
  let drag = null;

  svg.addEventListener('pointerdown', (e) => {
    drag = { x: e.clientX, y: e.clientY, cx: cloud.view.cx, cy: cloud.view.cy };
    svg.classList.add('dragging');
    svg.setPointerCapture(e.pointerId);
  });
  svg.addEventListener('pointermove', (e) => {
    if (!drag) return;
    cloud.view.cx = drag.cx - (e.clientX - drag.x) / cloud.view.scale;
    cloud.view.cy = drag.cy - (e.clientY - drag.y) / cloud.view.scale;
    drawCloud();
  });
  const endDrag = (e) => {
    drag = null;
    svg.classList.remove('dragging');
    if (e.pointerId !== undefined && svg.hasPointerCapture?.(e.pointerId)) {
      svg.releasePointerCapture(e.pointerId);
    }
  };
  svg.addEventListener('pointerup', endDrag);
  svg.addEventListener('pointercancel', endDrag);

  svg.addEventListener('wheel', (e) => {
    e.preventDefault();
    const r = svg.getBoundingClientRect();
    const { w, h: hh } = cloudSize();
    /* Keep the point under the cursor fixed while zooming. */
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const dx = (mx - w / 2) / cloud.view.scale + cloud.view.cx;
    const dy = (my - hh / 2) / cloud.view.scale + cloud.view.cy;
    const factor = Math.exp(-e.deltaY * 0.0015);
    cloud.view.scale = Math.max(20, Math.min(40000, cloud.view.scale * factor));
    cloud.view.cx = dx - (mx - w / 2) / cloud.view.scale;
    cloud.view.cy = dy - (my - hh / 2) / cloud.view.scale;
    drawCloud();
  }, { passive: false });

  const zoomBy = (f) => { cloud.view.scale *= f; drawCloud(); };
  $('zoomIn').onclick = () => zoomBy(1.35);
  $('zoomOut').onclick = () => zoomBy(1 / 1.35);
  $('zoomReset').onclick = () => { fitView(); drawCloud(); };

  let raf = null;
  window.addEventListener('resize', () => {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(drawCloud);
  });
}

/* ----------------------------------------------------------------- wire */

function bindSlider(id, labelId) {
  const s = $(id), l = $(labelId);
  const upd = () => { l.textContent = s.value; };
  s.addEventListener('input', upd);
  upd();
}

function initTabs() {
  $('tabs').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-tab]');
    if (!b) return;
    for (const el of $('tabs').children) el.classList.toggle('active', el === b);
    for (const el of document.querySelectorAll('.tab')) {
      el.classList.toggle('active', el.id === 'tab-' + b.dataset.tab);
    }
    if (b.dataset.tab === 'cloud' && cloud.data) { fitView(); drawCloud(); }
  });
}

async function initHealth() {
  try {
    const { data } = await api('/health');
    $('health').replaceChildren(
      h('b', {}, 'ready'),
      ` · ${data.cuda ? 'cuda' : 'cpu'} · up to ${data.max_models} resident`
      + (data.restricted ? ' · restricted model list' : ''));
  } catch (e) {
    $('health').textContent = 'server unreachable';
    showOffline();
  }
}

/* No API answered: say what this page needs, with the exact command for this origin. */
function showOffline() {
  let origin = '';
  try { origin = location.origin; } catch (_) {}
  $('offlineAt').textContent = API_BASE ? ' at ' + API_BASE : '';
  /* A file:// page has no origin to allow; any other page needs its own. */
  $('offlineCmd').textContent = !origin || origin === 'null'
    ? 'eigentoken serve --preload gpt2:center'
    : `eigentoken serve --cors-origin ${origin} --preload gpt2:center`;
  $('apiInput').value = API_BASE || 'http://localhost:8077';
  $('apiForm').onsubmit = (e) => {
    e.preventDefault();
    const url = $('apiInput').value.trim().replace(/\/+$/, '');
    try { localStorage.setItem('eigentoken.api', url); } catch (_) {}
    location.search = url ? '?api=' + encodeURIComponent(url) : '';
  };
  $('offline').hidden = false;
}

async function initModels() {
  try {
    const { data } = await api('/models');
    $('suggested').replaceChildren(...data.suggested.map((m) =>
      h('option', { value: m.id }, m.note)));
  } catch (_) { /* the datalist is a convenience, not a requirement */ }
}

let biasPresets = [];

function applyBiasPreset() {
  const p = biasPresets.find((x) => x.key === $('biasPreset').value);
  if (!p) return;
  $('biasXLabel').value = p.x_label; $('biasYLabel').value = p.y_label;
  $('biasALabel').value = p.a_label; $('biasBLabel').value = p.b_label;
  $('biasX').value = p.x.join(' '); $('biasY').value = p.y.join(' ');
  $('biasA').value = p.a.join(' '); $('biasB').value = p.b.join(' ');
}

async function initBiasPresets() {
  try {
    const { data } = await api('/bias/presets');
    biasPresets = data.presets;
    $('biasPreset').replaceChildren(...biasPresets.map((p) =>
      h('option', { value: p.key }, `${p.x_label} vs ${p.y_label}  /  ${p.a_label} vs ${p.b_label}`)));
    applyBiasPreset();
  } catch (_) { /* the preset dropdown is a convenience, not a requirement */ }
}

function main() {
  loadSpecs();
  renderSpecs();
  initTabs();
  initHealth();
  initModels();
  initBiasPresets();
  initCloudInteraction();
  refreshLoaded();
  setInterval(refreshLoaded, 4000);

  for (const [s, l] of [['cloudK', 'cloudKLbl'], ['cloudC', 'cloudCLbl'],
                        ['cmpDims', 'cmpDimsLbl'], ['nbK', 'nbKLbl'],
                        ['anK', 'anKLbl'], ['axEx', 'axExLbl'],
                        ['xmNb', 'xmNbLbl'], ['biasPerm', 'biasPermLbl']]) bindSlider(s, l);

  $('biasPreset').onchange = applyBiasPreset;

  $('addModel').onclick = () => {
    specs.push({ uid: nextSpecId++, model_id: 'gpt2', matrix: 'input', center: true, whiten: false });
    commit();
  };

  $('runCloud').onclick = () => { cloud.trail = []; runCloud(); };
  $('cloudWord').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { cloud.trail = []; runCloud(); }
  });
  $('cloudMode').onchange = () => { if (cloud.data) runCloud(); };

  $('runCompare').onclick = (e) => run('/compare', {
    specs: specs.map(specPayload),
    words: splitWords($('cmpWords').value),
    dims: Number($('cmpDims').value),
    pooling: $('pooling').value,
  }, 'out-compare', renderCompare, e.target);

  $('runNeighbors').onclick = (e) => run('/neighbors', {
    specs: specs.map(specPayload),
    words: splitWords($('nbWords').value),
    k: Number($('nbK').value),
    metric: $('nbMetric').value,
    pooling: $('pooling').value,
    exclude_variants: $('nbExVar').checked,
  }, 'out-neighbors', renderNeighbors, e.target);

  $('runAnalogy').onclick = (e) => run('/analogy', {
    specs: specs.map(specPayload),
    a: $('anA').value, b: $('anB').value, c: $('anC').value,
    expect: $('anE').value, k: Number($('anK').value),
    pooling: $('pooling').value,
  }, 'out-analogy', renderAnalogy, e.target);

  $('runAxis').onclick = (e) => run('/axis', {
    specs: specs.map(specPayload),
    pairs: $('axPairs').value.split('\n').map((l) => l.trim()).filter(Boolean),
    probes: splitWords($('axProbes').value),
    extremes: Number($('axEx').value),
    pooling: $('pooling').value,
  }, 'out-axis', renderAxis, e.target);

  $('runBias').onclick = (e) => run('/bias', {
    specs: specs.map(specPayload),
    x: splitWords($('biasX').value), y: splitWords($('biasY').value),
    a: splitWords($('biasA').value), b: splitWords($('biasB').value),
    x_label: $('biasXLabel').value.trim(), y_label: $('biasYLabel').value.trim(),
    a_label: $('biasALabel').value.trim(), b_label: $('biasBLabel').value.trim(),
    permutations: Number($('biasPerm').value),
    pooling: $('pooling').value,
  }, 'out-bias', renderBias, e.target);

  $('runCross').onclick = (e) => run('/crossmodel', {
    specs: specs.map(specPayload),
    words: splitWords($('xmWords').value),
    neighbors: Number($('xmNb').value),
    single_token_only: $('xmSingle').checked,
    pooling: $('pooling').value,
  }, 'out-crossmodel', renderCross, e.target);

  $('runTokens').onclick = (e) => run('/tokens', {
    spec: specPayload(specForSelect('tkSpec')),
    words: splitWords($('tkWords').value),
    pooling: $('pooling').value,
  }, 'out-tokens', renderTokens, e.target);

  runCloud();
}

document.addEventListener('DOMContentLoaded', main);
