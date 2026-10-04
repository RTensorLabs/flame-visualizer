<script lang="ts">
  import { app, setFocus } from '../../lib/state.svelte';
  import { diverging, robustScale, sequential, css } from '../../lib/colors';
  import { norm } from '../../lib/derived';
  import { pct, signed } from '../../lib/format';
  import CellGrid from '../../lib/viz/CellGrid.svelte';
  import ArcView from '../../lib/viz/ArcView.svelte';
  import HeadPicker from '../../lib/viz/HeadPicker.svelte';
  import { attnHover } from '../../lib/viz/hovers';

  let allDims = $state(false);
  const tr = $derived(app.trace!);
  const cfg = $derived(app.cfg!);
  const l = $derived(app.block);
  const h = $derived(app.head);
  const hd = $derived(cfg.d_model / cfg.n_heads);
  const labels = $derived(tr.ids.map((id) => app.tok(id)));
  const probs = $derived(tr.L(l, 'probs'));

  const z = $derived(app.zoom && app.zoom.i < tr.T ? app.zoom : { i: tr.T - 1, j: Math.max(0, tr.T - 2) });
  const q = $derived(tr.L(l, 'q').sub(h, z.i));
  const k = $derived(tr.L(l, 'k').sub(h, z.j));
  const prods = $derived(Array.from(q, (v, d) => v * k[d]));
  const sum = $derived(prods.reduce((a, b) => a + b, 0));
  const order = $derived(Array.from(prods.keys()).sort((a, b) => Math.abs(prods[b]) - Math.abs(prods[a])));
  const shown = $derived(allDims ? Array.from(prods.keys()) : order.slice(0, 10));
  const pmax = $derived(Math.max(...prods.map(Math.abs), 1e-6));
  const vs = $derived(robustScale(tr.L(l, 'q').data));
  const row = $derived(Array.from({ length: z.i + 1 }, (_, j) => j).sort((a, b) => probs.at(h, z.i, b) - probs.at(h, z.i, a)));
  const rowShown = $derived(row.slice(0, 10));
  const rest = $derived(row.slice(10).reduce((a, j) => a + probs.at(h, z.i, j), 0));

  const contrib = $derived(Array.from({ length: cfg.n_heads }, (_, hh) => norm(tr.L(l, 'head_contrib').sub(hh, app.focus))));
  const cmax = $derived(Math.max(...contrib, 1e-6));
  const qPre = $derived(tr.L(l, 'q_pre').sub(h, z.i));
</script>

<section class="card pad" aria-labelledby="heads-h">
  <div class="h-row">
    <h3 id="heads-h">Heads in block {l + 1}</h3>
    <span class="sub">pick one to inspect</span>
  </div>
  <HeadPicker layer={l} value={h} onpick={(x) => (app.head = x)} />
</section>

<div class="grid-2">
  <section class="card pad" aria-labelledby="pat-h">
    <div class="h-row">
      <h3 id="pat-h">Attention pattern · head {h + 1}</h3>
      <span class="sub">click a square for the maths</span>
    </div>
    <CellGrid
      label="Attention weights, block {l + 1} head {h + 1}, {tr.T} by {tr.T}"
      rows={tr.T}
      cols={tr.T}
      fit
      cell={30}
      minCell={5}
      rowLabels={labels}
      colLabels={labels}
      activeRow={app.focus}
      pin={z}
      onrowclick={setFocus}
      color={(i, j) => (j > i ? null : sequential(app.palette, app.palette.attn, Math.pow(probs.at(h, i, j), 0.7)))}
      text={(i, j) => (j <= i && probs.at(h, i, j) >= 0.15 ? String(Math.round(probs.at(h, i, j) * 100)) : '')}
      onhover={(i, j) => attnHover(l, h, i, j)}
      onselect={(i, j) => (app.zoom = { i, j })}
    />
  </section>
  <section class="card pad" aria-labelledby="arc2-h">
    <div class="h-row">
      <h3 id="arc2-h">“{app.word(tr.ids[app.focus])}” looks back at</h3>
      <span class="sub">focused token</span>
    </div>
    <ArcView weights={probs.sub(h, app.focus)} query={app.focus} />
    <h4 class="eyebrow mt">What each head wrote into “{app.word(tr.ids[app.focus])}”</h4>
    <ol class="bars">
      {#each contrib as c, hh (hh)}
        <li class:on={hh === h}>
          <button type="button" onclick={() => (app.head = hh)} aria-label="Head {hh + 1}, contribution size {c.toFixed(2)}"
            >H{hh + 1}</button
          >
          <span class="track" aria-hidden="true"><i style:width="{(c / cmax) * 100}%"></i></span>
          <span class="v">{c.toFixed(2)}</span>
        </li>
      {/each}
    </ol>
    <p class="note">Length of each head’s output vector (after W<sub>o</sub>) for this token: how much it changes the residual stream.</p>
  </section>
</div>

<section class="card pad" aria-labelledby="zoom-h">
  <div class="h-row">
    <h3 id="zoom-h">How “{app.word(tr.ids[z.i])}” scores “{app.word(tr.ids[z.j])}”</h3>
    <span class="sub">block {l + 1} · head {h + 1} · {hd}-dimensional query and key</span>
  </div>
  <div class="grid-2">
    <div>
      <h4 class="eyebrow">1 · query × key, dimension by dimension</h4>
      <table class="zt">
        <thead
          ><tr
            ><th scope="col">dim</th><th scope="col">query</th><th scope="col">key</th><th scope="col">product</th><th scope="col"
              ><span class="sr-only">value</span></th
            ></tr
          ></thead
        >
        <tbody>
          {#each shown as d (d)}
            {@const qc = diverging(app.palette, q[d], vs)}
            {@const kc = diverging(app.palette, k[d], vs)}
            <tr>
              <td class="faint">d{d + 1}</td>
              <td><span class="num"><i style:background={css(qc)} aria-hidden="true"></i>{q[d].toFixed(2)}</span></td>
              <td><span class="num"><i style:background={css(kc)} aria-hidden="true"></i>{k[d].toFixed(2)}</span></td>
              <td class="barc"
                ><span class="div" aria-hidden="true"
                  ><i class={prods[d] >= 0 ? 'p' : 'n'} style:width="{(Math.abs(prods[d]) / pmax) * 50}%"></i></span
                ></td
              >
              <td class="v">{signed(prods[d], 3)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
      <button class="btn ghost sm" type="button" onclick={() => (allDims = !allDims)} aria-expanded={allDims}
        >{allDims ? 'Show the 10 largest' : `Show all ${hd} dimensions`}</button
      >
      <p class="sum">
        sum of all {hd} <b>{signed(sum)}</b> <span>× 1/√{hd}</span> = <b>score {signed(sum / Math.sqrt(hd))}</b>
        <span class="faint">(engine: {signed(tr.L(l, 'scores').at(h, z.i, z.j))})</span>
      </p>
    </div>
    <div>
      <h4 class="eyebrow">2 · softmax over everything “{app.word(tr.ids[z.i])}” can see</h4>
      <ol class="sm-list">
        {#each rowShown as j (j)}
          <li class:hit={j === z.j}>
            <button type="button" class="w" onclick={() => (app.zoom = { i: z.i, j })}>{labels[j]}</button>
            <span class="mono faint">{signed(tr.L(l, 'scores').at(h, z.i, j), 2)}</span>
            <span class="track" aria-hidden="true"><i style:width="{probs.at(h, z.i, j) * 100}%"></i></span>
            <span class="v">{pct(probs.at(h, z.i, j))}</span>
          </li>
        {/each}
      </ol>
      {#if rest > 0}<p class="note">+ {row.length - 10} more tokens share {pct(rest)}.</p>{/if}
      <h4 class="eyebrow mt">RoPE · the query before and after rotation</h4>
      <CellGrid
        label="Query before RoPE"
        rows={1}
        cols={hd}
        fit
        cell={14}
        cellH={16}
        gap={1}
        color={(_, d) => diverging(app.palette, qPre[d], vs)}
        onhover={(_, d) => ({
          tag: 'Query before RoPE',
          title: `dim ${d + 1}`,
          body: 'x·W_q, before the position-dependent rotation.',
          calc: signed(qPre[d]),
        })}
      />
      <CellGrid
        label="Query after RoPE"
        rows={1}
        cols={hd}
        fit
        cell={14}
        cellH={16}
        gap={1}
        color={(_, d) => diverging(app.palette, q[d], vs)}
        onhover={(_, d) => ({
          tag: 'Query after RoPE',
          title: `dim ${d + 1}`,
          body: `Pairs (d, d+${hd / 2}) are rotated by an angle that grows with position ${z.i}, so scores depend on relative distance.`,
          calc: signed(q[d]),
        })}
      />
      <p class="note">Rotation keeps the length (‖q‖ = {norm(qPre).toFixed(3)} → {norm(q).toFixed(3)}) and only changes direction.</p>
    </div>
  </div>
</section>

<style>
  .mt {
    margin-top: 18px !important;
    margin-bottom: 8px !important;
  }
  h4.eyebrow {
    margin: 0 0 10px;
  }
  .bars,
  .sm-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .bars li {
    display: grid;
    grid-template-columns: 40px 1fr 44px;
    gap: 10px;
    align-items: center;
  }
  .bars button {
    border: 0;
    background: none;
    font: 600 12.5px var(--mono);
    color: var(--muted);
    text-align: left;
    padding: 2px 0;
  }
  .bars li.on button {
    color: var(--attn);
  }
  .bars .track > i,
  .sm-list .track > i {
    background: var(--attn);
  }
  .v {
    font: 500 12px var(--mono);
    text-align: right;
    color: var(--muted);
  }
  .zt {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  .zt th {
    font-size: 11.5px;
    font-weight: 600;
    color: var(--faint);
    text-align: left;
    padding-bottom: 6px;
  }
  .zt td {
    padding: 2px 4px 2px 0;
    font-family: var(--mono);
  }
  .num {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-width: 64px;
    font: 500 11.5px var(--mono);
  }
  .num i {
    width: 14px;
    height: 14px;
    border-radius: 4px;
    flex: none;
  }
  .barc {
    width: 40%;
  }
  .div {
    position: relative;
    display: block;
    height: 12px;
    background: var(--cell);
    border-radius: 4px;
  }
  .div::before {
    content: '';
    position: absolute;
    left: 50%;
    top: -2px;
    bottom: -2px;
    width: 1px;
    background: var(--faint);
  }
  .div i {
    position: absolute;
    top: 2px;
    bottom: 2px;
    border-radius: 2px;
  }
  .div i.p {
    left: 50%;
    background: var(--pos);
  }
  .div i.n {
    right: 50%;
    background: var(--neg);
  }
  .btn.sm {
    min-height: 30px;
    padding: 5px 10px;
    font-size: 12.5px;
    margin-top: 8px;
  }
  .sum {
    margin: 12px 0 0;
    padding: 10px 12px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    font-size: 13px;
    color: var(--muted);
  }
  .sum b {
    font-family: var(--mono);
    color: var(--text);
  }
  .sm-list li {
    display: grid;
    grid-template-columns: minmax(60px, 110px) 52px 1fr 48px;
    gap: 8px;
    align-items: center;
    padding: 2px 6px;
    border-radius: 6px;
  }
  .sm-list li.hit {
    background: var(--attn-soft);
  }
  .sm-list .w {
    border: 0;
    background: none;
    padding: 0;
    text-align: left;
    font: 500 12.5px var(--mono);
    white-space: pre;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .sm-list .mono {
    font-size: 11px;
  }
</style>
