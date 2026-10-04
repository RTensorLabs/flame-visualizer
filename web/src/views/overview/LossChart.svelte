<!-- Training curve from the model's metrics.jsonl: train loss (every log step) and validation loss. -->
<script lang="ts">
  import { app } from '../../lib/state.svelte';

  interface Point {
    step: number;
    v: number;
  }
  let train = $state<Point[]>([]);
  let val = $state<Point[]>([]);
  let failed = $state(false);
  let hover = $state<number | null>(null);
  let width = $state(640);

  const base: string = import.meta.env.VITE_MODEL_BASE ?? '/model/';
  $effect(() => {
    fetch(base + 'metrics.jsonl')
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(String(r.status)))))
      .then((t) => {
        const rows = t
          .split('\n')
          .filter(Boolean)
          .map((l) => JSON.parse(l) as Record<string, number>);
        train = rows.filter((r) => 'loss' in r).map((r) => ({ step: r.step, v: r.loss }));
        val = rows.filter((r) => 'val_loss' in r).map((r) => ({ step: r.step, v: r.val_loss }));
      })
      .catch(() => (failed = true));
  });

  // light/dark steps validated with the dataviz palette checker (categorical, 2 slots)
  const cTrain = $derived(app.theme === 'dark' ? '#3b82f6' : '#2563eb');
  const cVal = $derived(app.theme === 'dark' ? '#ea580c' : '#c2410c');

  const H = 220;
  const m = { l: 40, r: 92, t: 12, b: 28 };
  const maxStep = $derived(Math.max(1, ...train.map((p) => p.step)));
  const maxV = $derived(Math.ceil(Math.max(1, ...train.map((p) => p.v), ...val.map((p) => p.v))));
  const x = (s: number) => m.l + (s / maxStep) * (width - m.l - m.r);
  const y = (v: number) => m.t + (1 - v / maxV) * (H - m.t - m.b);
  const line = (ps: Point[]) => ps.map((p, i) => `${i ? 'L' : 'M'}${x(p.step).toFixed(1)},${y(p.v).toFixed(1)}`).join('');
  const yTicks = $derived(Array.from({ length: maxV + 1 }, (_, i) => i).filter((v) => maxV <= 5 || v % 2 === 0));
  const xTicks = $derived([0, 0.25, 0.5, 0.75, 1].map((f) => Math.round((f * maxStep) / 100) * 100));
  const near = (ps: Point[], s: number) => ps.reduce((a, b) => (Math.abs(b.step - s) < Math.abs(a.step - s) ? b : a), ps[0]);
  const hTrain = $derived(hover !== null && train.length ? near(train, hover) : null);
  const hVal = $derived(hover !== null && val.length ? near(val, hover) : null);
  const lastT = $derived(train.at(-1));
  const lastV = $derived(val.at(-1));

  function onmove(ev: MouseEvent) {
    const r = (ev.currentTarget as SVGElement).getBoundingClientRect();
    const s = ((ev.clientX - r.left - m.l) / (width - m.l - m.r)) * maxStep;
    hover = s < 0 || s > maxStep ? null : s;
  }
</script>

<section class="card pad" aria-labelledby="loss-h">
  <div class="h-row">
    <h2 id="loss-h">How it learned</h2>
    <span class="sub">cross-entropy loss over {maxStep.toLocaleString()} training steps</span>
  </div>
  {#if failed}
    <p class="note">Training log unavailable.</p>
  {:else if train.length}
    <ul class="legend" aria-label="Series">
      <li><i style:background={cTrain}></i>Training loss</li>
      <li><i style:background={cVal}></i>Validation loss</li>
    </ul>
    <div class="plot" bind:clientWidth={width}>
      <svg
        viewBox="0 0 {width} {H}"
        {width}
        height={H}
        role="img"
        aria-label="Loss falls from {train[0].v.toFixed(2)} to {lastT?.v.toFixed(2)} (training) and {lastV?.v.toFixed(2)} (validation)."
        onmousemove={onmove}
        onmouseleave={() => (hover = null)}
      >
        {#each yTicks as v (v)}
          <line class="grid" x1={m.l} x2={width - m.r} y1={y(v)} y2={y(v)} />
          <text class="ax" x={m.l - 8} y={y(v) + 4} text-anchor="end">{v}</text>
        {/each}
        {#each xTicks as s (s)}
          <text class="ax" x={x(s)} y={H - 8} text-anchor="middle">{s.toLocaleString()}</text>
        {/each}
        <path d={line(train)} stroke={cTrain} />
        <path d={line(val)} stroke={cVal} />
        {#if lastT}<circle cx={x(lastT.step)} cy={y(lastT.v)} r="4" fill={cTrain} class="ring" /><text
            class="end"
            x={x(lastT.step) + 10}
            y={y(lastT.v) + 14}>train {lastT.v.toFixed(2)}</text
          >{/if}
        {#if lastV}<circle cx={x(lastV.step)} cy={y(lastV.v)} r="4" fill={cVal} class="ring" /><text
            class="end"
            x={x(lastV.step) + 10}
            y={y(lastV.v) - 6}>val {lastV.v.toFixed(2)}</text
          >{/if}
        {#if hover !== null && hTrain}
          <line class="cross" x1={x(hTrain.step)} x2={x(hTrain.step)} y1={m.t} y2={H - m.b} />
          <circle cx={x(hTrain.step)} cy={y(hTrain.v)} r="4" fill={cTrain} class="ring" />
          {#if hVal}<circle cx={x(hVal.step)} cy={y(hVal.v)} r="4" fill={cVal} class="ring" />{/if}
        {/if}
      </svg>
      {#if hover !== null && hTrain}
        <div class="tt" style:left="{Math.min(x(hTrain.step) + 12, width - 170)}px" aria-hidden="true">
          <b>step {hTrain.step.toLocaleString()}</b>
          <span><i style:background={cTrain}></i>train {hTrain.v.toFixed(3)}</span>
          {#if hVal}<span><i style:background={cVal}></i>val {hVal.v.toFixed(3)} <small>(step {hVal.step})</small></span>{/if}
        </div>
      {/if}
    </div>
    <details>
      <summary>Show validation data as a table</summary>
      <table>
        <thead><tr><th scope="col">Step</th><th scope="col">Validation loss</th><th scope="col">Training loss (nearest)</th></tr></thead>
        <tbody>
          {#each val as p (p.step)}
            <tr><td>{p.step.toLocaleString()}</td><td>{p.v.toFixed(3)}</td><td>{near(train, p.step).v.toFixed(3)}</td></tr>
          {/each}
        </tbody>
      </table>
    </details>
  {:else}
    <p class="note">Loading training log…</p>
  {/if}
</section>

<style>
  .plot {
    position: relative;
  }
  svg {
    display: block;
    max-width: 100%;
    overflow: visible;
  }
  path {
    fill: none;
    stroke-width: 2;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
  .grid {
    stroke: var(--border);
    stroke-width: 1;
  }
  .cross {
    stroke: var(--border2);
    stroke-width: 1;
  }
  .ax {
    font: 500 11px var(--mono);
    fill: var(--faint);
  }
  .end {
    font: 600 12px var(--mono);
    fill: var(--text);
  }
  .ring {
    stroke: var(--surface);
    stroke-width: 2;
  }
  .legend {
    display: flex;
    gap: 16px;
    list-style: none;
    margin: 0 0 8px;
    padding: 0;
    font-size: 13px;
    color: var(--muted);
  }
  .legend i,
  .tt i {
    display: inline-block;
    width: 14px;
    height: 3px;
    border-radius: 2px;
    margin-right: 6px;
    vertical-align: 3px;
  }
  .tt {
    position: absolute;
    top: 10px;
    display: flex;
    flex-direction: column;
    gap: 2px;
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 8px;
    padding: 8px 10px;
    font: 500 12px var(--mono);
    box-shadow: var(--shadow);
    pointer-events: none;
    min-width: 150px;
  }
  details {
    margin-top: 10px;
    font-size: 13px;
  }
  summary {
    cursor: pointer;
    color: var(--muted);
  }
  table {
    border-collapse: collapse;
    margin-top: 8px;
    font: 500 12.5px var(--mono);
  }
  th,
  td {
    text-align: right;
    padding: 3px 14px 3px 0;
    border-bottom: 1px solid var(--border);
  }
  th {
    font-family: var(--sans);
    color: var(--muted);
  }
</style>
