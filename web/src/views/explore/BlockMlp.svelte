<script lang="ts">
  import { app } from '../../lib/state.svelte';
  import { diverging, robustScale } from '../../lib/colors';
  import { topAbs } from '../../lib/derived';
  import { signed } from '../../lib/format';
  import CellGrid from '../../lib/viz/CellGrid.svelte';
  import { vecHover } from '../../lib/viz/hovers';

  const tr = $derived(app.trace!);
  const cfg = $derived(app.cfg!);
  const l = $derived(app.block);
  const fw = $derived(app.word(tr.ids[app.focus]));
  const COLS = 64;
  const rowsFor = (n: number) => Math.ceil(n / COLS);

  const stages = $derived([
    { k: 'mlp_norm', t: 'Comes in', code: 'RMSNorm(x)', s: `${cfg.d_model} numbers` },
    { k: 'gate', t: 'Gate', code: 'silu(x·W_gate)', s: 'how much to let through', op: `↓ widen ${cfg.d_model} → ${cfg.mlp_hidden}` },
    { k: 'up', t: 'Content', code: 'x·W_up', s: 'what to carry' },
    { k: 'hidden', t: 'Hidden', code: 'gate ⊙ up', s: `${cfg.mlp_hidden} neurons`, op: '↓ multiply element by element' },
    { k: 'mlp_out', t: 'Down', code: 'h·W_down', s: `back to ${cfg.d_model}`, op: `↓ squeeze ${cfg.mlp_hidden} → ${cfg.d_model}` },
    {
      k: 'x_out',
      t: 'Goes out',
      code: 'x + down',
      s: l + 1 < cfg.n_layers ? `input to block ${l + 2}` : 'input to the output head',
      op: '↓ add to the residual stream',
    },
  ]);
  const hid = $derived(tr.L(l, 'hidden').sub(app.focus));
  const top = $derived(topAbs(hid, 8));
</script>

<section class="card pad" aria-labelledby="mlp-h">
  <div class="h-row">
    <h3 id="mlp-h">Following “{fw}” through the MLP</h3>
    <span class="sub">click a token in the prompt to switch</span>
  </div>
  {#each stages as s (s.k)}
    {@const v = tr.L(l, s.k).sub(app.focus)}
    {@const sc = robustScale(tr.L(l, s.k).data)}
    {#if s.op}<p class="op">{s.op}</p>{/if}
    <div class="prow">
      <div class="lab"><b>{s.t}</b><code>{s.code}</code><small>{s.s}</small></div>
      <CellGrid
        label="{s.t} ({s.code}) for token {fw}, {v.length} values"
        rows={rowsFor(v.length)}
        cols={COLS}
        fit
        cell={11}
        minCell={4}
        gap={1}
        radius={2}
        color={(i, j) => (i * COLS + j < v.length ? diverging(app.palette, v[i * COLS + j], sc) : null)}
        onhover={(i, j) => (i * COLS + j < v.length ? vecHover(s.k, l, app.focus, i * COLS + j) : null)}
      />
    </div>
  {/each}
</section>

<section class="card pad" aria-labelledby="topn-h">
  <div class="h-row">
    <h3 id="topn-h">Strongest neurons for “{fw}”</h3>
    <span class="sub">block {l + 1}</span>
  </div>
  <table>
    <thead
      ><tr><th scope="col">neuron</th><th scope="col">gate</th><th scope="col">× content</th><th scope="col">= activation</th></tr></thead
    >
    <tbody>
      {#each top as n (n)}
        <tr
          ><td>#{n + 1}</td><td>{signed(tr.L(l, 'gate').at(app.focus, n))}</td><td>{signed(tr.L(l, 'up').at(app.focus, n))}</td><td
            ><b>{signed(hid[n])}</b></td
          ></tr
        >
      {/each}
    </tbody>
  </table>
</section>

<style>
  .prow {
    display: grid;
    grid-template-columns: 180px minmax(0, 1fr);
    gap: 16px;
    padding: 10px 0;
    align-items: start;
  }
  .lab b {
    display: block;
    font-size: 13.5px;
  }
  .lab code {
    font-size: 11.5px;
    color: var(--muted);
  }
  .lab small {
    display: block;
    color: var(--faint);
    font-size: 12px;
  }
  .op {
    margin: 0;
    padding-left: 196px;
    font-size: 12.5px;
    color: var(--faint);
  }
  table {
    border-collapse: collapse;
    font: 500 13px var(--mono);
  }
  th {
    font: 600 12px var(--sans);
    color: var(--faint);
    text-align: right;
    padding: 0 18px 6px 0;
  }
  td {
    text-align: right;
    padding: 4px 18px 4px 0;
    border-top: 1px solid var(--border);
  }
  @media (max-width: 900px) {
    .prow {
      grid-template-columns: 1fr;
    }
    .op {
      padding-left: 0;
    }
  }
</style>
