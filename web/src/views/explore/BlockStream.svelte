<!-- Logit lens: what the residual stream would predict after each block, for every token. -->
<script lang="ts">
  import { app, setFocus } from '../../lib/state.svelte';
  import { css, onColor, sequential } from '../../lib/colors';
  import { norm } from '../../lib/derived';
  import { pct } from '../../lib/format';

  const tr = $derived(app.trace!);
  const cfg = $derived(app.cfg!);
  const layers = $derived(Array.from({ length: cfg.n_layers }, (_, l) => l));
  const ids = $derived(tr.ids);
</script>

<section class="card pad" aria-labelledby="lens-h">
  <div class="h-row">
    <h3 id="lens-h">Logit lens</h3>
    <span class="sub">top guess for the token after each position, decoded after every block</span>
  </div>
  <div class="scroll">
    <table>
      <caption class="sr-only">Top predicted next token and its probability, per input token (rows) and block (columns)</caption>
      <thead>
        <tr
          ><th scope="col">token</th>{#each layers as l (l)}<th scope="col" class:cur={l === app.block}>B{l + 1}</th>{/each}<th scope="col"
            >actual next</th
          ></tr
        >
      </thead>
      <tbody>
        {#each ids as id, i (i)}
          <tr class:on={i === app.focus}>
            <th scope="row"><button type="button" onclick={() => setFocus(i)}>{app.tok(id)}</button></th>
            {#each layers as l (l)}
              {@const g = tr.L(l, 'lens_ids').at(i, 0)}
              {@const p = tr.L(l, 'lens_p').at(i, 0)}
              {@const bg = sequential(app.palette, app.palette.out, Math.min(1, p * 1.1))}
              {@const hit = i + 1 < ids.length && g === ids[i + 1]}
              <td style:background={css(bg)} style:color={onColor(bg)} class:hit title="{app.tok(g)} · {pct(p)}">
                {app.tok(g)}<small>{pct(p)}</small>
              </td>
            {/each}
            <td class="next">{i + 1 < ids.length ? app.tok(ids[i + 1]) : '?'}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  <p class="note">
    Early blocks are unsure; confidence builds as each block adds to the stream. Outlined cells match the token that actually comes next in
    your prompt.
  </p>
</section>

<section class="card pad" aria-labelledby="norm-h">
  <div class="h-row">
    <h3 id="norm-h">Size of “{app.word(ids[app.focus])}” in the residual stream</h3>
    <span class="sub">‖x‖ entering each block, and after the last</span>
  </div>
  <ol class="norms">
    {#each [...layers.map((l) => ({ label: `in B${l + 1}`, v: norm(tr.L(l, 'x_in').sub(app.focus)) })), { label: 'out', v: norm(tr
            .L(cfg.n_layers - 1, 'x_out')
            .sub(app.focus)) }] as r, k (k)}
      <li>
        <span>{r.label}</span><span class="track" aria-hidden="true"
          ><i style:width="{Math.min(100, (r.v / Math.max(...layers.map((l) => norm(tr.L(l, 'x_out').sub(app.focus))), 1e-6)) * 100)}%"
          ></i></span
        ><span class="v">{r.v.toFixed(2)}</span>
      </li>
    {/each}
  </ol>
</section>

<style>
  .scroll {
    overflow-x: auto;
  }
  table {
    border-collapse: separate;
    border-spacing: 3px;
    font-size: 12.5px;
    width: 100%;
  }
  thead th {
    font-size: 11.5px;
    font-weight: 600;
    color: var(--faint);
    padding: 4px;
  }
  thead th.cur {
    color: var(--attn);
  }
  tbody th {
    text-align: right;
    white-space: nowrap;
  }
  tbody th button {
    border: 0;
    background: none;
    font: 500 12.5px var(--mono);
    color: var(--muted);
    white-space: pre;
  }
  tr.on th button {
    color: var(--accent);
    font-weight: 700;
  }
  td {
    border-radius: 6px;
    padding: 5px 4px;
    text-align: center;
    font-family: var(--mono);
    white-space: pre;
    max-width: 110px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  td small {
    display: block;
    font-size: 10px;
    opacity: 0.8;
  }
  td.hit {
    box-shadow: inset 0 0 0 2px var(--text);
  }
  td.next {
    color: var(--muted);
    background: none;
  }
  .norms {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .norms li {
    display: grid;
    grid-template-columns: 60px 1fr 54px;
    gap: 10px;
    align-items: center;
    font-size: 12.5px;
  }
  .norms .track > i {
    background: var(--emb);
  }
  .v {
    font: 500 12px var(--mono);
    text-align: right;
  }
</style>
