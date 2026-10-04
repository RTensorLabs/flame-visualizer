<script lang="ts">
  import { app, setFocus, type BlockTab } from '../../lib/state.svelte';
  import { diverging, robustScale } from '../../lib/colors';
  import { cosine } from '../../lib/derived';
  import { signed } from '../../lib/format';
  import CellGrid from '../../lib/viz/CellGrid.svelte';
  import Inspector from '../../lib/components/Inspector.svelte';
  import Candidates from '../../lib/components/Candidates.svelte';
  import { vecHover } from '../../lib/viz/hovers';
  import Rail from './Rail.svelte';
  import BlockAttention from './BlockAttention.svelte';
  import BlockMlp from './BlockMlp.svelte';
  import BlockStream from './BlockStream.svelte';

  const tr = $derived(app.trace!);
  const cfg = $derived(app.cfg!);
  const labels = $derived(tr.ids.map((id) => app.tok(id)));
  const emb = $derived(tr.t('embed'));
  const embScale = $derived(robustScale(emb.data));
  const fin = $derived(tr.t('final_norm'));
  const finScale = $derived(robustScale(fin.data));
  const tabs: [BlockTab, string][] = [
    ['attention', 'Attention'],
    ['mlp', 'Feed-forward'],
    ['stream', 'Residual stream'],
  ];
  const pipe = $derived([
    ['RMSNorm', 'a'],
    [`Attention ×${cfg.n_heads}`, 'a'],
    ['⊕', 'a'],
    ['RMSNorm', 'f'],
    ['SwiGLU', 'f'],
    ['⊕', 'f'],
  ]);
  const lit = (t: string, g: string) =>
    (app.tab === 'attention' && g === 'a') || (app.tab === 'mlp' && g === 'f') || (app.tab === 'stream' && t === '⊕');
</script>

<div class="explore">
  <Rail />

  <div class="stack main fade-in">
    {#if app.stage === 'tokens'}
      <section class="card pad" aria-labelledby="st-h">
        <span class="tag t-out">Stage 0 · Tokenizer</span>
        <h2 id="st-h">Text is cut into tokens</h2>
        <p class="muted">
          A byte-level BPE tokenizer with {cfg.vocab_size.toLocaleString()} entries: ids 0–255 are raw bytes, 256 marks the end of a story, and
          258 onwards are merges learned from TinyStories. Common words become one token; rare words split into pieces.
        </p>
      </section>
      <section class="card pad" aria-labelledby="tk-h">
        <div class="h-row">
          <h3 id="tk-h">Your prompt as token ids</h3>
          <span class="sub">{tr.T} tokens · context limit {cfg.max_seq_len}</span>
        </div>
        <table class="tok-t">
          <thead><tr><th scope="col">#</th><th scope="col">token</th><th scope="col">id</th><th scope="col">bytes</th></tr></thead>
          <tbody>
            {#each tr.ids as id, i (i)}
              <tr class:on={i === app.focus}>
                <td class="faint">{i}</td>
                <td><button type="button" onclick={() => setFocus(i)} aria-pressed={i === app.focus}>{labels[i]}</button></td>
                <td>{id}</td>
                <td class="faint">{app.info!.tokenHex[id]}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </section>
    {:else if app.stage === 'embedding'}
      <section class="card pad" aria-labelledby="se-h">
        <span class="tag t-emb">Stage 1 · Embedding</span>
        <h2 id="se-h">Each id becomes a vector of {cfg.d_model} numbers</h2>
        <p class="muted">
          Row <span class="mono">id</span> of the {cfg.vocab_size.toLocaleString()} × {cfg.d_model} embedding table. This vector starts the residual
          stream that every block reads from and writes back to. The same table is reused, transposed, as the output head.
        </p>
      </section>
      <section class="card pad" aria-labelledby="ev-h">
        <div class="h-row">
          <h3 id="ev-h">Embedding vectors</h3>
          <span class="sub">all {cfg.d_model} dimensions</span>
        </div>
        <CellGrid
          label="Embedding vectors, {tr.T} tokens by {cfg.d_model} dimensions"
          rows={tr.T}
          cols={cfg.d_model}
          fit
          cell={6}
          cellH={16}
          minCell={2}
          gap={0}
          radius={0}
          rowLabels={labels}
          activeRow={app.focus}
          onrowclick={setFocus}
          color={(i, j) => diverging(app.palette, emb.at(i, j), embScale)}
          onhover={(i, j) => vecHover('embed', 0, i, j)}
        />
      </section>
      <section class="card pad" aria-labelledby="es-h">
        <div class="h-row">
          <h3 id="es-h">Similarity between tokens</h3>
          <span class="sub">cosine</span>
        </div>
        <CellGrid
          label="Cosine similarity of token embeddings"
          rows={tr.T}
          cols={tr.T}
          fit
          cell={28}
          minCell={6}
          rowLabels={labels}
          colLabels={labels}
          activeRow={app.focus}
          onrowclick={setFocus}
          color={(i, j) => diverging(app.palette, cosine(emb.sub(i), emb.sub(j)), 0.6)}
          onhover={(i, j) => ({
            tag: 'Similarity',
            title: `“${labels[i]}” vs “${labels[j]}”`,
            body: 'Cosine similarity of the two embedding vectors.',
            calc: `cos = ${signed(cosine(emb.sub(i), emb.sub(j)))}`,
          })}
        />
      </section>
    {:else if app.stage === 'output'}
      <section class="card pad" aria-labelledby="so-h">
        <span class="tag t-out">Stage 3 · Output</span>
        <h2 id="so-h">Choosing the next token</h2>
        <p class="muted">
          The last token’s vector after block {cfg.n_layers} goes through a final RMSNorm, is scored against all {cfg.vocab_size.toLocaleString()}
          embedding rows (tied LM head), divided by the temperature and sampled.
        </p>
        <p class="pipe" aria-label="Pipeline">
          <span>RMSNorm</span>→<span>LM head {cfg.d_model}→{cfg.vocab_size}</span>→<span>÷ T</span>→<span>softmax</span>→<span
            >top-k / top-p</span
          >→<span>sample</span>
        </p>
      </section>
      <div class="grid-2">
        <section class="card pad" aria-labelledby="cd-h">
          <div class="h-row">
            <h3 id="cd-h">Candidates</h3>
            <span class="sub">real distribution over {cfg.vocab_size.toLocaleString()} tokens</span>
          </div>
          <Candidates limit={12} />
        </section>
        <section class="card pad" aria-labelledby="fv-h">
          <div class="h-row">
            <h3 id="fv-h">Final vector of “{app.word(tr.ids[tr.T - 1])}”</h3>
            <span class="sub">after the final RMSNorm</span>
          </div>
          <CellGrid
            label="Final normalised vector, {cfg.d_model} values"
            rows={4}
            cols={64}
            fit
            cell={10}
            minCell={4}
            gap={1}
            radius={2}
            color={(i, j) => diverging(app.palette, fin.at(tr.T - 1, i * 64 + j), finScale)}
            onhover={(i, j) => vecHover('final', 0, tr.T - 1, i * 64 + j)}
          />
          <p class="note">Each candidate’s logit is the dot product of this vector with that token’s embedding row.</p>
        </section>
      </div>
    {:else}
      <section class="card pad" aria-labelledby="sb-h">
        <div class="bh">
          <div>
            <span class="tag t-attn">Block {app.block + 1} of {cfg.n_layers}</span>
            <h2 id="sb-h">
              {app.tab === 'attention' ? 'Self-attention' : app.tab === 'mlp' ? 'SwiGLU feed-forward' : 'What the stream knows so far'}
            </h2>
            <p class="muted">
              {#if app.tab === 'attention'}{cfg.n_heads} heads run in parallel. Each compares every token’s query with earlier tokens’ keys, then
                blends their values.
              {:else if app.tab === 'mlp'}Applied to each token on its own: widen to {cfg.mlp_hidden}, gate, squeeze back to {cfg.d_model},
                add to the stream.
              {:else}Decode the residual stream after every block to watch the prediction take shape (logit lens).{/if}
            </p>
          </div>
          <div class="ctl">
            <div class="seg" role="group" aria-label="Block">
              {#each Array.from({ length: cfg.n_layers }, (_, k) => k) as k (k)}
                <button type="button" aria-pressed={k === app.block} aria-label="Block {k + 1}" onclick={() => (app.block = k)}
                  >{k + 1}</button
                >
              {/each}
            </div>
            <div class="seg" role="group" aria-label="View">
              {#each tabs as [k, t] (k)}
                <button type="button" aria-pressed={app.tab === k} onclick={() => (app.tab = k)}>{t}</button>
              {/each}
            </div>
          </div>
        </div>
        <p class="pipe" aria-label="Block pipeline">
          {#each pipe as [t, g], k (k)}{#if k}<span class="arr" aria-hidden="true">→</span>{/if}<span
              class:a={lit(t, g) && g === 'a'}
              class:f={lit(t, g) && g === 'f'}
              class:on={lit(t, g)}>{t}</span
            >{/each}
        </p>
      </section>
      {#if app.tab === 'attention'}<BlockAttention />{:else if app.tab === 'mlp'}<BlockMlp />{:else}<BlockStream />{/if}
    {/if}
  </div>

  <aside class="side" aria-label="Details">
    <Inspector />
    <section class="card pad nextc" aria-labelledby="nx-h">
      <div class="h-row">
        <h2 id="nx-h">Next token</h2>
        <button class="link" type="button" onclick={() => (app.stage = 'output')}>details →</button>
      </div>
      <Candidates limit={5} controls="none" actions={false} />
    </section>
  </aside>
</div>

<style>
  .explore {
    display: grid;
    grid-template-columns: 236px minmax(0, 1fr) 320px;
    gap: 18px;
    align-items: start;
    padding-bottom: 30px;
  }
  .side {
    position: sticky;
    top: 80px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  h2 {
    margin: 10px 0 6px;
    font-size: 22px;
    letter-spacing: -0.02em;
  }
  .nextc h2 {
    margin: 0;
    font-size: 15px;
  }
  .muted {
    margin: 0;
    max-width: 680px;
  }
  .bh {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
  }
  .ctl {
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: flex-end;
  }
  .pipe {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    margin: 14px 0 0;
    font-size: 12px;
    color: var(--faint);
  }
  .pipe span:not(.arr) {
    font: 500 12px var(--mono);
    padding: 4px 9px;
    border-radius: 7px;
    border: 1px solid var(--border);
    background: var(--surface2);
    color: var(--muted);
  }
  .pipe span.on {
    color: var(--text);
    border-color: var(--border2);
  }
  .pipe span.a {
    color: var(--attn);
    background: var(--attn-soft);
    border-color: color-mix(in oklab, var(--attn) 45%, transparent);
  }
  .pipe span.f {
    color: var(--ffn);
    background: var(--ffn-soft);
    border-color: color-mix(in oklab, var(--ffn) 45%, transparent);
  }
  .tok-t {
    border-collapse: collapse;
    font: 500 13px var(--mono);
    width: 100%;
  }
  .tok-t th {
    font: 600 12px var(--sans);
    color: var(--faint);
    text-align: left;
    padding: 0 12px 6px 0;
  }
  .tok-t td {
    padding: 4px 12px 4px 0;
    border-top: 1px solid var(--border);
  }
  .tok-t button {
    font: 600 13px var(--mono);
    white-space: pre;
    border: 1px solid var(--border);
    background: var(--surface2);
    border-radius: 6px;
    padding: 2px 8px;
  }
  .tok-t button[aria-pressed='true'] {
    border-color: var(--accent);
    background: var(--accent-soft);
  }
  .link {
    border: 0;
    background: none;
    color: var(--out);
    font-size: 13px;
    padding: 0;
  }
  @media (max-width: 1180px) {
    .explore {
      grid-template-columns: 210px minmax(0, 1fr);
    }
    .side {
      grid-column: 1 / -1;
      position: static;
      display: grid;
      grid-template-columns: 1fr 1fr;
    }
  }
  @media (max-width: 860px) {
    .explore,
    .side {
      grid-template-columns: 1fr;
    }
    .ctl {
      align-items: flex-start;
    }
  }
</style>
