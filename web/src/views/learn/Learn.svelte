<script lang="ts">
  import { app, go, setFocus } from '../../lib/state.svelte';
  import { cosine, topAbs } from '../../lib/derived';
  import { diverging, robustScale, sequential } from '../../lib/colors';
  import { signed } from '../../lib/format';
  import CellGrid from '../../lib/viz/CellGrid.svelte';
  import ArcView from '../../lib/viz/ArcView.svelte';
  import HeadPicker from '../../lib/viz/HeadPicker.svelte';
  import Candidates from '../../lib/components/Candidates.svelte';
  import { attnHover, vecHover } from '../../lib/viz/hovers';

  const STEPS = [
    { t: 'Tokens → numbers', d: 'Embedding', c: '--emb', cls: 't-emb' },
    { t: 'Looking back', d: 'Self-attention', c: '--attn', cls: 't-attn' },
    { t: 'Thinking it over', d: 'Feed-forward (SwiGLU)', c: '--ffn', cls: 't-ffn' },
    { t: 'Choosing a token', d: 'Prediction', c: '--out', cls: 't-out' },
  ];

  let allDims = $state(false);
  let attnBlock = $state(0);
  let attnHead = $state(0);
  let ffnBlock = $state(2);

  const tr = $derived(app.trace!);
  const cfg = $derived(app.cfg!);
  const labels = $derived(tr.ids.map((id) => app.tok(id)));
  const fw = $derived(app.word(tr.ids[app.focus]));
  const st = $derived(STEPS[app.step]);

  const emb = $derived(tr.t('embed'));
  const embScale = $derived(robustScale(emb.data));
  const dims = $derived(allDims ? cfg.d_model : 32);
  const sims = $derived(Array.from({ length: tr.T }, (_, i) => Array.from({ length: tr.T }, (_, j) => cosine(emb.sub(i), emb.sub(j)))));

  const probs = $derived(tr.L(attnBlock, 'probs'));
  const hid = $derived(tr.L(ffnBlock, 'hidden'));
  const hidRow = $derived(hid.sub(app.focus));
  const hidScale = $derived(robustScale(hid.data));
  const topN = $derived(topAbs(hidRow, 6));
  const active = $derived(Array.from(hidRow).filter((v) => Math.abs(v) > hidScale * 0.25).length);

  function step(k: number) {
    app.step = Math.max(0, Math.min(3, k));
  }
  function onkey(e: KeyboardEvent) {
    const t = e.target as HTMLElement;
    if (t.closest('input, canvas, textarea, select')) return;
    if (e.key === 'ArrowRight') step(app.step + 1);
    if (e.key === 'ArrowLeft') step(app.step - 1);
  }
</script>

<svelte:window onkeydown={onkey} />

<nav aria-label="Lesson steps">
  <ol class="steps">
    {#each STEPS as s, i (s.t)}
      <li class:done={i < app.step} style="--sc: var({s.c})">
        <button type="button" aria-current={i === app.step ? 'step' : undefined} onclick={() => step(i)}>
          <span class="bar"><i></i></span>
          <span class="lbl"><span class="n">0{i + 1}</span><span class="t">{s.t}</span></span>
          <span class="d">{s.d}</span>
        </button>
      </li>
    {/each}
  </ol>
</nav>

<div class="learn">
  <article class="card pad lesson" style="--sc: var({st.c})" aria-labelledby="lesson-h">
    <span class="tag {st.cls}">Step {app.step + 1} · {st.d}</span>
    {#if app.step === 0}
      <h2 id="lesson-h">Every token becomes a list of numbers</h2>
      <p>
        The model can’t read letters. Your text is cut into <b>tokens</b> (whole words, pieces of words, or punctuation) and each token id
        picks one row of a <b>{cfg.vocab_size.toLocaleString()} × {cfg.d_model}</b> table: its embedding.
      </p>
      <p>Tokens used in similar ways end up with similar numbers, which is why the similarity grid lights up for related tokens.</p>
      <p class="tryit"><b>Try it:</b> hover or Tab to any square. Orange is positive, blue is negative, grey is near zero.</p>
      <details>
        <summary>The formula</summary>
        <p class="fx">x = E[token_id] <span class="faint">— E is {cfg.vocab_size} × {cfg.d_model}</span></p>
      </details>
    {:else if app.step === 1}
      <h2 id="lesson-h">What does “{fw}” pay attention to?</h2>
      <p>
        Before predicting, every token looks back at the tokens before it and decides how much each one matters. Thicker arcs mean more
        attention.
      </p>
      <p>
        Each of the {cfg.n_layers} blocks has <b>{cfg.n_heads} heads</b> that learn different habits. The model can never look ahead, so the top-right
        of the grid is hatched.
      </p>
      <p class="tryit"><b>Try it:</b> switch block or head, or click another token in the prompt to make it the one doing the looking.</p>
      <details>
        <summary>The formula</summary>
        <p class="fx">
          Q, K, V = x·W<sub>q</sub>, x·W<sub>k</sub>, x·W<sub>v</sub> <span class="faint">— Q and K rotated by RoPE</span><br />A =
          softmax(Q·Kᵀ / √{cfg.d_model / cfg.n_heads} + causal mask)<br />x ← x + (A·V)·W<sub>o</sub>
        </p>
      </details>
    {:else if app.step === 2}
      <h2 id="lesson-h">Mulling over what it gathered</h2>
      <p>
        Next, each token goes through a small network on its own. It <b>widens</b> from {cfg.d_model} to {cfg.mlp_hidden} neurons, a
        <b>gate</b>
        decides which may fire, then it <b>squeezes</b> back to {cfg.d_model}.
      </p>
      <p>Much of what the model has memorised about stories lives in these layers.</p>
      <p class="tryit">
        <b>Why SwiGLU?</b> One path says <i>how much</i> to let through, the other carries <i>what</i>. Multiplying them works like a dimmer
        switch.
      </p>
      <details>
        <summary>The formula</summary>
        <p class="fx">
          h = silu(x·W<sub>gate</sub>) ⊙ (x·W<sub>up</sub>) <span class="faint">— {cfg.mlp_hidden} wide</span><br />x ← x + h·W<sub
            >down</sub
          >
        </p>
      </details>
    {:else}
      <h2 id="lesson-h">Picking the next token</h2>
      <p>
        After {cfg.n_layers} blocks, the last token’s vector is compared with all {cfg.vocab_size.toLocaleString()} tokens. Softmax turns those
        scores into probabilities.
      </p>
      <p><b>Temperature</b> sets how adventurous the pick is: 0 always takes the favourite, higher values take risks.</p>
      <p class="tryit"><b>Try it:</b> drag the slider, then sample a token to watch the story grow.</p>
      <details>
        <summary>The formula</summary>
        <p class="fx">
          logits = RMSNorm(x)·Eᵀ <span class="faint">— tied embeddings, {cfg.d_model} → {cfg.vocab_size}</span><br />p = softmax(logits / T)
        </p>
      </details>
    {/if}
    <div class="nav">
      <button class="btn ghost" type="button" disabled={app.step === 0} onclick={() => step(app.step - 1)}>← Back</button>
      {#if app.step < 3}
        <button class="btn" type="button" onclick={() => step(app.step + 1)}>Next: {STEPS[app.step + 1].t} →</button>
      {:else}
        <button class="btn" type="button" onclick={() => go('explore')}>Open the explorer →</button>
      {/if}
    </div>
  </article>

  <div class="stack fade-in">
    {#if app.step === 0}
      <section class="card pad" aria-labelledby="emb-h">
        <div class="h-row">
          <h3 id="emb-h">Embedding lookup</h3>
          <label class="sub"><input type="checkbox" bind:checked={allDims} /> show all {cfg.d_model} dimensions</label>
        </div>
        <CellGrid
          label="Embedding vectors, {tr.T} tokens by {dims} dimensions"
          rows={tr.T}
          cols={dims}
          fit
          cell={14}
          cellH={18}
          minCell={2}
          gap={allDims ? 0 : 2}
          rowLabels={labels}
          activeRow={app.focus}
          onrowclick={setFocus}
          color={(i, j) => diverging(app.palette, emb.at(i, j), embScale)}
          onhover={(i, j) => vecHover('embed', 0, i, j)}
        />
      </section>
      <section class="card pad" aria-labelledby="sim-h">
        <div class="h-row">
          <h3 id="sim-h">Which tokens look alike?</h3>
          <span class="sub">cosine similarity of embeddings</span>
        </div>
        <CellGrid
          label="Cosine similarity between token embeddings"
          rows={tr.T}
          cols={tr.T}
          fit
          cell={30}
          minCell={6}
          rowLabels={labels}
          colLabels={labels}
          activeRow={app.focus}
          onrowclick={setFocus}
          color={(i, j) => diverging(app.palette, sims[i][j], 0.6)}
          onhover={(i, j) => ({
            tag: 'Similarity',
            title: `“${labels[i]}” vs “${labels[j]}”`,
            body: '1 means the two embeddings point the same way, 0 unrelated, −1 opposite.',
            calc: `cos = ${signed(sims[i][j])}`,
          })}
        />
      </section>
    {:else if app.step === 1}
      <section class="card pad" aria-labelledby="arc-h">
        <div class="h-row">
          <h3 id="arc-h">“{fw}” looks back at…</h3>
          <div class="seg" role="group" aria-label="Block">
            {#each Array.from({ length: cfg.n_layers }, (_, k) => k) as k (k)}
              <button type="button" aria-pressed={k === attnBlock} onclick={() => (attnBlock = k)} aria-label="Block {k + 1}"
                >B{k + 1}</button
              >
            {/each}
          </div>
        </div>
        <ArcView weights={probs.sub(attnHead, app.focus)} query={app.focus} />
      </section>
      <section class="card pad" aria-labelledby="hp-h">
        <div class="h-row">
          <h3 id="hp-h">Pick a head</h3>
          <span class="sub">block {attnBlock + 1} · each learns its own habit</span>
        </div>
        <HeadPicker layer={attnBlock} value={attnHead} onpick={(h) => (attnHead = h)} />
      </section>
      <section class="card pad" aria-labelledby="hm-h">
        <div class="h-row">
          <h3 id="hm-h">Who looks at whom</h3>
          <span class="sub">rows look at columns</span>
        </div>
        <CellGrid
          label="Attention weights, block {attnBlock + 1} head {attnHead + 1}"
          rows={tr.T}
          cols={tr.T}
          fit
          cell={34}
          minCell={6}
          rowLabels={labels}
          colLabels={labels}
          activeRow={app.focus}
          onrowclick={setFocus}
          color={(i, j) => (j > i ? null : sequential(app.palette, app.palette.attn, Math.pow(probs.at(attnHead, i, j), 0.7)))}
          text={(i, j) => (j <= i && probs.at(attnHead, i, j) >= 0.15 ? String(Math.round(probs.at(attnHead, i, j) * 100)) : '')}
          onhover={(i, j) => attnHover(attnBlock, attnHead, i, j)}
        />
      </section>
    {:else if app.step === 2}
      <section class="card pad" aria-labelledby="ffn-h">
        <div class="h-row">
          <h3 id="ffn-h">Neurons firing for “{fw}”</h3>
          <div class="seg" role="group" aria-label="Block">
            {#each Array.from({ length: cfg.n_layers }, (_, k) => k) as k (k)}
              <button type="button" aria-pressed={k === ffnBlock} onclick={() => (ffnBlock = k)} aria-label="Block {k + 1}">B{k + 1}</button
              >
            {/each}
          </div>
        </div>
        <CellGrid
          label="{cfg.mlp_hidden} SwiGLU neuron activations for token {fw} in block {ffnBlock + 1}"
          rows={Math.ceil(cfg.mlp_hidden / 32)}
          cols={32}
          fit
          cell={18}
          minCell={7}
          color={(i, j) => (i * 32 + j < cfg.mlp_hidden ? diverging(app.palette, hidRow[i * 32 + j], hidScale) : null)}
          onhover={(i, j) => (i * 32 + j < cfg.mlp_hidden ? vecHover('hidden', ffnBlock, app.focus, i * 32 + j) : null)}
        />
        <p class="note">
          {active} of {cfg.mlp_hidden} neurons are clearly active (|h| above a quarter of the typical scale). Strongest:
          {#each topN as k, n (k)}<span class="mono">#{k + 1}</span> ({signed(hidRow[k], 2)}){n < topN.length - 1 ? ', ' : '.'}{/each}
        </p>
      </section>
      <section class="card pad" aria-labelledby="flow-h">
        <div class="h-row">
          <h3 id="flow-h">Widen → gate → squeeze</h3>
          <span class="sub">block {ffnBlock + 1}, token “{fw}”</span>
        </div>
        {#each [['mlp_norm', 'Comes in', `${cfg.d_model} numbers`, cfg.d_model], ['gate', 'Gate · silu(x·W_gate)', 'how much to let through', cfg.mlp_hidden], ['up', 'Content · x·W_up', 'what to carry', cfg.mlp_hidden], ['mlp_out', 'Goes out · h·W_down', `back to ${cfg.d_model}, added to the stream`, cfg.d_model]] as [k, t, s, n] (k)}
          {@const v = tr.L(ffnBlock, k as string).sub(app.focus)}
          {@const sc = robustScale(tr.L(ffnBlock, k as string).data)}
          <div class="prow">
            <div><b>{t}</b><small>{s}</small></div>
            <CellGrid
              label="{t}, {n} values"
              rows={1}
              cols={n as number}
              fit
              cell={10}
              cellH={22}
              minCell={1}
              gap={0}
              radius={0}
              color={(_, j) => diverging(app.palette, v[j], sc)}
              onhover={(_, j) => vecHover(k as string, ffnBlock, app.focus, j)}
            />
          </div>
        {/each}
      </section>
    {:else}
      <section class="card pad" aria-labelledby="pred-h">
        <div class="h-row">
          <h3 id="pred-h">Most likely next tokens</h3>
          <span class="sub">after “{app.word(tr.ids[tr.T - 1])}”</span>
        </div>
        <Candidates limit={10} controls="temperature" />
      </section>
    {/if}
  </div>
</div>

<style>
  .steps {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    list-style: none;
    margin: 0 0 18px;
    padding: 0;
  }
  .steps button {
    width: 100%;
    text-align: left;
    border: 0;
    background: none;
    padding: 0 14px 4px 0;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .bar {
    display: block;
    height: 4px;
    border-radius: 99px;
    background: var(--border);
    overflow: hidden;
  }
  .bar i {
    display: block;
    height: 100%;
    width: 0;
    background: var(--sc);
    transition: width 0.4s;
  }
  li.done .bar i,
  [aria-current='step'] .bar i {
    width: 100%;
  }
  .lbl {
    display: flex;
    gap: 10px;
    align-items: baseline;
  }
  .n {
    font: 600 12px var(--mono);
    color: var(--faint);
  }
  .t {
    font-weight: 600;
    font-size: 14px;
    color: var(--muted);
  }
  .d {
    font-size: 12.5px;
    color: var(--faint);
  }
  [aria-current='step'] .t {
    color: var(--text);
  }
  [aria-current='step'] .n {
    color: var(--sc);
  }
  .learn {
    display: grid;
    grid-template-columns: minmax(300px, 380px) minmax(0, 1fr);
    gap: 18px;
    align-items: start;
    padding-bottom: 30px;
  }
  .lesson {
    position: sticky;
    top: 80px;
  }
  .lesson h2 {
    font-size: 26px;
    line-height: 1.15;
    letter-spacing: -0.025em;
    margin: 14px 0 12px;
  }
  .lesson p {
    color: var(--muted);
    margin: 0 0 12px;
  }
  .lesson p b {
    color: var(--text);
  }
  .tryit {
    border-left: 3px solid var(--sc);
    background: var(--surface2);
    padding: 10px 14px;
    border-radius: 0 10px 10px 0;
    font-size: 14px;
  }
  details {
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-top: 14px;
  }
  summary {
    cursor: pointer;
    padding: 10px 14px;
    font-weight: 600;
    font-size: 13.5px;
  }
  .fx {
    padding: 0 14px 12px;
    margin: 0 !important;
    font: 13px/1.8 var(--mono);
  }
  .nav {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    margin-top: 20px;
  }
  .prow {
    display: grid;
    grid-template-columns: 190px minmax(0, 1fr);
    gap: 14px;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px dashed var(--border);
  }
  .prow:last-child {
    border: 0;
  }
  .prow b {
    display: block;
    font-size: 13.5px;
  }
  .prow small {
    color: var(--faint);
    font-size: 12px;
  }
  label.sub {
    display: flex;
    gap: 6px;
    align-items: center;
    cursor: pointer;
  }
  @media (max-width: 900px) {
    .learn {
      grid-template-columns: 1fr;
    }
    .lesson {
      position: static;
    }
    .steps {
      grid-template-columns: repeat(2, 1fr);
      row-gap: 14px;
    }
    .prow {
      grid-template-columns: 1fr;
    }
  }
</style>
