<script lang="ts">
  import { app, go, generate } from '../../lib/state.svelte';
  import ModelDiagram from './ModelDiagram.svelte';
  import LossChart from './LossChart.svelte';

  const c = $derived(app.cfg);
</script>

{#if app.trace && c}
  <ModelDiagram />
  <div class="feat">
    <button class="card fcard" type="button" onclick={() => go('learn')}>
      <span class="mono faint">01</span><b>Learn →</b><span>Four short steps from tokens to a prediction. No maths needed.</span>
    </button>
    <button class="card fcard" type="button" onclick={() => go('explore')}>
      <span class="mono faint">02</span><b>Explore →</b><span
        >Every block, head and neuron. Click any attention cell to see its dot product.</span
      >
    </button>
    <button class="card fcard" type="button" onclick={() => generate(10)} disabled={app.busy}>
      <span class="mono faint">03</span><b>Generate →</b><span>Sample ten tokens and watch the whole diagram update.</span>
    </button>
  </div>
  <div class="grid-2 bottom">
    <LossChart />
    <section class="card pad" aria-labelledby="arch-h">
      <div class="h-row">
        <h2 id="arch-h">Architecture</h2>
        <span class="sub">LLaMA-style decoder</span>
      </div>
      <table>
        <tbody>
          <tr><th scope="row">Blocks × heads</th><td>{c.n_layers} × {c.n_heads} (head dim {c.d_model / c.n_heads})</td></tr>
          <tr><th scope="row">Model width</th><td>{c.d_model}</td></tr>
          <tr><th scope="row">Feed-forward</th><td>SwiGLU, {c.mlp_hidden} hidden</td></tr>
          <tr><th scope="row">Normalisation</th><td>RMSNorm (ε = {c.norm_eps}), pre-norm</td></tr>
          <tr><th scope="row">Positions</th><td>RoPE, θ = {c.rope_theta.toLocaleString()}</td></tr>
          <tr><th scope="row">Vocabulary</th><td>{c.vocab_size.toLocaleString()} byte-level BPE tokens</td></tr>
          <tr><th scope="row">Context</th><td>{c.max_seq_len} tokens</td></tr>
          <tr><th scope="row">Output head</th><td>{c.tie_embeddings ? 'tied to the embedding table' : 'separate'}</td></tr>
          <tr
            ><th scope="row">Training data</th><td
              ><a href="https://huggingface.co/datasets/roneneldan/TinyStories" rel="noopener" target="_blank">TinyStories</a></td
            ></tr
          >
        </tbody>
      </table>
      <p class="note">
        Trained with <a href="https://github.com/RTensorLabs/FLAME" rel="noopener" target="_blank">FLAME</a>. This page runs an independent
        re-implementation compiled to WebAssembly; automated parity tests check it against FLAME token-for-token.
      </p>
    </section>
  </div>
{/if}

<style>
  .feat {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin: 18px 0;
  }
  .fcard {
    display: flex;
    flex-direction: column;
    gap: 6px;
    text-align: left;
    padding: 18px 20px;
    transition:
      transform 0.2s,
      border-color 0.2s;
  }
  .fcard:hover:not(:disabled) {
    transform: translateY(-3px);
    border-color: var(--border2);
  }
  .fcard b {
    font-size: 15px;
  }
  .fcard span:last-child {
    font-size: 13.5px;
    color: var(--muted);
  }
  .bottom {
    margin-bottom: 20px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
  }
  th,
  td {
    text-align: left;
    padding: 7px 0;
    border-bottom: 1px solid var(--border);
  }
  th {
    font-weight: 500;
    color: var(--muted);
    width: 45%;
  }
  @media (max-width: 860px) {
    .feat {
      grid-template-columns: 1fr;
    }
  }
</style>
