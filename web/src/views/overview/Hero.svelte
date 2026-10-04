<script lang="ts">
  import { app, go } from '../../lib/state.svelte';
  import { compact } from '../../lib/format';
  const c = $derived(app.cfg);
  const params = $derived(
    c ? c.vocab_size * c.d_model + c.n_layers * (4 * c.d_model ** 2 + 3 * c.d_model * c.mlp_hidden + 2 * c.d_model) + c.d_model : 0,
  );
</script>

<section class="hero" aria-labelledby="hero-h">
  <p class="pill"><span aria-hidden="true"></span>FLAME · a small language model trained from scratch in Rust</p>
  <h1 id="hero-h">See inside a <span class="grad">language model</span>,<br />one number at a time.</h1>
  <p class="lede">
    Type a sentence and watch a real 5.9M-parameter model turn tokens into vectors, mix them with attention, and choose what comes next. It
    runs entirely in your browser; every square, dot and ribbon is a value from the actual forward pass.
  </p>
  <div class="cta">
    <button class="btn accent lg" type="button" onclick={() => go('learn')}>Start the 4-step tour</button>
    <button class="btn ghost lg" type="button" onclick={() => go('explore')}>Open the explorer</button>
  </div>
  {#if c}
    <dl class="stats">
      <div>
        <dt>parameters</dt>
        <dd>{compact(params)}</dd>
      </div>
      <div>
        <dt>blocks</dt>
        <dd>{c.n_layers}</dd>
      </div>
      <div>
        <dt>heads / block</dt>
        <dd>{c.n_heads}</dd>
      </div>
      <div>
        <dt>dimensions</dt>
        <dd>{c.d_model}</dd>
      </div>
      <div>
        <dt>token vocabulary</dt>
        <dd>{c.vocab_size.toLocaleString()}</dd>
      </div>
    </dl>
  {/if}
</section>

<style>
  .hero {
    position: relative;
    text-align: center;
    padding: 64px 0 26px;
    max-width: 920px;
    margin: 0 auto;
  }
  .pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 550;
    color: var(--muted);
    padding: 6px 14px;
    margin: 0;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: color-mix(in oklab, var(--surface) 75%, transparent);
    backdrop-filter: blur(8px);
  }
  .pill span {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent-bg);
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0% {
      box-shadow: 0 0 0 0 color-mix(in oklab, var(--accent-bg) 45%, transparent);
    }
    100% {
      box-shadow: 0 0 0 8px transparent;
    }
  }
  h1 {
    font-size: clamp(34px, 5.4vw, 62px);
    line-height: 1.05;
    letter-spacing: -0.04em;
    font-weight: 750;
    margin: 22px 0 18px;
  }
  .grad {
    background: linear-gradient(95deg, #f97316, #ef4444 45%, #a855f7 90%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  .lede {
    font-size: 17px;
    color: var(--muted);
    max-width: 660px;
    margin: 0 auto;
  }
  .cta {
    display: flex;
    gap: 10px;
    justify-content: center;
    margin-top: 28px;
    flex-wrap: wrap;
  }
  .stats {
    display: flex;
    justify-content: center;
    gap: 34px;
    margin: 36px 0 0;
    flex-wrap: wrap;
  }
  .stats div {
    display: flex;
    flex-direction: column-reverse;
    align-items: center;
  }
  dd {
    margin: 0;
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -0.02em;
  }
  dt {
    font-size: 12.5px;
    color: var(--faint);
  }
</style>
