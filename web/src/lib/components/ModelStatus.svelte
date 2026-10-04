<script lang="ts">
  import { app, loadModel } from '../state.svelte';
</script>

{#if app.status === 'loading'}
  <div class="status card pad" role="status">
    <p><b>Loading the model into your browser…</b> <span class="faint">23 MB, cached after the first visit</span></p>
    <div
      class="track"
      role="progressbar"
      aria-label="Model download"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(app.progress * 100)}
    >
      <i style:width="{app.progress * 100}%"></i>
    </div>
  </div>
{:else if app.status === 'error'}
  <div class="status card pad err" role="alert">
    <p><b>Could not load the model.</b> {app.error}</p>
    <button class="btn" type="button" onclick={loadModel}>Try again</button>
  </div>
{/if}

<style>
  .status {
    margin: 18px 0;
  }
  p {
    margin: 0 0 12px;
  }
  .track > i {
    background: var(--accent-bg);
  }
  .err {
    border-color: var(--k);
  }
</style>
