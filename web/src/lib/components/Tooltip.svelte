<script lang="ts">
  import { tip } from '../inspect.svelte';
  let el: HTMLDivElement | undefined = $state();
  const left = $derived(el ? Math.min(tip.x + 14, innerWidth - el.offsetWidth - 8) : tip.x);
</script>

<div bind:this={el} class="tip" class:show={tip.show} style:left="{left}px" style:top="{tip.y - 38}px" aria-hidden="true">{tip.text}</div>

<style>
  .tip {
    position: fixed;
    z-index: 90;
    pointer-events: none;
    background: var(--text);
    color: var(--bg);
    font: 500 12px var(--mono);
    padding: 5px 9px;
    border-radius: 6px;
    opacity: 0;
    transition: opacity 0.08s;
    white-space: nowrap;
  }
  .show {
    opacity: 1;
  }
</style>
