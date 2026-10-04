<script lang="ts">
  import { onMount } from 'svelte';
  import { app, boot } from './lib/state.svelte';
  import TopBar from './lib/components/TopBar.svelte';
  import PromptBar from './lib/components/PromptBar.svelte';
  import ModelStatus from './lib/components/ModelStatus.svelte';
  import Tooltip from './lib/components/Tooltip.svelte';
  import ParticleBg from './views/overview/ParticleBg.svelte';
  import Hero from './views/overview/Hero.svelte';
  import Overview from './views/overview/Overview.svelte';
  import Learn from './views/learn/Learn.svelte';
  import Explore from './views/explore/Explore.svelte';

  const titles = {
    overview: 'FLAME Visualizer · see inside a language model',
    learn: 'Learn · FLAME Visualizer',
    explore: 'Explore · FLAME Visualizer',
  };
  $effect(() => {
    document.title = titles[app.view];
  });
  onMount(() => void boot());
</script>

<a class="skip-link" href="#main">Skip to content</a>
{#if app.view === 'overview'}<ParticleBg />{/if}
<TopBar />
<main id="main" class="wrap" tabindex="-1">
  {#if app.view === 'overview'}<Hero />{/if}
  <ModelStatus />
  <PromptBar />
  {#if app.status === 'ready' && app.trace}
    {#if app.view === 'overview'}
      <Overview />
    {:else if app.view === 'learn'}
      <Learn />
    {:else}
      <Explore />
    {/if}
  {/if}
</main>
<footer class="wrap">
  <p>
    TinyStories-5M, trained with <a href="https://github.com/RTensorLabs/FLAME" rel="noopener" target="_blank">FLAME</a>. Runs locally in
    your browser via WebAssembly; nothing you type leaves your device.
    {#if app.info}Weights: <a
        href={app.info.source.startsWith('http') ? app.info.source.replace(/resolve\/[^/]+\/$/, '') : 'https://huggingface.co'}
        rel="noopener"
        target="_blank">model card</a
      >.{/if}
  </p>
</footer>
<Tooltip />
<p class="sr-only" aria-live="polite">{app.inspect && app.view !== 'explore' ? `${app.inspect.title}: ${app.inspect.calc ?? ''}` : ''}</p>

<style>
  main {
    position: relative;
    z-index: 1;
    outline: none;
  }
  footer {
    position: relative;
    z-index: 1;
    padding: 24px 28px 40px;
    font-size: 13px;
    color: var(--faint);
    text-align: center;
  }
</style>
