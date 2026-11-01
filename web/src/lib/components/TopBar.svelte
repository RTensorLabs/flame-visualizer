<script lang="ts">
  import { app, setTheme, type View } from '../state.svelte';
  import { compact } from '../format';

  const links: [View, string][] = [
    ['overview', 'Overview'],
    ['learn', 'Learn'],
    ['explore', 'Explore'],
  ];
  const params = $derived.by(() => {
    const c = app.cfg;
    if (!c) return 0;
    return c.vocab_size * c.d_model + c.n_layers * (4 * c.d_model ** 2 + 3 * c.d_model * c.mlp_hidden + 2 * c.d_model) + c.d_model;
  });
</script>

<header class="top">
  <div class="wrap bar">
    <a class="brand" href="#/overview" aria-label="FLAME Visualizer home">
      <img class="logo" src="./brand/flame-logo.png" width="32" height="32" alt="" />
      FLAME <span class="faint">Visualizer</span>
    </a>
    <nav aria-label="Primary">
      <ul>
        {#each links as [v, label] (v)}
          <li><a href="#/{v}" aria-current={app.view === v ? 'page' : undefined}>{label}</a></li>
        {/each}
      </ul>
    </nav>
    <div class="sp"></div>
    {#if app.cfg}
      <p class="chip" title="Model loaded in your browser">
        <span class="dot" aria-hidden="true"></span><b>TinyStories-5M</b>
        <span class="hide-s">{compact(params)} params · {app.cfg.n_layers} blocks · {app.cfg.n_heads} heads · d={app.cfg.d_model}</span>
      </p>
    {/if}
    <button
      class="icon"
      type="button"
      aria-label="Dark theme"
      aria-pressed={app.theme === 'dark'}
      onclick={() => setTheme(app.theme === 'dark' ? 'light' : 'dark')}
    >
      {#if app.theme === 'dark'}
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"
          ><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></svg
        >
      {:else}
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          aria-hidden="true"
          ><circle cx="12" cy="12" r="4" /><path
            d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"
          /></svg
        >
      {/if}
    </button>
  </div>
</header>

<style>
  .top {
    position: sticky;
    top: 0;
    z-index: 40;
    background: color-mix(in oklab, var(--bg) 84%, transparent);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
  }
  .bar {
    display: flex;
    align-items: center;
    gap: 22px;
    height: 60px;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    color: var(--text);
    text-decoration: none;
    letter-spacing: -0.01em;
  }
  .brand .faint {
    font-weight: 450;
  }
  .logo {
    width: 32px;
    height: 32px;
    flex: none;
    object-fit: contain;
  }
  ul {
    display: flex;
    gap: 2px;
    list-style: none;
    margin: 0;
    padding: 3px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
  }
  nav a {
    display: block;
    padding: 6px 14px;
    border-radius: 7px;
    font-weight: 550;
    font-size: 13.5px;
    color: var(--muted);
    text-decoration: none;
  }
  nav a:hover {
    color: var(--text);
  }
  nav a[aria-current='page'] {
    background: var(--text);
    color: var(--bg);
  }
  .sp {
    flex: 1;
  }
  .chip {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0;
    font-size: 12.5px;
    color: var(--muted);
    padding: 6px 12px;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--surface);
  }
  .chip b {
    color: var(--text);
  }
  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 0 3px rgb(34 197 94 / 0.2);
  }
  .icon {
    width: 36px;
    height: 36px;
    display: grid;
    place-items: center;
    border: 1px solid var(--border);
    border-radius: 9px;
    background: var(--surface);
    color: var(--muted);
  }
  .icon:hover {
    color: var(--text);
  }
  @media (max-width: 900px) {
    .hide-s {
      display: none;
    }
    .bar {
      gap: 10px;
    }
    .brand .faint {
      display: none;
    }
  }
  @media (max-width: 560px) {
    .chip {
      display: none;
    }
  }
</style>
