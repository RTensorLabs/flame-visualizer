<script lang="ts">
  import { app, generate, MAX_TOKENS, PRESETS, setFocus, setText } from '../state.svelte';

  const ready = $derived(app.status === 'ready');
</script>

<section class="prompt card" aria-label="Prompt">
  <form
    class="row"
    onsubmit={(e) => {
      e.preventDefault();
      void setText(app.text, true);
    }}
  >
    <label for="prompt-input" class="eyebrow">Prompt</label>
    <input
      id="prompt-input"
      value={app.text}
      oninput={(e) => setText(e.currentTarget.value)}
      maxlength={2000}
      spellcheck="false"
      autocomplete="off"
      disabled={!ready}
      aria-describedby="prompt-help"
    />
    <button class="btn ghost" type="submit" disabled={!ready}>Run</button>
    <button
      class="btn accent"
      type="button"
      disabled={!ready || app.busy}
      onclick={() => generate(1)}
      title="Sample one token with the current settings">Next token</button
    >
    <button class="btn ghost" type="button" disabled={!ready || app.busy} onclick={() => generate(10)} title="Sample ten tokens">+10</button
    >
  </form>
  <div class="presets">
    <span class="faint" id="prompt-help">Try:</span>
    {#each PRESETS as p (p)}
      <button type="button" class="preset" disabled={!ready} onclick={() => setText(p, true)}>{p.split(' ').slice(0, 4).join(' ')}…</button>
    {/each}
  </div>
  {#if app.trace}
    <div class="toks" aria-busy={app.busy}>
      <ul aria-label="Tokens. Select one to follow it through the model.">
        {#each app.trace.ids as id, i (i)}
          <li>
            <button
              type="button"
              class="tok"
              class:new={i === app.lastAdded}
              aria-pressed={i === app.focus}
              onclick={() => setFocus(i)}
              title="token #{id}">{app.tok(id)}<small>{id}</small></button
            >
          </li>
        {/each}
      </ul>
      <p class="meta faint">
        {app.trace.T} tokens{#if app.truncated}
          · showing the last {MAX_TOKENS} ({app.truncated} earlier hidden){/if} · traced in {app.trace.ms.toFixed(0)} ms{#if app.busy}
          · <span class="spin" aria-hidden="true"></span> updating{/if}
      </p>
    </div>
  {/if}
</section>

<style>
  .prompt {
    margin: 22px 0 18px;
  }
  .row {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 10px 10px 10px 16px;
    border-bottom: 1px solid var(--border);
  }
  input {
    flex: 1;
    min-width: 0;
    border: 1px solid transparent;
    border-radius: 8px;
    outline: 0;
    background: none;
    font-size: 16px;
    font-weight: 500;
    padding: 7px 8px;
  }
  input:focus-visible {
    border-color: var(--focus);
  }
  .presets {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    align-items: center;
    padding: 10px 16px 0;
    font-size: 12.5px;
  }
  .preset {
    border: 1px solid var(--border);
    background: var(--surface2);
    border-radius: 999px;
    padding: 4px 11px;
    font-size: 12.5px;
    color: var(--muted);
  }
  .preset:hover:not(:disabled) {
    color: var(--text);
    border-color: var(--border2);
  }
  .toks {
    padding: 12px 16px 12px;
  }
  ul {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .tok {
    display: inline-flex;
    align-items: baseline;
    gap: 5px;
    border: 1px solid var(--border);
    background: var(--surface2);
    border-radius: 7px;
    padding: 3px 8px;
    font: 500 13.5px var(--mono);
    white-space: pre;
  }
  .tok small {
    font-size: 10.5px;
    color: var(--faint);
  }
  .tok:hover {
    border-color: var(--border2);
  }
  .tok[aria-pressed='true'] {
    border-color: var(--accent);
    background: var(--accent-soft);
    box-shadow: 0 0 0 3px color-mix(in oklab, var(--accent) 16%, transparent);
  }
  .tok.new {
    animation: pop 0.45s cubic-bezier(0.2, 0.9, 0.3, 1.3);
  }
  @keyframes pop {
    from {
      transform: scale(0.7);
      opacity: 0;
    }
  }
  .meta {
    margin: 10px 0 0;
    font-size: 12.5px;
  }
  .spin {
    display: inline-block;
    width: 10px;
    height: 10px;
    border: 2px solid var(--border2);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    vertical-align: -1px;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  @media (max-width: 700px) {
    .row {
      flex-wrap: wrap;
    }
    input {
      flex-basis: 100%;
    }
  }
</style>
