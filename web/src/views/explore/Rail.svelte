<script lang="ts">
  import { app, type Stage } from '../../lib/state.svelte';
  import { pct } from '../../lib/format';

  type Item = { key: string; stage: Stage; block?: number; title: string; sub: string; color: string };
  const items = $derived.by<Item[]>(() => {
    const c = app.cfg!;
    return [
      { key: 'tokens', stage: 'tokens', title: 'Tokens', sub: `${app.trace!.T}`, color: '--faint' },
      { key: 'embedding', stage: 'embedding', title: 'Embedding', sub: `${c.vocab_size}×${c.d_model}`, color: '--emb' },
      ...Array.from({ length: c.n_layers }, (_, l): Item => ({
        key: `b${l}`,
        stage: 'block',
        block: l,
        title: `Block ${l + 1}`,
        sub: 'attn + mlp',
        color: '--attn',
      })),
      { key: 'output', stage: 'output', title: 'Output', sub: `→ ${c.vocab_size}`, color: '--out' },
    ];
  });
  const current = $derived(app.stage === 'block' ? `b${app.block}` : app.stage);
  const last = $derived(app.trace!.T - 1);

  function pick(it: Item) {
    app.stage = it.stage;
    if (it.block !== undefined) app.block = it.block;
  }
  function onkeydown(e: KeyboardEvent) {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    e.preventDefault();
    const k = items.findIndex((it) => it.key === current) + (e.key === 'ArrowDown' ? 1 : -1);
    const it = items[Math.max(0, Math.min(items.length - 1, k))];
    pick(it);
    (e.currentTarget as HTMLElement).closest('ol')?.querySelector<HTMLButtonElement>(`[data-key="${it.key}"]`)?.focus();
  }
</script>

<nav class="card rail" aria-label="Model architecture">
  <h2 class="eyebrow">Architecture</h2>
  <ol>
    {#each items as it (it.key)}
      <li>
        <button
          type="button"
          data-key={it.key}
          aria-current={it.key === current ? 'location' : undefined}
          style="--nc: var({it.color})"
          onclick={() => pick(it)}
          {onkeydown}
        >
          <span class="nt">{it.title}<small>{it.sub}</small></span>
          {#if it.block !== undefined && app.trace!.has(`L${it.block}.lens_ids`)}
            {@const id = app.trace!.L(it.block, 'lens_ids').at(last, 0)}
            {@const p = app.trace!.L(it.block, 'lens_p').at(last, 0)}
            <span class="lens"
              ><span>guess <b>{app.tok(id)}</b></span><span class="track" aria-hidden="true"><i style:width="{p * 100}%"></i></span><span
                class="pp">{pct(p)}</span
              ></span
            >
          {/if}
        </button>
      </li>
    {/each}
  </ol>
  <p class="foot">
    The line is the <b>residual stream</b>: each block reads from it and adds back. “guess” is what the model would predict next if it
    stopped there (logit lens).
  </p>
</nav>

<style>
  .rail {
    position: sticky;
    top: 80px;
    padding: 12px 10px;
  }
  h2 {
    padding: 4px 10px 10px;
  }
  ol {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  li {
    position: relative;
  }
  li::before {
    content: '';
    position: absolute;
    left: 14px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--border);
  }
  li:first-child::before {
    top: 50%;
  }
  li:last-child::before {
    bottom: 50%;
  }
  button {
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 5px;
    width: 100%;
    text-align: left;
    border: 1px solid transparent;
    background: none;
    border-radius: 10px;
    padding: 9px 10px 9px 30px;
  }
  button::after {
    content: '';
    position: absolute;
    left: 10px;
    top: 50%;
    width: 10px;
    height: 10px;
    margin-top: -5px;
    border-radius: 50%;
    background: var(--surface);
    border: 2px solid var(--nc);
  }
  button:hover {
    background: var(--surface2);
  }
  button[aria-current='location'] {
    background: var(--surface2);
    border-color: var(--border);
  }
  button[aria-current='location']::after {
    background: var(--nc);
  }
  .nt {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-weight: 600;
    font-size: 13.5px;
  }
  .nt small {
    font: 500 11px var(--mono);
    color: var(--faint);
  }
  .lens {
    display: grid;
    grid-template-columns: minmax(0, auto) 1fr 38px;
    gap: 7px;
    align-items: center;
    font-size: 12px;
    color: var(--muted);
  }
  .lens > span:first-child {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .lens b {
    color: var(--text);
    font-family: var(--mono);
    white-space: pre;
  }
  .lens .track {
    height: 4px;
  }
  .lens .track > i {
    background: var(--out);
  }
  .pp {
    font: 500 11px var(--mono);
    text-align: right;
  }
  .foot {
    font-size: 12px;
    color: var(--faint);
    padding: 10px 10px 2px;
    border-top: 1px solid var(--border);
    margin: 8px 0 0;
  }
  @media (max-width: 1180px) {
    .rail {
      position: static;
    }
  }
</style>
