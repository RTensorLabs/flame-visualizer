<!--
  Canvas grid of coloured cells (matrices, vectors, neuron maps). Canvas keeps thousands of
  cells fast; hover, click and arrow-key navigation report back through callbacks.
-->
<script lang="ts">
  import { app, type Inspect } from '../state.svelte';
  import { css, onColor, type RGB } from '../colors';
  import { hideTip, showInspect } from '../inspect.svelte';

  interface Props {
    rows: number;
    cols: number;
    /** Cell colour, or null for a masked (hatched) cell. */
    color: (i: number, j: number) => RGB | null;
    text?: (i: number, j: number) => string;
    rowLabels?: string[];
    colLabels?: string[];
    labelWidth?: number;
    cell?: number;
    cellH?: number;
    minCell?: number;
    gap?: number;
    radius?: number;
    fit?: boolean;
    activeRow?: number;
    pin?: { i: number; j: number } | null;
    label: string;
    onhover?: (i: number, j: number) => Inspect | null;
    onselect?: (i: number, j: number) => void;
    onrowclick?: (i: number) => void;
  }
  let {
    rows,
    cols,
    color,
    text,
    rowLabels,
    colLabels,
    labelWidth = 92,
    cell = 26,
    cellH,
    minCell = 4,
    gap = 2,
    radius = 3,
    fit = false,
    activeRow = -1,
    pin = null,
    label,
    onhover,
    onselect,
    onrowclick,
  }: Props = $props();

  let canvas: HTMLCanvasElement | undefined = $state();
  let width = $state(0);
  let cursor = $state<{ i: number; j: number } | null>(null);

  const lw = $derived(rowLabels ? labelWidth : 0);
  const size = $derived(fit && width ? Math.max(minCell, Math.min(cell, (width - lw - 8 - gap * (cols - 1)) / cols)) : cell);
  const h = $derived(cellH ?? size);
  const cw = $derived(cols * size + (cols - 1) * gap);
  const chh = $derived(rows * h + (rows - 1) * gap);

  $effect(() => {
    if (!canvas) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.ceil(cw * dpr);
    canvas.height = Math.ceil(chh * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, chh);
    const p = app.palette;
    const r = Math.min(radius, size / 3, h / 3);
    const fontPx = Math.max(8, Math.min(11, size * 0.4));
    ctx.font = `500 ${fontPx}px ${getComputedStyle(canvas).getPropertyValue('--mono') || 'monospace'}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const x = j * (size + gap);
        const y = i * (h + gap);
        const c = color(i, j);
        ctx.beginPath();
        ctx.roundRect(x, y, size, h, r);
        if (!c) {
          ctx.save();
          ctx.clip();
          ctx.strokeStyle = css(p.cell);
          ctx.lineWidth = 1;
          for (let s = -h; s < size; s += 4) {
            ctx.beginPath();
            ctx.moveTo(x + s, y + h);
            ctx.lineTo(x + s + h, y);
            ctx.stroke();
          }
          ctx.restore();
          continue;
        }
        ctx.fillStyle = css(c);
        ctx.fill();
        const tx = text?.(i, j);
        if (tx && size >= 18) {
          ctx.fillStyle = onColor(c);
          ctx.fillText(tx, x + size / 2, y + h / 2 + 0.5);
        }
      }
    }
    const outline = (i: number, j: number, col: string, w: number) => {
      ctx.strokeStyle = col;
      ctx.lineWidth = w;
      ctx.beginPath();
      ctx.roundRect(j * (size + gap) - 1, i * (h + gap) - 1, size + 2, h + 2, r + 1);
      ctx.stroke();
    };
    if (activeRow >= 0 && activeRow < rows) {
      ctx.strokeStyle = css(p.attn, 0.7);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(-0.5, activeRow * (h + gap) - 1.5, cw + 1, h + 3, r + 1);
      ctx.stroke();
    }
    if (pin && pin.i < rows && pin.j < cols) outline(pin.i, pin.j, css(p.text), 2);
    if (cursor) outline(cursor.i, cursor.j, css(p.out), 2.5);
  });

  function cellAt(ev: MouseEvent): { i: number; j: number } | null {
    const rect = canvas!.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    const j = Math.floor(x / (size + gap));
    const i = Math.floor(y / (h + gap));
    if (i < 0 || j < 0 || i >= rows || j >= cols) return null;
    if (x - j * (size + gap) > size || y - i * (h + gap) > h) return null;
    return { i, j };
  }

  function report(c: { i: number; j: number } | null, x: number, y: number) {
    if (!c || !onhover) return hideTip();
    const info = onhover(c.i, c.j);
    if (info) showInspect(info, x, y);
    else hideTip();
  }

  function onmove(ev: MouseEvent) {
    report(cellAt(ev), ev.clientX, ev.clientY);
  }

  function onclick(ev: MouseEvent) {
    const c = cellAt(ev);
    if (c && onselect && color(c.i, c.j)) onselect(c.i, c.j);
  }

  function onkeydown(ev: KeyboardEvent) {
    const c = cursor ?? { i: Math.max(0, activeRow), j: 0 };
    const moves: Record<string, [number, number]> = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    if (ev.key in moves) {
      ev.preventDefault();
      const [di, dj] = moves[ev.key];
      const n = { i: Math.max(0, Math.min(rows - 1, c.i + di)), j: Math.max(0, Math.min(cols - 1, c.j + dj)) };
      cursor = n;
      const rect = canvas!.getBoundingClientRect();
      report(n, rect.left + n.j * (size + gap) + size / 2, rect.top + n.i * (h + gap));
    } else if ((ev.key === 'Enter' || ev.key === ' ') && cursor && onselect && color(cursor.i, cursor.j)) {
      ev.preventDefault();
      onselect(cursor.i, cursor.j);
    } else if (ev.key === 'Escape') {
      cursor = null;
      hideTip();
    }
  }
</script>

<div class="cg" bind:clientWidth={width}>
  {#if colLabels}
    <div class="cols" style:padding-left="{lw}px" aria-hidden="true">
      {#each colLabels as c, j (j)}
        <span style:width="{size}px" style:margin-right="{j < cols - 1 ? gap : 0}px"><span>{c}</span></span>
      {/each}
    </div>
  {/if}
  <div class="body">
    {#if rowLabels}
      <div class="rows" style:width="{lw}px">
        {#each rowLabels as rl, i (i)}
          {#if onrowclick}
            <button
              type="button"
              class="rl"
              class:on={i === activeRow}
              style:height="{h}px"
              style:margin-bottom="{i < rows - 1 ? gap : 0}px"
              style:font-size="{Math.min(13, Math.max(9, h * 0.75))}px"
              onclick={() => onrowclick(i)}
              aria-label="Focus token {i + 1}: {rl}">{rl}</button
            >
          {:else}
            <span
              class="rl"
              class:on={i === activeRow}
              style:height="{h}px"
              style:margin-bottom="{i < rows - 1 ? gap : 0}px"
              style:font-size="{Math.min(13, Math.max(9, h * 0.75))}px">{rl}</span
            >
          {/if}
        {/each}
      </div>
    {/if}
    <!-- role="application" is the ARIA role for a custom keyboard-driven widget; Svelte's linter
         does not list it as interactive, so its two heuristics are silenced here deliberately. -->
    <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
    <div
      class="cv"
      class:clickable={!!onselect}
      tabindex="0"
      role="application"
      aria-roledescription="chart"
      aria-label="{label}. Use arrow keys to inspect values{onselect ? ', Enter to select' : ''}."
      onmousemove={onmove}
      onmouseleave={hideTip}
      {onclick}
      {onkeydown}
      onblur={() => {
        cursor = null;
        hideTip();
      }}
    >
      <canvas bind:this={canvas} style:width="{cw}px" style:height="{chh}px" aria-hidden="true"></canvas>
    </div>
  </div>
</div>

<style>
  .cg {
    max-width: 100%;
    overflow-x: auto;
    padding: 4px 4px 4px 0;
  }
  .body {
    display: flex;
    align-items: flex-start;
  }
  .cv {
    flex: none;
    cursor: crosshair;
    border-radius: 4px;
  }
  .cv.clickable {
    cursor: pointer;
  }
  canvas {
    display: block;
  }
  .rows {
    flex: none;
    display: flex;
    flex-direction: column;
    padding-right: 8px;
  }
  .rl {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    border: 0;
    background: none;
    padding: 0;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-family: var(--mono);
  }
  button.rl:hover {
    color: var(--text);
  }
  .rl.on {
    color: var(--accent);
    font-weight: 700;
  }
  .cols {
    display: flex;
    height: 64px;
    align-items: flex-end;
    margin-bottom: 4px;
  }
  .cols > span {
    flex: none;
    display: flex;
    justify-content: center;
    font-size: 11px;
    color: var(--muted);
    font-family: var(--mono);
  }
  .cols > span > span {
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    white-space: nowrap;
    max-height: 62px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
