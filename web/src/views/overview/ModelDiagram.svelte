<!--
  The whole model as one animated diagram (Transformer-Explainer style), drawn from the real trace:
  tokens → embeddings → Q/K/V → attention → blend → SwiGLU MLP → (× blocks) → LM head → probabilities.
-->
<script lang="ts">
  import { app, go, setFocus, type BlockTab, type Stage } from '../../lib/state.svelte';
  import { css, diverging, robustScale, type RGB } from '../../lib/colors';
  import { distribution } from '../../lib/sampling';
  import { pct } from '../../lib/format';
  import { hideTip, showInspect } from '../../lib/inspect.svelte';

  const WINDOW = 12;
  let block = $state(app.block);
  let head = $state(0);
  let hot = $state(-1);
  let flow = $state(true);
  let playing = $state(false);
  let stage = $state(-1);
  let timer: ReturnType<typeof setInterval> | undefined;

  const tr = $derived(app.trace!);
  const cfg = $derived(app.cfg!);
  const T = $derived(tr.T);
  const s0 = $derived(Math.max(0, T - WINDOW));
  const win = $derived(Array.from({ length: T - s0 }, (_, k) => s0 + k));
  const n = $derived(win.length);
  const focus = $derived(Math.max(app.focus, s0));
  const key = $derived(tr.ids.join(','));

  const STAGES: { t: string; d: () => string; go: [Stage, BlockTab?] }[] = $derived([
    {
      t: 'Tokens → embeddings',
      go: ['embedding'],
      d: () =>
        `Each token id selects one row of the ${cfg.vocab_size.toLocaleString()} × ${cfg.d_model} embedding table. That vector is where the token starts in the residual stream.`,
    },
    {
      t: 'Query · Key · Value',
      go: ['block', 'attention'],
      d: () =>
        `After RMSNorm, every head projects each token three ways: the query asks, the key advertises, the value carries content. RoPE rotates Q and K by position.`,
    },
    {
      t: 'Masked self-attention',
      go: ['block', 'attention'],
      d: () =>
        `Every query is compared with every earlier key. A bigger dot means more attention. The empty upper triangle is the causal mask: no peeking at future tokens.`,
    },
    {
      t: 'Blend and add',
      go: ['block', 'attention'],
      d: () =>
        `Each token takes a weighted mix of the values it attended to. All ${cfg.n_heads} heads are merged by Wo and added back onto the token's vector.`,
    },
    {
      t: 'SwiGLU feed-forward',
      go: ['block', 'mlp'],
      d: () =>
        `Then each token on its own: widen ${cfg.d_model} → ${cfg.mlp_hidden} neurons, gate with SiLU, squeeze back to ${cfg.d_model}, add again. The squares are the real neurons for “${app.word(tr.ids[focus])}”.`,
    },
    {
      t: `× ${cfg.n_layers} blocks`,
      go: ['block', 'stream'],
      d: () =>
        `The same block structure repeats ${cfg.n_layers} times with different weights. The residual stream carries every token's vector through all of them.`,
    },
    {
      t: 'Next-token probabilities',
      go: ['output'],
      d: () =>
        `Only the last token's final vector is used: RMSNorm, then the tied LM head scores all ${cfg.vocab_size.toLocaleString()} tokens and softmax turns scores into probabilities. Top guess: “${app.word(dist.order[0])}”.`,
    },
  ]);

  // ---- geometry ----
  const cy = 340;
  const rowH = $derived(Math.min(38, 430 / n));
  const ty = (k: number) => cy - ((n - 1) * rowH) / 2 + k * rowH;
  const gs = $derived(Math.min(12, 120 / n));
  const gC = { k: cy - 160, q: cy, v: cy + 160 };
  const gy = (g: 'k' | 'q' | 'v', k: number) => gC[g] - ((n - 1) * gs) / 2 + k * gs;
  const cell = $derived(Math.min(26, 300 / n));
  const X = $derived.by(() => {
    const lab = 120,
      emb = 132,
      bl = 216,
      qkv = 320;
    const mat = qkv + 40 + 90;
    const matW = n * cell;
    const out = mat + matW + 80;
    const mlp = out + 40 + 70;
    const mlpW = 230;
    const bo = mlp + mlpW + 70;
    const br = bo + 40 + 36;
    const head = br + 64;
    const prob = head + 70 + 110;
    return { lab, emb, bl, qkv, mat, matW, out, mlp, mlpW, bo, br, head, prob, W: prob + 260 };
  });
  const my0 = $derived(cy - X.matW / 2);
  const ry = (k: number) => my0 + k * cell + cell / 2;
  const cx = (k: number) => X.mat + k * cell + cell / 2;
  const my = (k: number) => cy - 120 + (n > 1 ? k / (n - 1) : 0.5) * 240;
  const top = 66,
    bot = 626;
  const rw = $derived(Math.max(1.6, rowH * 0.2));

  // ---- data ----
  const P = $derived(app.palette);
  const L = $derived(Math.min(block, cfg.n_layers - 1));
  const embS = $derived(robustScale(tr.t('embed').data));
  const qkvS = $derived(robustScale(tr.L(L, 'q').data));
  const outS = $derived(robustScale(tr.L(L, 'head_out').data));
  const xS = $derived(robustScale(tr.L(L, 'x_out').data));
  const hid = $derived(tr.L(L, 'hidden').sub(focus));
  const hidS = $derived(robustScale(tr.L(L, 'hidden').data));
  const dist = $derived(distribution(tr.t('logits_last').data, app.sampling));
  const probs = $derived(dist.order.slice(0, 7));
  const pmax = $derived(dist.p[dist.order[0]]);
  const py = (k: number) => cy - (6 * 36) / 2 + k * 36;

  const curve = (x1: number, y1: number, x2: number, y2: number) => {
    const dx = (x2 - x1) * 0.5;
    return `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
  };
  const glyph = (v: Float32Array, dims: number, scale: number): RGB[] => Array.from(v.subarray(0, dims), (x) => diverging(P, x, scale));
  const label = (i: number) => {
    const s = app.tok(tr.ids[i]);
    return s.length > 11 ? s.slice(0, 10) + '…' : s;
  };
  const dim = (i: number) => hot >= 0 && hot !== i;

  // ---- play ----
  function playStep(k: number) {
    stage = k;
  }
  function togglePlay() {
    if (playing) return stop(false);
    playing = true;
    stage = stage < 0 || stage >= STAGES.length - 1 ? 0 : stage + 1;
    timer = setInterval(() => {
      if (stage >= STAGES.length - 1) stop(false);
      else stage++;
    }, 3600);
  }
  function stop(reset: boolean) {
    clearInterval(timer);
    playing = false;
    if (reset) stage = -1;
  }
  $effect(() => () => clearInterval(timer));
  const sv = (s: number) => (stage < 0 ? '' : s === stage ? 'cur' : s < stage ? 'on' : 'off');

  function open(s: number) {
    const [st, tab] = STAGES[s].go;
    stop(false);
    app.stage = st;
    if (tab) app.tab = tab;
    app.block = L;
    app.head = head;
    go('explore');
  }
  function key_(e: KeyboardEvent, s: number) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      open(s);
    }
  }
  function tipAtt(ev: MouseEvent, i: number, j: number) {
    const a = tr.L(L, 'probs').at(head, i, j);
    showInspect(
      {
        tag: `Attention · block ${L + 1} · head ${head + 1}`,
        title: `“${label(i)}” → “${label(j)}”`,
        body: `Query of “${label(i)}” · key of “${label(j)}”, scaled and softmaxed.`,
        calc: `score ${tr.L(L, 'scores').at(head, i, j).toFixed(3)} → ${pct(a)}`,
      },
      ev.clientX,
      ev.clientY,
    );
  }
</script>

<section class="card dcard" aria-labelledby="dg-title">
  <div class="dhead">
    <div>
      <h2 id="dg-title">The whole model, live</h2>
      <p class="faint">
        Hover a token to trace its path · click a part to open it in Explore{#if s0 > 0}
          · showing the last {WINDOW} of {T} tokens{/if}
      </p>
    </div>
    <div class="ctl">
      <div class="seg" role="group" aria-label="Block">
        {#each Array.from({ length: cfg.n_layers }, (_, k) => k) as k (k)}
          <button type="button" aria-pressed={k === L} aria-label="Block {k + 1}" onclick={() => (block = k)}>{k + 1}</button>
        {/each}
      </div>
      <div class="seg" role="group" aria-label="Head">
        {#each Array.from({ length: cfg.n_heads }, (_, k) => k) as k (k)}
          <button type="button" aria-pressed={k === head} aria-label="Head {k + 1}" onclick={() => (head = k)}>H{k + 1}</button>
        {/each}
      </div>
      <button class="btn accent" type="button" onclick={togglePlay} aria-pressed={playing}
        >{playing ? '❚❚ Pause' : '▶ Play forward pass'}</button
      >
      <button class="btn ghost" type="button" aria-pressed={flow} onclick={() => (flow = !flow)}>Flow</button>
    </div>
  </div>

  <div class="dwrap">
    {#key key}
      <svg
        class="dg"
        class:noflow={!flow}
        class:hl={hot >= 0}
        class:playing={stage >= 0}
        viewBox="0 0 {X.W} 650"
        role="group"
        aria-label="Diagram of the model processing {T} tokens"
        onmouseleave={() => (hot = -1)}
      >
        <defs>
          <linearGradient id="blkg" x1="0" y1="0" x2="1" y2="1"
            ><stop offset="0" stop-color="var(--surface)" /><stop offset="1" stop-color="var(--surface2)" /></linearGradient
          >
          <marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"
            ><path d="M0 0L8 4L0 8z" fill="var(--faint)" /></marker
          >
        </defs>
        <text class="ttl" x={X.lab} y="40" text-anchor="end">Tokens</text>
        <text class="ttl" x={X.emb + 22} y="40" text-anchor="middle">Embedding</text>
        <text class="ttl" x={X.prob + 100} y="40" text-anchor="middle">Probabilities</text>

        <!-- block stack (pseudo-3D) -->
        <g
          class="g {sv(5)}"
          role="button"
          tabindex="0"
          aria-label="Residual stream: open in Explore"
          onclick={() => open(5)}
          onkeydown={(e) => key_(e, 5)}
        >
          {#each Array.from({ length: cfg.n_layers - 1 }, (_, k) => cfg.n_layers - 1 - k) as k (k)}
            <rect class="stk" x={X.bl + k * 9} y={top - k * 9} width={X.br - X.bl} height={bot - top} rx="22" opacity={0.9 - k * 0.13} />
          {/each}
          <rect class="blk" x={X.bl} y={top} width={X.br - X.bl} height={bot - top} rx="22" />
          <text class="bt" x={X.bl + 22} y={top + 32}>Transformer block {L + 1} of {cfg.n_layers}</text>
          <text class="badge" x={X.br - 18} y={top + 32} text-anchor="end">×{cfg.n_layers}</text>
          <path class="res" d="M{X.bl + 6},{top + 54} L{X.br - 6},{top + 54}" marker-end="url(#ah)" />
          <text class="sm" x={(X.bl + X.br) / 2} y={top + 48} text-anchor="middle">residual stream: x + attention(x) + mlp(x)</text>
        </g>

        <!-- embeddings -->
        <g
          class="g {sv(0)}"
          role="button"
          tabindex="0"
          aria-label="Embeddings: open in Explore"
          onclick={() => open(0)}
          onkeydown={(e) => key_(e, 0)}
        >
          {#each win as i, k (i)}
            <g class="tk" class:dim={dim(i)} role="presentation" onmouseenter={() => (hot = i)}>
              <text
                class="lbl"
                class:on={i === focus}
                x={X.lab}
                y={ty(k) + 4.5}
                text-anchor="end"
                onclick={(e) => {
                  e.stopPropagation();
                  setFocus(i);
                }}
                role="presentation">{label(i)}</text
              >
              {#each glyph(tr.t('embed').sub(i), 8, embS) as c, d (d)}
                <rect
                  x={X.emb + d * 5.5}
                  y={ty(k) - Math.max(6, rowH * 0.55) / 2}
                  width="5.9"
                  height={Math.max(6, rowH * 0.55)}
                  fill={css(c)}
                />
              {/each}
            </g>
          {/each}
        </g>

        <!-- Q K V -->
        <g
          class="g {sv(1)}"
          role="button"
          tabindex="0"
          aria-label="Query, key and value: open in Explore"
          onclick={() => open(1)}
          onkeydown={(e) => key_(e, 1)}
        >
          {#each win as i, k (i)}
            {#each [['k', P.k], ['q', P.q], ['v', P.v]] as [g, c] (g)}
              <path
                class="rb tk"
                class:dim={dim(i)}
                d={curve(X.emb + 44, ty(k), X.qkv, gy(g as 'k', k))}
                stroke={css(c as RGB)}
                stroke-width={rw}
                pathLength="1"
              />
              <path
                class="fl tk"
                class:dim={dim(i)}
                d={curve(X.emb + 44, ty(k), X.qkv, gy(g as 'k', k))}
                stroke={css(c as RGB)}
                pathLength="1"
              />
            {/each}
          {/each}
          {#each [['k', 'Key', P.k, 'k'], ['q', 'Query', P.q, 'q'], ['v', 'Value', P.v, 'v']] as [g, name, c, t] (g)}
            <text class="gl" x={X.qkv + 20} y={gy(g as 'k', 0) - gs / 2 - 8} text-anchor="middle" style:fill={css(c as RGB)}>{name}</text>
            {#each win as i, k (i)}
              {#each glyph(tr.L(L, t as string).sub(head, i), 16, qkvS) as col, d (d)}
                <rect
                  class="tk"
                  class:dim={dim(i)}
                  x={X.qkv + d * 2.5}
                  y={gy(g as 'k', k) - Math.max(3, gs * 0.72) / 2}
                  width="2.7"
                  height={Math.max(3, gs * 0.72)}
                  fill={css(col)}
                />
              {/each}
            {/each}
          {/each}
        </g>

        <!-- attention matrix -->
        <g
          class="g {sv(2)}"
          role="button"
          tabindex="0"
          aria-label="Attention pattern: open in Explore"
          onclick={() => open(2)}
          onkeydown={(e) => key_(e, 2)}
        >
          {#each win as i, k (i)}
            <path
              class="rb tk thin"
              class:dim={dim(i)}
              d={curve(X.qkv + 40, gy('q', k), X.mat - 8, ry(k))}
              stroke={css(P.q)}
              stroke-width="1.3"
              pathLength="1"
            />
            <path
              class="rb tk thin"
              class:dim={dim(i)}
              d="M{X.qkv + 40},{gy('k', k)} C{cx(k)},{gy('k', k)} {cx(k)},{gy('k', k)} {cx(k)},{my0 - 8}"
              stroke={css(P.k)}
              stroke-width="1.3"
              pathLength="1"
            />
          {/each}
          <rect class="matbg" x={X.mat - 4} y={my0 - 4} width={X.matW + 8} height={X.matW + 8} rx="8" />
          {#each win as i, a (i)}
            {#each win as j, b (j)}
              {#if j > i}
                <circle class="msk" cx={cx(b)} cy={ry(a)} r="1.2" />
              {:else}
                {@const w = tr.L(L, 'probs').at(head, i, j)}
                <circle
                  class="dot tk"
                  class:dim={dim(i)}
                  role="presentation"
                  cx={cx(b)}
                  cy={ry(a)}
                  r={Math.max(1.6, cell * 0.46 * Math.sqrt(w))}
                  fill={css(P.attn)}
                  fill-opacity={0.3 + 0.7 * Math.sqrt(w)}
                  onmousemove={(e) => tipAtt(e, i, j)}
                  onmouseleave={hideTip}
                />
              {/if}
            {/each}
          {/each}
          <text class="sm" x={X.mat + X.matW / 2} y={my0 + X.matW + 26} text-anchor="middle"
            >attention · head {head + 1} · softmax(QKᵀ/√{cfg.d_model / cfg.n_heads})</text
          >
        </g>

        <!-- blend → out -->
        <g
          class="g {sv(3)}"
          role="button"
          tabindex="0"
          aria-label="Attention output: open in Explore"
          onclick={() => open(3)}
          onkeydown={(e) => key_(e, 3)}
        >
          {#each win as i, k (i)}
            <path
              class="rb tk thin"
              class:dim={dim(i)}
              d={curve(X.qkv + 40, gy('v', k), X.out, ry(k))}
              stroke={css(P.v)}
              stroke-width="1.1"
              pathLength="1"
            />
            <path
              class="rb tk"
              class:dim={dim(i)}
              d={curve(X.mat + X.matW + 6, ry(k), X.out, ry(k))}
              stroke={css(P.mix)}
              stroke-width={Math.max(1.6, cell * 0.3)}
              pathLength="1"
            />
            <path class="fl tk" class:dim={dim(i)} d={curve(X.mat + X.matW + 6, ry(k), X.out, ry(k))} stroke={css(P.mix)} pathLength="1" />
          {/each}
          <text class="gl" x={X.out + 20} y={my0 - 14} text-anchor="middle" style:fill={css(P.mix)}>Out</text>
          {#each win as i, k (i)}
            {#each glyph(tr.L(L, 'head_out').sub(head, i), 16, outS) as col, d (d)}
              <rect
                class="tk"
                class:dim={dim(i)}
                x={X.out + d * 2.5}
                y={ry(k) - Math.max(3, cell * 0.6) / 2}
                width="2.7"
                height={Math.max(3, cell * 0.6)}
                fill={css(col)}
              />
            {/each}
          {/each}
        </g>

        <!-- MLP -->
        <g
          class="g {sv(4)}"
          role="button"
          tabindex="0"
          aria-label="Feed-forward network: open in Explore"
          onclick={() => open(4)}
          onkeydown={(e) => key_(e, 4)}
        >
          {#each win as i, k (i)}
            <path
              class="rb tk"
              class:dim={dim(i)}
              d={curve(X.out + 40, ry(k), X.mlp, my(k))}
              stroke="var(--ffn)"
              stroke-width={rw}
              pathLength="1"
            />
            <path class="fl tk" class:dim={dim(i)} d={curve(X.out + 40, ry(k), X.mlp, my(k))} stroke="var(--ffn)" pathLength="1" />
            <path
              class="rb tk"
              class:dim={dim(i)}
              d={curve(X.mlp + X.mlpW, my(k), X.bo, ty(k))}
              stroke="var(--ffn)"
              stroke-width={rw}
              pathLength="1"
            />
          {/each}
          <rect class="mlp" x={X.mlp} y={cy - 150} width={X.mlpW} height="300" rx="16" />
          <text class="bt" x={X.mlp + X.mlpW / 2} y={cy - 122} text-anchor="middle">SwiGLU MLP</text>
          <text class="sm" x={X.mlp + X.mlpW / 2} y={cy - 104} text-anchor="middle">{cfg.d_model} → {cfg.mlp_hidden} → {cfg.d_model}</text>
          {#each Array.from(hid) as v, k (k)}
            <rect
              class="neu"
              x={X.mlp + 14 + (k % 32) * 6.3}
              y={cy - 88 + Math.floor(k / 32) * 9.4}
              width="5.2"
              height="7.6"
              rx="1.4"
              fill={css(diverging(P, v, hidS))}
            />
          {/each}
          <text class="sm" x={X.mlp + X.mlpW / 2} y={cy + 140} text-anchor="middle">{cfg.mlp_hidden} neurons for “{label(focus)}”</text>
          {#each win as i, k (i)}
            {#each glyph(tr.L(L, 'x_out').sub(i), 8, xS) as col, d (d)}
              <rect
                class="tk"
                class:dim={dim(i)}
                x={X.bo + d * 5}
                y={ty(k) - Math.max(6, rowH * 0.55) / 2}
                width="5.4"
                height={Math.max(6, rowH * 0.55)}
                fill={css(col)}
              />
            {/each}
          {/each}
        </g>

        <!-- output -->
        <g
          class="g {sv(6)}"
          role="button"
          tabindex="0"
          aria-label="Next-token probabilities: open in Explore"
          onclick={() => open(6)}
          onkeydown={(e) => key_(e, 6)}
        >
          {#each win as i, k (i)}
            {#if i === T - 1}
              <path
                class="rb tk"
                class:dim={dim(i)}
                d={curve(X.bo + 40, ty(k), X.head, cy)}
                stroke="var(--accent-bg)"
                stroke-width="4"
                pathLength="1"
              />
              <path class="fl tk" class:dim={dim(i)} d={curve(X.bo + 40, ty(k), X.head, cy)} stroke="var(--accent-bg)" pathLength="1" />
            {:else}
              <path
                class="rb tk stub"
                class:dim={dim(i)}
                d="M{X.bo + 40},{ty(k)} L{X.br + 10},{ty(k)}"
                stroke="var(--border2)"
                stroke-width={rw}
                pathLength="1"
              />
            {/if}
          {/each}
          <rect class="lmh" x={X.head} y={cy - 64} width="70" height="128" rx="12" />
          <text class="bt" x={X.head + 35} y={cy - 8} text-anchor="middle">LM</text>
          <text class="bt" x={X.head + 35} y={cy + 12} text-anchor="middle">head</text>
          <text class="sm" x={X.head + 35} y={cy + 34} text-anchor="middle">→ {cfg.vocab_size}</text>
          {#each probs as id, k (id)}
            {@const p = dist.p[id]}
            <path
              class="rb pr"
              d="M{X.head + 70},{cy} C{X.head + 125},{cy} {X.prob - 55},{py(k)} {X.prob},{py(k)}"
              stroke="var(--out)"
              stroke-width={Math.max(1, p * 34)}
              opacity={0.25 + p}
              pathLength="1"
            />
            <text class="pw" class:top={k === 0} x={X.prob + 8} y={py(k) + 5}>{app.tok(id).slice(0, 10)}</text>
            <rect class="pbg" x={X.prob + 96} y={py(k) - 7} width="110" height="14" rx="5" />
            <rect
              x={X.prob + 96}
              y={py(k) - 7}
              width={Math.max(2, (p / pmax) * 110)}
              height="14"
              rx="5"
              fill="var(--out)"
              opacity={k ? 0.55 : 1}
            />
            <text class="pp" x={X.prob + 214} y={py(k) + 4.5}>{pct(p)}</text>
          {/each}
        </g>
      </svg>
    {/key}
  </div>

  {#if stage >= 0}
    <div class="cap" aria-live="polite">
      <div>
        <p class="eyebrow">
          Step {stage + 1} of {STAGES.length}
          <span class="dots">
            {#each STAGES as s, i (s.t)}<button
                type="button"
                class:on={i === stage}
                class:done={i < stage}
                aria-label="Go to step {i + 1}: {s.t}"
                onclick={() => {
                  stop(false);
                  playStep(i);
                }}
              ></button>{/each}
          </span>
        </p>
        <h3>{STAGES[stage].t}</h3>
        <p>{STAGES[stage].d()}</p>
      </div>
      <div class="capbtn">
        <button
          class="btn ghost"
          type="button"
          disabled={stage === 0}
          onclick={() => {
            stop(false);
            playStep(stage - 1);
          }}
          aria-label="Previous step">←</button
        >
        <button class="btn ghost" type="button" onclick={() => open(stage)}>Open in Explore</button>
        {#if stage < STAGES.length - 1}
          <button
            class="btn"
            type="button"
            onclick={() => {
              stop(false);
              playStep(stage + 1);
            }}>Next →</button
          >
        {:else}
          <button class="btn" type="button" onclick={() => stop(true)}>Finish</button>
        {/if}
        <button class="btn ghost" type="button" onclick={() => stop(true)} aria-label="Close walkthrough">✕</button>
      </div>
    </div>
  {/if}

  <ul class="legend" aria-label="Colour legend">
    <li><i style:background="var(--q)"></i>query</li>
    <li><i style:background="var(--k)"></i>key</li>
    <li><i style:background="var(--v)"></i>value</li>
    <li><i style:background="var(--mix)"></i>attention output</li>
    <li><i style:background="var(--ffn)"></i>feed-forward</li>
    <li><i style:background="var(--out)"></i>probability</li>
  </ul>
</section>

<style>
  .dcard {
    position: relative;
    padding: 18px 20px 16px;
    background: color-mix(in oklab, var(--surface) 90%, transparent);
    backdrop-filter: blur(10px);
  }
  .dhead {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 4px;
  }
  .dhead h2 {
    margin: 0;
    font-size: 17px;
    letter-spacing: -0.01em;
  }
  .dhead p {
    margin: 2px 0 0;
    font-size: 13px;
  }
  .ctl {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
  }
  .dwrap {
    overflow-x: auto;
  }
  .dg {
    display: block;
    width: 100%;
    min-width: 980px;
    height: auto;
  }
  .dg text {
    fill: var(--muted);
    font-size: 13px;
    font-family: var(--sans);
  }
  .dg .ttl {
    font-size: 12px;
    font-weight: 650;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    fill: var(--faint);
  }
  .dg .lbl {
    font: 550 15px var(--mono);
    fill: var(--text);
    cursor: pointer;
    white-space: pre;
  }
  .dg .lbl.on {
    fill: var(--accent);
    font-weight: 750;
  }
  .dg .gl {
    font-weight: 700;
  }
  .dg .bt {
    font-size: 15px;
    font-weight: 700;
    fill: var(--text);
  }
  .dg .sm {
    font-size: 12px;
    fill: var(--faint);
  }
  .dg .badge {
    font: 700 15px var(--mono);
    fill: var(--accent);
  }
  .stk {
    fill: var(--surface2);
    stroke: var(--border);
  }
  .blk {
    fill: url(#blkg);
    stroke: var(--border2);
    filter: drop-shadow(0 10px 30px rgb(0 0 0 / 0.12));
  }
  .res {
    stroke: var(--border2);
    stroke-dasharray: 4 5;
    fill: none;
  }
  .matbg {
    fill: var(--surface2);
    stroke: var(--border);
  }
  .msk {
    fill: var(--border2);
  }
  .mlp {
    fill: color-mix(in oklab, var(--ffn) 7%, var(--surface));
    stroke: color-mix(in oklab, var(--ffn) 45%, transparent);
  }
  .lmh {
    fill: color-mix(in oklab, var(--out) 8%, var(--surface));
    stroke: color-mix(in oklab, var(--out) 45%, transparent);
  }
  .pw {
    font: 600 15px var(--mono);
    fill: var(--text) !important;
    white-space: pre;
  }
  .pw.top {
    fill: var(--out) !important;
    font-weight: 800;
  }
  .pbg {
    fill: var(--cell);
  }
  .pp {
    font: 500 13px var(--mono);
  }
  .rb {
    fill: none;
    stroke-linecap: round;
    opacity: 0.32;
    stroke-dasharray: 1;
    stroke-dashoffset: 1;
    animation: draw 0.9s cubic-bezier(0.4, 0, 0.2, 1) forwards;
  }
  .rb.thin {
    opacity: 0.4;
  }
  .rb.stub {
    opacity: 0.6;
  }
  .fl {
    fill: none;
    stroke-width: 2.2;
    stroke-linecap: round;
    stroke-dasharray: 0.006 0.05;
    opacity: 0.9;
    animation: flow 2.2s linear infinite;
    pointer-events: none;
  }
  .noflow .fl {
    display: none;
  }
  @keyframes flow {
    to {
      stroke-dashoffset: -0.056;
    }
  }
  @keyframes draw {
    to {
      stroke-dashoffset: 0;
    }
  }
  .g {
    transition: opacity 0.45s;
    cursor: pointer;
    outline: none;
  }
  .g:focus-visible > rect.blk,
  .g:focus-visible {
    outline: 2px solid var(--focus);
  }
  .hl .tk.dim {
    opacity: 0.07;
  }
  .hl .rb.tk:not(.dim) {
    opacity: 0.85;
  }
  .playing .g.off {
    opacity: 0.1;
  }
  .playing .g.on {
    opacity: 0.55;
  }
  .playing .g.cur {
    opacity: 1;
  }
  .playing .g.cur .neu {
    animation: fire 1.4s ease-in-out infinite;
  }
  @keyframes fire {
    50% {
      opacity: 0.3;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .fl {
      display: none;
    }
    .rb {
      animation: none;
      stroke-dasharray: none;
      stroke-dashoffset: 0;
    }
  }
  .cap {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 8px 24px;
    align-items: center;
    margin-top: 10px;
    background: var(--text);
    color: var(--bg);
    border-radius: 14px;
    padding: 14px 18px;
    animation: fade-in 0.3s both;
  }
  .cap .eyebrow {
    color: color-mix(in oklab, var(--bg) 70%, transparent);
    display: flex;
    gap: 12px;
    align-items: center;
  }
  .cap h3 {
    margin: 2px 0 4px;
    font-size: 16px;
  }
  .cap p {
    margin: 0;
    font-size: 14px;
    max-width: 780px;
  }
  .dots {
    display: inline-flex;
    gap: 4px;
  }
  .dots button {
    width: 20px;
    height: 6px;
    border: 0;
    padding: 0;
    border-radius: 9px;
    background: color-mix(in oklab, var(--bg) 25%, transparent);
  }
  .dots button.done {
    background: color-mix(in oklab, var(--bg) 60%, transparent);
  }
  .dots button.on {
    background: var(--accent-bg);
  }
  .capbtn {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }
  .capbtn .btn {
    background: var(--bg);
    color: var(--text);
    border-color: var(--bg);
  }
  .capbtn .btn.ghost {
    background: transparent;
    color: var(--bg);
    border-color: color-mix(in oklab, var(--bg) 35%, transparent);
  }
  .legend {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 14px;
    margin: 8px 0 0;
    padding: 0;
    list-style: none;
    font-size: 12.5px;
    color: var(--muted);
  }
  .legend i {
    display: inline-block;
    width: 11px;
    height: 11px;
    border-radius: 3px;
    margin-right: 6px;
    vertical-align: -1px;
  }
  @media (max-width: 860px) {
    .cap {
      grid-template-columns: 1fr;
    }
  }
</style>
