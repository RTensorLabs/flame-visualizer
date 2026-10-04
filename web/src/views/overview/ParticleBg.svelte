<!-- Decorative hero background: tokens drifting along curved "residual streams". -->
<script lang="ts">
  import { app } from '../../lib/state.svelte';

  let cv: HTMLCanvasElement | undefined = $state();

  $effect(() => {
    if (!cv) return;
    const canvas = cv;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dark = app.theme === 'dark';
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const COLORS = ['#f97316', '#fb923c', '#ef4444', '#3b82f6', '#14b8a6', '#a78bfa'];
    type Lane = { y: number; a: number; f: number; ph: number; sp: number; c: string };
    type P = { l: Lane; x: number; v: number; r: number; c: string; o: number };
    let W = 0,
      H = 0,
      lanes: Lane[] = [],
      parts: P[] = [],
      raf = 0,
      visible = true;
    const t0 = performance.now();

    const spawn = (anywhere: boolean): P => {
      const l = lanes[Math.floor(Math.random() * lanes.length)];
      return {
        l,
        x: anywhere ? Math.random() * W : -20,
        v: 0.35 + Math.random() * 1.1,
        r: 0.8 + Math.random() * 1.8,
        c: Math.random() < 0.7 ? l.c : COLORS[Math.floor(Math.random() * COLORS.length)],
        o: Math.random() * 12 - 6,
      };
    };
    const yAt = (l: Lane, x: number, t: number) => l.y + Math.sin(x * l.f + l.ph + t * l.sp) * l.a;

    function resize() {
      const dpr = Math.min(devicePixelRatio || 1, 2);
      W = canvas.clientWidth;
      H = canvas.clientHeight;
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
      const nl = Math.max(7, Math.round(H / 70));
      lanes = Array.from({ length: nl }, (_, k) => ({
        y: (H * (k + 0.5)) / nl,
        a: 18 + Math.random() * 34,
        f: 0.0018 + Math.random() * 0.0022,
        ph: Math.random() * 6.28,
        sp: 0.00025 + Math.random() * 0.0004,
        c: COLORS[k % COLORS.length],
      }));
      parts = Array.from({ length: Math.round(W / 16) }, () => spawn(true));
    }

    function frame(now: number) {
      const t = now - t0;
      ctx!.clearRect(0, 0, W, H);
      ctx!.lineWidth = 1;
      for (const l of lanes) {
        ctx!.beginPath();
        for (let x = 0; x <= W; x += 12) {
          const y = yAt(l, x, t);
          if (x) ctx!.lineTo(x, y);
          else ctx!.moveTo(x, y);
        }
        ctx!.strokeStyle = dark ? 'rgba(255,255,255,.045)' : 'rgba(20,20,30,.05)';
        ctx!.stroke();
      }
      ctx!.globalCompositeOperation = dark ? 'lighter' : 'source-over';
      for (const p of parts) {
        if (!reduce) p.x += p.v;
        if (p.x > W + 30) Object.assign(p, spawn(false));
        const y = yAt(p.l, p.x, t) + p.o;
        const tail = 14 + p.v * 22;
        const g = ctx!.createLinearGradient(p.x - tail, y, p.x, y);
        g.addColorStop(0, p.c + '00');
        g.addColorStop(1, p.c + (dark ? 'cc' : '88'));
        ctx!.strokeStyle = g;
        ctx!.lineWidth = p.r;
        ctx!.beginPath();
        ctx!.moveTo(p.x - tail, yAt(p.l, p.x - tail, t) + p.o);
        ctx!.lineTo(p.x, y);
        ctx!.stroke();
        ctx!.fillStyle = p.c + (dark ? 'ff' : 'aa');
        ctx!.beginPath();
        ctx!.arc(p.x, y, p.r * 0.9, 0, 6.29);
        ctx!.fill();
      }
      ctx!.globalCompositeOperation = 'source-over';
      if (!reduce && visible && !document.hidden) raf = requestAnimationFrame(frame);
    }
    const start = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(frame);
    };
    const io = new IntersectionObserver(([e]) => {
      visible = e.isIntersecting;
      if (visible) start();
    });
    const onVis = () => !document.hidden && start();
    const onResize = () => {
      resize();
      start();
    };
    resize();
    start();
    io.observe(canvas);
    document.addEventListener('visibilitychange', onVis);
    addEventListener('resize', onResize);
    return () => {
      cancelAnimationFrame(raf);
      io.disconnect();
      document.removeEventListener('visibilitychange', onVis);
      removeEventListener('resize', onResize);
    };
  });
</script>

<canvas bind:this={cv} aria-hidden="true"></canvas>

<style>
  canvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 900px;
    z-index: 0;
    pointer-events: none;
    mask-image: linear-gradient(to bottom, #000 35%, transparent);
  }
</style>
