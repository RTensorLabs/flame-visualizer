/** Inspector texts for the values shown across the app. */
import { app, type Inspect } from '../state.svelte';
import { pct } from '../format';

const f = (x: number) => (Number.isFinite(x) ? (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(3) : '—');
const w = (i: number) => `“${app.word(app.trace!.ids[i])}”`;

export function vecHover(kind: string, layer: number, i: number, d: number): Inspect {
  const tr = app.trace!;
  const L = (n: string) => tr.L(layer, n);
  const D = app.cfg!.d_model;
  switch (kind) {
    case 'embed':
      return {
        tag: 'Embedding',
        title: `${w(i)} · dimension ${d + 1} of ${D}`,
        body: 'Straight from the embedding table: depends only on the token, not on its context.',
        calc: `E[${tr.ids[i]}][${d}] = ${f(tr.t('embed').at(i, d))}`,
      };
    case 'mlp_norm':
      return {
        tag: `Block ${layer + 1} · RMSNorm`,
        title: `${w(i)} · dim ${d + 1}`,
        body: 'The residual stream after attention, divided by its root-mean-square and scaled by a learned gain.',
        calc: `${f(L('x_mid').at(i, d))} ÷ ${L('rms_mid').at(i).toFixed(3)} × g = ${f(L('mlp_norm').at(i, d))}`,
      };
    case 'gate':
      return {
        tag: `Block ${layer + 1} · gate`,
        title: `${w(i)} · neuron ${d + 1}`,
        body: 'SiLU of the gate projection: near zero closes this neuron, large values open it.',
        calc: `silu(${f(L('gate_pre').at(i, d))}) = ${f(L('gate').at(i, d))}`,
      };
    case 'up':
      return {
        tag: `Block ${layer + 1} · up`,
        title: `${w(i)} · neuron ${d + 1}`,
        body: 'The content this neuron would contribute.',
        calc: `x·W_up = ${f(L('up').at(i, d))}`,
      };
    case 'hidden':
      return {
        tag: `Block ${layer + 1} · SwiGLU neuron`,
        title: `${w(i)} · neuron ${d + 1} of ${app.cfg!.mlp_hidden}`,
        body: 'Gate × content: the neuron’s actual activation for this token.',
        calc: `${f(L('gate').at(i, d))} × ${f(L('up').at(i, d))} = ${f(L('hidden').at(i, d))}`,
      };
    case 'mlp_out':
      return {
        tag: `Block ${layer + 1} · down`,
        title: `${w(i)} · dim ${d + 1}`,
        body: `${app.cfg!.mlp_hidden} neurons squeezed back to ${D} numbers.`,
        calc: `h·W_down = ${f(L('mlp_out').at(i, d))}`,
      };
    case 'x_out':
      return {
        tag: `Block ${layer + 1} · output`,
        title: `${w(i)} · dim ${d + 1}`,
        body: 'Residual add: the stream keeps what it had and gains what the MLP found.',
        calc: `${f(L('x_mid').at(i, d))} + ${f(L('mlp_out').at(i, d))} = ${f(L('x_out').at(i, d))}`,
      };
    case 'final':
      return {
        tag: 'Final RMSNorm',
        title: `${w(i)} · dim ${d + 1}`,
        body: 'The vector the LM head compares with every token in the vocabulary.',
        calc: `${f(tr.t('final_norm').at(i, d))}`,
      };
    default:
      return { tag: kind, title: `${w(i)} · ${d}`, body: '' };
  }
}

export function attnHover(layer: number, head: number, i: number, j: number): Inspect | null {
  if (j > i)
    return {
      tag: 'Causal mask',
      title: `${w(i)} cannot see ${w(j)}`,
      body: 'Future tokens are masked out, so the model can never peek ahead.',
    };
  const tr = app.trace!;
  return {
    tag: `Attention · block ${layer + 1} · head ${head + 1}`,
    title: `${w(i)} → ${w(j)}`,
    body: `Query of ${w(i)} · key of ${w(j)}, divided by √${app.cfg!.d_model / app.cfg!.n_heads}, then softmax over everything ${w(i)} can see. That share of ${w(j)}’s value flows into ${w(i)}.`,
    calc: `score ${f(tr.L(layer, 'scores').at(head, i, j))} → ${pct(tr.L(layer, 'probs').at(head, i, j))}`,
  };
}
