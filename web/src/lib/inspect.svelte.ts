/** Hover/keyboard inspection: one shared tooltip + the Explore inspector panel. */
import { app, type Inspect } from './state.svelte';

export const tip = $state({ x: 0, y: 0, show: false, text: '' });

export function showInspect(info: Inspect, x: number, y: number) {
  app.inspect = info;
  tip.text = info.calc ?? info.title;
  tip.x = x;
  tip.y = y;
  tip.show = true;
}

export function hideTip() {
  tip.show = false;
}
