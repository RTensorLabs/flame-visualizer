import { expect, test, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

async function ready(page: Page, hash = '#/overview') {
  await page.goto('/' + hash);
  await expect(page.getByText(/traced in \d+ ms/)).toBeVisible({ timeout: 45_000 });
}

test('loads the real model and tokenizes with FLAME BPE', async ({ page }) => {
  await ready(page);
  const toks = page.getByRole('list', { name: /Tokens/ }).getByRole('button');
  await expect(toks.first()).toContainText('Once');
  await expect(toks.first()).toContainText('431'); // FLAME token id for "Once"
});

test('greedy continuation matches FLAME', async ({ page }) => {
  await ready(page);
  await page.getByRole('textbox', { name: 'Prompt' }).fill('Once upon a time');
  await page.getByRole('button', { name: 'Run' }).click();
  await expect(page.getByRole('list', { name: /Tokens/ }).getByRole('button')).toHaveCount(4);
  await page.getByRole('link', { name: 'Explore' }).click();
  await page.getByRole('button', { name: /^Output/ }).click();
  await page.getByLabel('Temperature').fill('0');
  for (let i = 0; i < 6; i++) {
    await page.getByRole('button', { name: 'Next token', exact: true }).click();
    await expect(page.getByRole('list', { name: /Tokens/ }).getByRole('button')).toHaveCount(5 + i);
  }
  // FLAME (native) greedy output for this prompt: ", there was a little girl"
  await expect(page.getByRole('textbox', { name: 'Prompt' })).toHaveValue('Once upon a time, there was a little girl');
});

for (const view of ['overview', 'learn', 'explore']) {
  for (const theme of ['light', 'dark']) {
    test(`no automatically detectable accessibility violations: ${view} (${theme})`, async ({ page }) => {
      await page.addInitScript((t) => localStorage.setItem('flame-theme', t), theme);
      await page.emulateMedia({ reducedMotion: 'reduce' }); // audit settled colours, not mid-fade frames
      await ready(page, `#/${view}`);
      const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze();
      expect(results.violations.map((v) => `${v.id}: ${v.nodes.length} × ${v.help}`)).toEqual([]);
    });
  }
}

test('keyboard: skip link, nav and chart inspection', async ({ page }) => {
  await ready(page, '#/explore');
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: 'Skip to content' })).toBeFocused();
  const chart = page.getByRole('application').first();
  await chart.focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.locator('.insp')).not.toContainText('Hover or focus anything coloured');
});
