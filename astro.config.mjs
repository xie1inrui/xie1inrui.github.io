import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import tailwind from '@astrojs/tailwind';

export default defineConfig({
  site: process.env.SITE_URL ?? 'https://xie1inrui.github.io',
  base: process.env.BASE_PATH ?? '/',
  integrations: [mdx(), tailwind({ applyBaseStyles: false })],
  markdown: { shikiConfig: { theme: 'github-dark-dimmed', wrap: true } }
});
