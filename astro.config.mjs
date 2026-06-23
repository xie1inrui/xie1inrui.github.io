import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import tailwind from '@astrojs/tailwind';
const repositoryName = process.env.GITHUB_REPOSITORY?.split('/')[1];
const isGitHubPages = process.env.GITHUB_ACTIONS === 'true';
export default defineConfig({
  site: process.env.SITE_URL ?? 'https://xielinrui.github.io',
  base: process.env.BASE_PATH ?? (isGitHubPages && repositoryName ? `/${repositoryName}` : '/'),
  integrations: [mdx(), tailwind({ applyBaseStyles: false })],
  markdown: { shikiConfig: { theme: 'github-dark-dimmed', wrap: true } }
});
