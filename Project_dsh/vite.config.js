import { defineConfig } from 'vite'

/**
 * 离线交付后处理：
 * file:// 下浏览器禁止 ES module（<script type="module"> 会触发 CORS）。
 * app.js 已按 IIFE 打包，这里把构建产物里的 type="module" 去掉，
 * 并移除 crossorigin 属性，使 release/index.html 可双击直接打开。
 */
function offlineHtml() {
  return {
    name: 'offline-html',
    enforce: 'post',
    transformIndexHtml(html) {
      return html
        .replace(
          /<script type="module"[^>]*src="([^"]+)"[^>]*><\/script>/g,
          '<script defer src="$1"></script>'
        )
        .replace(/\s+crossorigin\b/g, '')
    }
  }
}

export default defineConfig({
  base: './',
  plugins: [offlineHtml()],
  build: {
    outDir: 'release',
    emptyOutDir: true,
    target: 'es2018',
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        format: 'iife',
        entryFileNames: 'app.js',
        assetFileNames: 'assets/[name][extname]'
      }
    }
  }
})
