import '../styles/main.css'
import { catalog } from './catalog.js'
import { renderHome } from './home.js'
import { renderAttention } from '../modules/attention.js'
import { renderOptimization } from '../modules/optimization.js'
import { renderData } from '../modules/data.js'
import { renderCnn } from '../modules/cnn.js'
import { renderSequence } from '../modules/sequence.js'
import { renderEmbeddings } from '../modules/embeddings.js'
import { renderComparison } from '../modules/comparison.js'
import { renderNotebook } from '../modules/notebook.js'

const SHARED = [{ id: 'data', title: '数据集' }, { id: 'notebook', title: '记录' }]
const MODULES = [
  { id: 'home', title: '实验首页' },
  ...catalog.map((m) => ({ id: m.id, title: m.title })),
  ...SHARED
]
const RENDERERS = {
  attention: renderAttention,
  optimization: renderOptimization,
  data: renderData,
  cnn: renderCnn,
  sequence: renderSequence,
  embeddings: renderEmbeddings,
  comparison: renderComparison,
  notebook: renderNotebook
}

function renderTopbar(activeId) {
  const header = document.createElement('header')
  header.className = 'topbar'
  const brand = document.createElement('a')
  brand.className = 'brand'
  brand.href = '#/'
  brand.textContent = 'TensorScope'
  header.appendChild(brand)
  const nav = document.createElement('nav')
  nav.className = 'tabs'
  nav.setAttribute('aria-label', '模块导航')
  for (const m of MODULES) {
    const a = document.createElement('a')
    a.href = '#/' + m.id
    a.textContent = m.title
    a.className = m.id === activeId ? 'tab active' : 'tab'
    nav.appendChild(a)
  }
  header.appendChild(nav)
  return header
}

function renderPlaceholder(content, id) {
  const mod = catalog.find((m) => m.id === id)
  const el = document.createElement('div')
  el.className = 'panel placeholder-panel'
  const h = document.createElement('h1')
  h.textContent = mod ? mod.title : '未知模块'
  const p = document.createElement('p')
  p.textContent = '模块骨架已就绪，算法实现将在后续阶段接入。'
  el.appendChild(h)
  el.appendChild(p)
  content.appendChild(el)
}

function route() {
  const app = document.getElementById('app')
  app.innerHTML = ''
  const id = location.hash.replace(/^#\/?/, '') || 'home'
  const activeId = MODULES.some((m) => m.id === id) ? id : 'home'
  app.appendChild(renderTopbar(activeId))
  const content = document.createElement('main')
  content.className = 'module-content'
  app.appendChild(content)
  if (activeId === 'home') renderHome(content)
  else if (RENDERERS[activeId]) RENDERERS[activeId](content)
  else renderPlaceholder(content, activeId)
}

window.addEventListener('hashchange', route)
// 兼容脚本被放在 <head> 且未加 defer 的情况：等 DOM 就绪再首渲染
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', route)
} else {
  route()
}
