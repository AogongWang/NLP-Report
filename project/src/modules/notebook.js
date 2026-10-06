import {
  listRecords,
  deleteRecord,
  clearRecords,
  exportRecordsText,
  importRecordsText
} from '../data/notebook.js'
import { downloadText } from '../data/export.js'

/** 实验记录页（I04）：查看、删除、导出、导入已保存的实验 */
export function renderNotebook(container) {
  const root = document.createElement('div')
  root.className = 'module notebook-module'
  container.appendChild(root)
  root.innerHTML = `
    <div class="module-head">
      <h1>实验记录</h1>
      <p class="module-desc">各模块「保存到记录」的实验：可查看配置与结果、删除、导出/导入（JSON）。</p>
    </div>
    <div class="nb-layout">
      <aside class="nb-controls"></aside>
      <section class="nb-main"></section>
    </div>
  `

  const controlsEl = root.querySelector('.nb-controls')
  const mainEl = root.querySelector('.nb-main')

  function render() {
    mainEl.innerHTML = ''
    const records = listRecords()
    if (records.length === 0) {
      const hint = document.createElement('div')
      hint.className = 'panel placeholder-panel'
      hint.innerHTML = '<h1>暂无记录</h1><p>在 M01~M06 模块里点「保存到记录」后，会出现在这里。</p>'
      mainEl.appendChild(hint)
      return
    }
    records.forEach((r) => {
      const card = document.createElement('div')
      card.className = 'nb-card'
      const head = document.createElement('div')
      head.className = 'nb-card-head'
      const title = document.createElement('span')
      title.className = 'cls-name'
      title.textContent = `${r.module || '未命名模块'} · ${r.name || ''}`
      const meta = document.createElement('span')
      meta.className = 'dim'
      meta.textContent = r.savedAt ? new Date(r.savedAt).toLocaleString() : ''
      head.appendChild(title)
      head.appendChild(meta)
      card.appendChild(head)

      const body = document.createElement('pre')
      body.className = 'nb-json'
      body.textContent = JSON.stringify({ config: r.config, result: r.result }, null, 2)
      card.appendChild(body)

      const del = document.createElement('button')
      del.type = 'button'
      del.className = 'btn'
      del.textContent = '删除'
      del.addEventListener('click', () => {
        deleteRecord(r.id)
        render()
      })
      card.appendChild(del)
      mainEl.appendChild(card)
    })
  }

  function buildControls() {
    controlsEl.innerHTML = ''
    const g = document.createElement('div')
    g.className = 'control-group'
    const l = document.createElement('div')
    l.className = 'control-label'
    l.textContent = '操作'
    g.appendChild(l)
    const row = document.createElement('div')
    row.className = 'btn-row'
    const mk = (label, fn) => {
      const b = document.createElement('button')
      b.type = 'button'
      b.className = 'btn'
      b.textContent = label
      b.addEventListener('click', fn)
      row.appendChild(b)
    }
    mk('导出全部', () => {
      downloadText('tensorscope-notebook.json', exportRecordsText())
    })
    mk('导入', () => {
      const inp = document.createElement('input')
      inp.type = 'file'
      inp.accept = '.json,application/json'
      inp.addEventListener('change', async () => {
        const f = inp.files[0]
        if (!f) return
        try {
          const n = importRecordsText(await f.text())
          render()
          window.alert(`已导入 ${n} 条记录`)
        } catch (e) {
          window.alert('导入失败：' + e.message)
        }
      })
      inp.click()
    })
    mk('清空', () => {
      if (window.confirm('确定清空全部记录？')) {
        clearRecords()
        render()
      }
    })
    g.appendChild(row)
    controlsEl.appendChild(g)
    const note = document.createElement('p')
    note.className = 'dim'
    note.textContent = '记录保存在浏览器本地（localStorage），离线可用；导出可得到 JSON 文件备份。'
    controlsEl.appendChild(note)
  }

  buildControls()
  render()
}
