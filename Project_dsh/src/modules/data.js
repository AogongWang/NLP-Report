import { parseCSV, parseJSONL, parseTXT, analyzeSamples, validateSamples } from '../data/datasets.js'

// 内置示例（人工整理的中文短文本二分类，仅作演示）
const BUILTIN = {
  sentiment: {
    name: '内置：情感二分类（示例）',
    samples: [
      { text: '这部电影太棒了，演员演技在线', label: '正面' },
      { text: '剧情拖沓，看了想睡觉', label: '负面' },
      { text: '画面精美，音乐也很好听', label: '正面' },
      { text: '浪费了我两个小时', label: '负面' },
      { text: '整体不错，值得一看', label: '正面' },
      { text: '剧情混乱，完全看不懂', label: '负面' },
      { text: '主角演技精湛，情节紧凑', label: '正面' },
      { text: '太无聊了，中途就退场了', label: '负面' },
      { text: '特效震撼，值回票价', label: '正面' },
      { text: '编剧脑子有坑，结局莫名其妙', label: '负面' }
    ]
  }
}

export function renderData(container) {
  const state = {
    dataset: null,
    preview: [],
    stats: null,
    errors: []
  }

  const root = document.createElement('div')
  root.className = 'module data-module'
  container.appendChild(root)
  root.innerHTML = `
    <div class="module-head">
      <h1>数据集管理</h1>
      <p class="module-desc">导入 TXT / CSV / JSONL 数据，预览并检查标签分布、长度分布与重复项。本地处理，不上传。</p>
    </div>
    <div class="data-layout">
      <aside class="data-controls"></aside>
      <section class="data-main"></section>
    </div>
  `

  const controlsEl = root.querySelector('.data-controls')
  const mainEl = root.querySelector('.data-main')

  function setDataset(name, samples) {
    state.dataset = { name, samples }
    state.errors = validateSamples(samples).errors
    state.stats = analyzeSamples(samples)
    state.preview = samples.slice(0, 20)
    renderMain()
  }

  function renderMain() {
    mainEl.innerHTML = ''
    if (!state.dataset) {
      const hint = document.createElement('div')
      hint.className = 'panel placeholder-panel'
      hint.innerHTML = '<h1>请导入数据</h1><p>选择文件或使用内置示例。</p>'
      mainEl.appendChild(hint)
      return
    }
    const wrap = document.createElement('div')
    wrap.className = 'data-result'

    // 概览
    const head = document.createElement('div')
    head.className = 'chart-title'
    head.textContent = `${state.dataset.name}（${state.stats.count} 条样本）`
    wrap.appendChild(head)

    if (state.errors.length) {
      const err = document.createElement('div')
      err.className = 'error-box'
      err.textContent = `校验：${state.errors.length} 条异常（第 ${state.errors.map((e) => e.index + 1).slice(0, 5).join(', ')} …）`
      wrap.appendChild(err)
    }

    // 标签分布
    const labelTitle = document.createElement('div')
    labelTitle.className = 'chart-title'
    labelTitle.textContent = '标签分布'
    wrap.appendChild(labelTitle)
    const labels = Object.entries(state.stats.labelDistribution)
    const labelList = document.createElement('ul')
    labelList.className = 'stat-list'
    labels.forEach(([label, count]) => {
      const li = document.createElement('li')
      li.innerHTML = `<span>${label}</span><span class="num">${count}</span>`
      labelList.appendChild(li)
    })
    wrap.appendChild(labelList)

    // 长度统计
    const lenTitle = document.createElement('div')
    lenTitle.className = 'chart-title'
    lenTitle.textContent = '长度统计（字符数）'
    wrap.appendChild(lenTitle)
    const lenRow = document.createElement('div')
    lenRow.className = 'dim'
    lenRow.textContent = `最短 ${state.stats.lengthStats.min} · 最长 ${state.stats.lengthStats.max} · 平均 ${state.stats.lengthStats.avg}`
    wrap.appendChild(lenRow)

    // 重复
    const dupTitle = document.createElement('div')
    dupTitle.className = 'chart-title'
    dupTitle.textContent = '重复检查'
    wrap.appendChild(dupTitle)
    const dupRow = document.createElement('div')
    dupRow.className = 'dim'
    dupRow.textContent = state.stats.duplicatePairs.length > 0 ? `发现 ${state.stats.duplicatePairs.length} 对重复文本` : '未发现重复文本'
    wrap.appendChild(dupRow)

    // 预览表
    const pvTitle = document.createElement('div')
    pvTitle.className = 'chart-title'
    pvTitle.textContent = `预览（前 ${state.preview.length} 条）`
    wrap.appendChild(pvTitle)
    const table = document.createElement('table')
    table.className = 'preview-table'
    const thead = document.createElement('thead')
    const hr = document.createElement('tr')
    for (const key of Object.keys(state.preview[0] || { text: '', label: '' })) {
      const th = document.createElement('th')
      th.textContent = key
      hr.appendChild(th)
    }
    thead.appendChild(hr)
    table.appendChild(thead)
    const tbody = document.createElement('tbody')
    state.preview.forEach((s) => {
      const tr = document.createElement('tr')
      for (const key of Object.keys(s)) {
        const td = document.createElement('td')
        td.textContent = s[key]
        tr.appendChild(td)
      }
      tbody.appendChild(tr)
    })
    table.appendChild(tbody)
    wrap.appendChild(table)

    mainEl.appendChild(wrap)
  }

  function buildControls() {
    controlsEl.innerHTML = ''

    // 内置示例
    const builtinGroup = document.createElement('div')
    builtinGroup.className = 'control-group'
    const bl = document.createElement('div')
    bl.className = 'control-label'
    bl.textContent = '内置示例'
    builtinGroup.appendChild(bl)
    for (const [key, ds] of Object.entries(BUILTIN)) {
      const b = document.createElement('button')
      b.type = 'button'
      b.className = 'btn'
      b.textContent = ds.name
      b.addEventListener('click', () => setDataset(ds.name, ds.samples.map((s) => ({ ...s }))))
      builtinGroup.appendChild(b)
    }
    controlsEl.appendChild(builtinGroup)

    // 文件导入
    const fileGroup = document.createElement('div')
    fileGroup.className = 'control-group'
    const fl = document.createElement('div')
    fl.className = 'control-label'
    fl.textContent = '导入文件（TXT / CSV / JSONL）'
    fileGroup.appendChild(fl)
    const fileInput = document.createElement('input')
    fileInput.type = 'file'
    fileInput.accept = '.txt,.csv,.jsonl,.json'
    fileInput.addEventListener('change', async () => {
      const file = fileInput.files[0]
      if (!file) return
      const text = await file.text()
      const name = file.name
      const lower = name.toLowerCase()
      let samples = []
      if (lower.endsWith('.jsonl') || lower.endsWith('.json')) {
        const { records, errors } = parseJSONL(text)
        if (errors.length) {
          window.alert(`JSONL 解析错误 ${errors.length} 条，第一条在第 ${errors[0].line} 行：${errors[0].message}`)
        }
        samples = records.map((r) => ({ text: String(r.text ?? r.content ?? ''), label: String(r.label ?? '') }))
      } else if (lower.endsWith('.csv')) {
        const rows = parseCSV(text)
        if (rows.length < 2) {
          window.alert('CSV 至少需要表头 + 一行数据')
          return
        }
        const header = rows[0]
        const textIdx = header.findIndex((h) => /text|content|句子|文本|内容|review/i.test(h))
        const labelIdx = header.findIndex((h) => /label|class|标签|类别|情感|y/i.test(h))
        if (textIdx < 0) {
          window.alert('未找到文本列（表头应包含 text/文本/内容 等）')
          return
        }
        samples = rows.slice(1).map((r) => ({
          text: String(r[textIdx] ?? ''),
          label: labelIdx >= 0 ? String(r[labelIdx] ?? '') : ''
        }))
      } else {
        const lines = parseTXT(text)
        samples = lines.map((l) => ({ text: l, label: '' }))
      }
      setDataset(name, samples)
    })
    fileGroup.appendChild(fileInput)
    controlsEl.appendChild(fileGroup)

    const note = document.createElement('p')
    note.className = 'dim'
    note.textContent = 'CSV：首行为表头，自动识别文本列（text/文本/内容）与标签列（label/标签/类别）。JSONL：每行 {"text":…,"label":…}。TXT：一行一句（无标签）。'
    controlsEl.appendChild(note)
  }

  buildControls()
  renderMain()
}
