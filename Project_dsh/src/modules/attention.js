import { attention as computeAttention } from '../algorithms/attention.js'
import { renderHeatmap, weightColor } from '../viz/heatmap.js'
import { renderInspector } from '../components/inspector.js'
import { downloadJson } from '../data/export.js'
import { saveRecord } from '../data/notebook.js'

const fmt = (x) => (Number.isFinite(x) ? x.toFixed(4) : '—')
const SCORE_LABEL = { dot: '点积', scaled: '缩放点积', additive: '加性' }

const DEFAULTS = {
  scoring: 'scaled',
  sourceTokens: ['猫', '吃', '鱼'],
  targetTokens: ['它'],
  K: [[1, 0], [0, 1], [1, 1]],
  V: [[2, 0], [0, 2], [1, 1]],
  Q: [[1, 0]],
  additive: { Wq: [[1, 0], [0, 1]], Wk: [[1, 0], [0, 1]], b: [0, 0], v: [1, 1] }
}

export function renderAttention(container) {
  const state = {
    scoring: DEFAULTS.scoring,
    sourceTokens: [...DEFAULTS.sourceTokens],
    targetTokens: [...DEFAULTS.targetTokens],
    K: DEFAULTS.K.map((r) => [...r]),
    V: DEFAULTS.V.map((r) => [...r]),
    Q: DEFAULTS.Q.map((r) => [...r]),
    additive: DEFAULTS.additive,
    mask: null,
    last: null,
    selected: null,
    baselineA: null,
    baselineB: null
  }

  const dk = () => state.Q[0].length
  const dv = () => state.V[0].length

  const root = document.createElement('div')
  root.className = 'module attention-module'
  container.appendChild(root)

  root.innerHTML = `
    <div class="module-head">
      <h1>M04 注意力机制</h1>
      <p class="module-desc">真实计算点积 / 缩放点积 / 加性注意力。点击权重格子追溯来源；参数改动实时更新。</p>
    </div>
    <div class="att-layout">
      <aside class="att-controls"></aside>
      <section class="att-main">
        <div class="att-editors"></div>
        <div class="att-heatmap-wrap"><div class="chart-title">注意力权重热力图 A = softmax(score)</div><div class="chart-hint">Shift + 点击 = 设置/取消掩码（标 ×）· 普通点击 = 选中查看</div></div>
        <div class="att-output"></div>
      </section>
      <aside class="att-inspector"></aside>
    </div>
    <section class="att-compare"></section>
  `

  const controlsEl = root.querySelector('.att-controls')
  const editorsEl = root.querySelector('.att-editors')
  const heatmapEl = root.querySelector('.att-heatmap-wrap')
  const outputEl = root.querySelector('.att-output')
  const inspectorEl = root.querySelector('.att-inspector')
  const compareEl = root.querySelector('.att-compare')

  function computeDot(Q, K) {
    return Q.map((qi) => K.map((kj) => qi.reduce((s, x, d) => s + x * kj[d], 0)))
  }
  function computeScaled(Q, K) {
    const scale = Math.sqrt(dk())
    return computeDot(Q, K).map((row) => row.map((s) => s / scale))
  }

  function recompute() {
    const scoreFn = { dot: computeDot, scaled: computeScaled }[state.scoring]
    const additive = state.scoring === 'additive' ? state.additive : null
    state.last = computeAttention(state.Q, state.K, state.V, { scoreFn, additive, mask: state.mask })
    renderResult()
    renderInspectorPane()
  }

  function renderResult() {
    if (!state.last) return
    const { weights, output } = state.last
    const heatmapHost = document.createElement('div')
    renderHeatmap(heatmapHost, {
      rowLabels: state.targetTokens,
      colLabels: state.sourceTokens,
      values: weights,
      format: fmt,
      color: weightColor,
      masked: state.mask,
      selected: state.selected,
      onCellClick: (i, j) => {
        state.selected = [i, j]
        renderInspectorPane()
      },
      onShiftClick: (i, j) => {
        if (!state.mask) state.mask = state.targetTokens.map(() => state.sourceTokens.map(() => true))
        state.mask[i][j] = !state.mask[i][j]
        recompute()
      }
    })
    const title = heatmapEl.querySelector('.chart-title')
    const hint = heatmapEl.querySelector('.chart-hint')
    heatmapEl.innerHTML = ''
    heatmapEl.appendChild(title)
    heatmapEl.appendChild(hint)
    heatmapEl.appendChild(heatmapHost)

    outputEl.innerHTML = ''
    const outTitle = document.createElement('div')
    outTitle.className = 'chart-title'
    outTitle.textContent = '加权输出 O = A·V 与每行权重分布'
    outputEl.appendChild(outTitle)
    weights.forEach((row, i) => {
      const barRow = document.createElement('div')
      barRow.className = 'weight-row'
      const label = document.createElement('span')
      label.className = 'weight-label'
      label.textContent = state.targetTokens[i]
      barRow.appendChild(label)
      const bars = document.createElement('div')
      bars.className = 'weight-bars'
      row.forEach((w, j) => {
        const bar = document.createElement('span')
        bar.className = 'weight-bar'
        bar.style.width = `${(w * 100).toFixed(1)}%`
        bar.title = `${state.targetTokens[i]} → ${state.sourceTokens[j]}: ${fmt(w)}`
        bars.appendChild(bar)
      })
      barRow.appendChild(bars)
      outputEl.appendChild(barRow)
    })
    const outRow = document.createElement('div')
    outRow.className = 'output-vector'
    const outLabel = document.createElement('span')
    outLabel.textContent = '输出向量 O'
    outRow.appendChild(outLabel)
    output.forEach((row, qi) => {
      const span = document.createElement('span')
      span.className = 'num'
      span.textContent = `${state.targetTokens[qi]}: [${row.map(fmt).join(', ')}]`
      outRow.appendChild(span)
    })
    outputEl.appendChild(outRow)
  }

  function renderInspectorPane() {
    if (!state.last || !state.selected) {
      inspectorEl.innerHTML = ''
      const hint = document.createElement('div')
      hint.className = 'inspector-hint'
      hint.textContent = '点击热力图任一格子查看计算来源。'
      inspectorEl.appendChild(hint)
      return
    }
    const [i, j] = state.selected
    const { scores, masked, weights } = state.last
    const qi = state.Q[i]
    const kj = state.K[j]
    const row = scores[i]
    const steps = []

    steps.push({
      title: '注意力权重',
      value: fmt(weights[i][j]),
      formula: 'A_i[j] = softmax(mask(score_i))_j = exp(score_j) / Σ_k exp(score_k)',
      inputs: row.map((s, k) => ({
        label: `score[${state.targetTokens[i]} → ${state.sourceTokens[k]}]`,
        value: fmt(s)
      })),
      meta: `形状 [${weights.length} × ${weights[0].length}]，位置 (${i}, ${j})`
    })

    if (state.mask && state.mask[i] && state.mask[i][j] === false) {
      steps.push({ title: '掩码', value: '屏蔽 (-∞)', formula: 'mask[i][j]=false → score 置 -∞，softmax 后权重为 0' })
    }

    const formulaBy = {
      dot: 'score(i,j) = q_i · k_j',
      scaled: 'score(i,j) = (q_i · k_j) / √d_k',
      additive: 'score(i,j) = vᵀ tanh(W_q q_i + W_k k_j + b)'
    }
    const inputs = []
    for (let d = 0; d < qi.length; d++) {
      inputs.push({ label: `q_i[${d}] × k_j[${d}]`, value: `${qi[d]} × ${kj[d]} = ${(qi[d] * kj[d]).toFixed(4)}` })
    }
    steps.push({
      title: '原始分数',
      value: fmt(masked[i][j] === -Infinity ? scores[i][j] : masked[i][j]),
      formula: formulaBy[state.scoring] || formulaBy.scaled,
      inputs,
      meta: `q_i = [${qi.join(', ')}]（目标「${state.targetTokens[i]}」），k_j = [${kj.join(', ')}]（源「${state.sourceTokens[j]}」）`
    })

    renderInspector(inspectorEl, steps)
  }

  function snapshot() {
    return {
      scoring: state.scoring,
      sourceTokens: [...state.sourceTokens],
      targetTokens: [...state.targetTokens],
      K: state.K.map((r) => [...r]),
      V: state.V.map((r) => [...r]),
      Q: state.Q.map((r) => [...r]),
      mask: state.mask ? state.mask.map((r) => [...r]) : null,
      weights: state.last ? state.last.weights.map((r) => [...r]) : null,
      output: state.last ? state.last.output.map((r) => [...r]) : null
    }
  }

  function configDiff(a, b) {
    const diffs = []
    if (a.scoring !== b.scoring) diffs.push(`打分方式：${SCORE_LABEL[a.scoring]} → ${SCORE_LABEL[b.scoring]}`)
    if (JSON.stringify(a.sourceTokens) !== JSON.stringify(b.sourceTokens)) diffs.push('源词集合不同')
    if (JSON.stringify(a.targetTokens) !== JSON.stringify(b.targetTokens)) diffs.push('查询词集合不同')
    if (JSON.stringify(a.K) !== JSON.stringify(b.K)) diffs.push('Key 向量不同')
    if (JSON.stringify(a.V) !== JSON.stringify(b.V)) diffs.push('Value 向量不同')
    if (JSON.stringify(a.Q) !== JSON.stringify(b.Q)) diffs.push('Query 向量不同')
    if (JSON.stringify(a.mask) !== JSON.stringify(b.mask)) diffs.push('掩码不同')
    return diffs
  }

  function renderCompare() {
    compareEl.innerHTML = ''
    const panel = document.createElement('div')
    panel.className = 'panel compare-panel'
    const head = document.createElement('div')
    head.className = 'chart-title'
    head.textContent = 'A/B 受控对照'
    panel.appendChild(head)

    const btnRow = document.createElement('div')
    btnRow.className = 'btn-row'
    const mkBtn = (label, fn) => {
      const b = document.createElement('button')
      b.type = 'button'
      b.className = 'btn'
      b.textContent = label
      b.addEventListener('click', fn)
      btnRow.appendChild(b)
    }
    mkBtn('设为基线 A', () => { state.baselineA = snapshot(); renderCompare() })
    mkBtn('设为对照 B', () => { state.baselineB = snapshot(); renderCompare() })
    mkBtn('清除对照', () => { state.baselineA = null; state.baselineB = null; renderCompare() })
    panel.appendChild(btnRow)

    if (state.baselineA) {
      const tag = document.createElement('div')
      tag.className = 'baseline-tag'
      tag.textContent = `基线 A：${SCORE_LABEL[state.baselineA.scoring]} · ${state.baselineA.targetTokens.length} 查询 × ${state.baselineA.sourceTokens.length} 源词`
      panel.appendChild(tag)
    }
    if (state.baselineB) {
      const tag = document.createElement('div')
      tag.className = 'baseline-tag'
      tag.textContent = `对照 B：${SCORE_LABEL[state.baselineB.scoring]} · ${state.baselineB.targetTokens.length} 查询 × ${state.baselineB.sourceTokens.length} 源词`
      panel.appendChild(tag)
    }

    if (state.baselineA && state.baselineB) {
      const diffs = configDiff(state.baselineA, state.baselineB)
      const diffTitle = document.createElement('div')
      diffTitle.className = 'chart-title'
      diffTitle.textContent = '配置差异'
      panel.appendChild(diffTitle)
      if (diffs.length === 0) {
        const p = document.createElement('p')
        p.className = 'dim'
        p.textContent = 'A 与 B 配置完全一致。'
        panel.appendChild(p)
      } else {
        const ul = document.createElement('ul')
        for (const d of diffs) {
          const li = document.createElement('li')
          li.textContent = d
          ul.appendChild(li)
        }
        panel.appendChild(ul)
      }
      const sameShape =
        state.baselineA.weights && state.baselineB.weights &&
        state.baselineA.weights.length === state.baselineB.weights.length &&
        state.baselineA.weights[0].length === state.baselineB.weights[0].length
      const wtTitle = document.createElement('div')
      wtTitle.className = 'chart-title'
      wtTitle.textContent = '权重差值 |A − B|'
      panel.appendChild(wtTitle)
      if (sameShape) {
        const diff = state.baselineA.weights.map((row, i) => row.map((w, j) => Math.abs(w - state.baselineB.weights[i][j])))
        const host = document.createElement('div')
        renderHeatmap(host, { rowLabels: state.baselineA.targetTokens, colLabels: state.baselineA.sourceTokens, values: diff, format: fmt, color: weightColor })
        panel.appendChild(host)
      } else {
        const p = document.createElement('p')
        p.className = 'dim'
        p.textContent = 'A 与 B 的 token 数量不一致，无法逐格对比。'
        panel.appendChild(p)
      }
    } else {
      const p = document.createElement('p')
      p.className = 'dim'
      p.textContent = '点击「设为基线 A」固定当前状态，修改后再「设为对照 B」对比。'
      panel.appendChild(p)
    }
    compareEl.appendChild(panel)
  }

  function exportExperiment() {
    const data = {
      kind: 'tensorscope.attention.experiment',
      version: 1,
      module: 'M04',
      createdAt: new Date().toISOString(),
      config: {
        scoring: state.scoring,
        sourceTokens: [...state.sourceTokens],
        targetTokens: [...state.targetTokens],
        Q: state.Q,
        K: state.K,
        V: state.V,
        additive: state.scoring === 'additive' ? state.additive : null,
        mask: state.mask
      },
      result: state.last ? { weights: state.last.weights, output: state.last.output } : null
    }
    downloadJson('attention-experiment.json', data)
  }

  function saveToNotebook() {
    saveRecord({
      module: 'M04 注意力',
      name: `注意力 · ${SCORE_LABEL[state.scoring]}`,
      config: {
        scoring: state.scoring,
        sourceTokens: [...state.sourceTokens],
        targetTokens: [...state.targetTokens],
        Q: state.Q,
        K: state.K,
        V: state.V,
        additive: state.scoring === 'additive' ? state.additive : null,
        mask: state.mask
      },
      result: state.last ? { weights: state.last.weights, output: state.last.output } : null
    })
    window.alert('已保存到实验记录')
  }

  function addSource() {
    const n = state.sourceTokens.length
    state.sourceTokens.push(`词${n + 1}`)
    state.K.push(new Array(dk()).fill(0))
    state.V.push(new Array(dv()).fill(0))
    if (state.mask) state.mask.forEach((row) => row.push(true))
    state.selected = null
    recompute()
    buildControls()
    renderEditors()
  }
  function removeSource(j) {
    if (state.sourceTokens.length <= 1) return
    state.sourceTokens.splice(j, 1)
    state.K.splice(j, 1)
    state.V.splice(j, 1)
    if (state.mask) state.mask.forEach((row) => row.splice(j, 1))
    state.selected = null
    recompute()
    buildControls()
    renderEditors()
  }
  function addTarget() {
    const n = state.targetTokens.length
    state.targetTokens.push(`查询${n + 1}`)
    state.Q.push(new Array(dk()).fill(0))
    if (state.mask) state.mask.push(new Array(state.sourceTokens.length).fill(true))
    state.selected = null
    recompute()
    buildControls()
    renderEditors()
  }
  function removeTarget(i) {
    if (state.targetTokens.length <= 1) return
    state.targetTokens.splice(i, 1)
    state.Q.splice(i, 1)
    if (state.mask) state.mask.splice(i, 1)
    state.selected = null
    recompute()
    buildControls()
    renderEditors()
  }

  function queryEditor(tokens, matrix) {
    const group = document.createElement('div')
    group.className = 'control-group'
    const label = document.createElement('div')
    label.className = 'control-label'
    label.textContent = '目标 Query 向量'
    group.appendChild(label)
    const table = document.createElement('table')
    table.className = 'vec-table'
    const thead = document.createElement('thead')
    const hr = document.createElement('tr')
    hr.innerHTML = '<th>token</th>' + matrix[0].map((_, d) => `<th>d${d + 1}</th>`).join('') + '<th></th>'
    thead.appendChild(hr)
    table.appendChild(thead)
    const tbody = document.createElement('tbody')
    matrix.forEach((row, i) => {
      const tr = document.createElement('tr')
      const tdTok = document.createElement('td')
      const tok = document.createElement('input')
      tok.type = 'text'
      tok.value = tokens[i]
      tok.addEventListener('input', () => { tokens[i] = tok.value; recompute() })
      tdTok.appendChild(tok)
      tr.appendChild(tdTok)
      row.forEach((v, d) => {
        const td = document.createElement('td')
        const num = document.createElement('input')
        num.type = 'number'; num.step = 'any'; num.value = v
        num.addEventListener('input', () => { matrix[i][d] = Number(num.value) || 0; recompute() })
        td.appendChild(num); tr.appendChild(td)
      })
      const td = document.createElement('td')
      const del = document.createElement('button')
      del.type = 'button'; del.className = 'icon-btn'; del.textContent = '✕'
      del.title = `删除 ${tokens[i]}`; del.disabled = matrix.length <= 1
      del.addEventListener('click', () => removeTarget(i))
      td.appendChild(del); tr.appendChild(td)
      tbody.appendChild(tr)
    })
    table.appendChild(tbody)
    group.appendChild(table)
    const addBtn = document.createElement('button')
    addBtn.type = 'button'; addBtn.className = 'btn'; addBtn.textContent = '＋ 添加查询'
    addBtn.addEventListener('click', addTarget)
    group.appendChild(addBtn)
    return group
  }

  function sourceEditor(tokens, K, V) {
    const group = document.createElement('div')
    group.className = 'control-group'
    const label = document.createElement('div')
    label.className = 'control-label'
    label.textContent = '源 Key / Value 向量'
    group.appendChild(label)
    const table = document.createElement('table')
    table.className = 'vec-table'
    const thead = document.createElement('thead')
    const hr = document.createElement('tr')
    hr.innerHTML = '<th>token</th>' +
      K[0].map((_, d) => `<th>k${d + 1}</th>`).join('') +
      V[0].map((_, d) => `<th>v${d + 1}</th>`).join('') + '<th></th>'
    thead.appendChild(hr)
    table.appendChild(thead)
    const tbody = document.createElement('tbody')
    K.forEach((row, i) => {
      const tr = document.createElement('tr')
      const tdTok = document.createElement('td')
      const tok = document.createElement('input')
      tok.type = 'text'
      tok.value = tokens[i]
      tok.addEventListener('input', () => { tokens[i] = tok.value; recompute() })
      tdTok.appendChild(tok)
      tr.appendChild(tdTok)
      K[i].forEach((v, d) => {
        const td = document.createElement('td')
        const num = document.createElement('input')
        num.type = 'number'; num.step = 'any'; num.value = v
        num.addEventListener('input', () => { K[i][d] = Number(num.value) || 0; recompute() })
        td.appendChild(num); tr.appendChild(td)
      })
      V[i].forEach((v, d) => {
        const td = document.createElement('td')
        const num = document.createElement('input')
        num.type = 'number'; num.step = 'any'; num.value = v
        num.addEventListener('input', () => { V[i][d] = Number(num.value) || 0; recompute() })
        td.appendChild(num); tr.appendChild(td)
      })
      const td = document.createElement('td')
      const del = document.createElement('button')
      del.type = 'button'; del.className = 'icon-btn'; del.textContent = '✕'
      del.title = `删除 ${tokens[i]}`; del.disabled = K.length <= 1
      del.addEventListener('click', () => removeSource(i))
      td.appendChild(del); tr.appendChild(td)
      tbody.appendChild(tr)
    })
    table.appendChild(tbody)
    group.appendChild(table)
    const addBtn = document.createElement('button')
    addBtn.type = 'button'; addBtn.className = 'btn'; addBtn.textContent = '＋ 添加源词'
    addBtn.addEventListener('click', addSource)
    group.appendChild(addBtn)
    return group
  }

  function maskEditor() {
    const group = document.createElement('div')
    group.className = 'control-group'
    const label = document.createElement('div')
    label.className = 'control-label'
    label.textContent = '掩码（行 = query，列 = key/value）'
    group.appendChild(label)
    const grid = document.createElement('div')
    grid.className = 'mask-grid'
    const head = document.createElement('div')
    head.className = 'mask-row mask-head'
    head.innerHTML = '<span>query\\key</span>' + state.sourceTokens.map((t) => `<span>${t}</span>`).join('')
    grid.appendChild(head)
    state.targetTokens.forEach((t, i) => {
      const row = document.createElement('div')
      row.className = 'mask-row'
      const rl = document.createElement('span')
      rl.textContent = t
      row.appendChild(rl)
      state.sourceTokens.forEach((_, j) => {
        const cell = document.createElement('button')
        cell.type = 'button'
        cell.className = 'mask-cell' + (state.mask && state.mask[i][j] ? ' masked' : '')
        cell.textContent = state.mask && state.mask[i][j] ? '✕' : ''
        cell.setAttribute('aria-label', `屏蔽 ${t} → ${state.sourceTokens[j]}`)
        cell.addEventListener('click', () => {
          if (!state.mask) state.mask = state.targetTokens.map(() => state.sourceTokens.map(() => false))
          state.mask[i][j] = !state.mask[i][j]
          cell.classList.toggle('masked', state.mask[i][j])
          cell.textContent = state.mask[i][j] ? '✕' : ''
          recompute()
        })
        row.appendChild(cell)
      })
      grid.appendChild(row)
    })
    group.appendChild(grid)
    return group
  }

  function renderEditors() {
    editorsEl.innerHTML = ''
    editorsEl.appendChild(queryEditor(state.targetTokens, state.Q))
    editorsEl.appendChild(sourceEditor(state.sourceTokens, state.K, state.V))
  }

  function buildControls() {
    controlsEl.innerHTML = ''

    const fieldset = document.createElement('fieldset')
    fieldset.className = 'control-group'
    const legend = document.createElement('legend')
    legend.textContent = '打分方式'
    fieldset.appendChild(legend)
    for (const [val, label] of Object.entries(SCORE_LABEL)) {
      const lab = document.createElement('label')
      lab.className = 'radio'
      const radio = document.createElement('input')
      radio.type = 'radio'
      radio.name = 'scoring'
      radio.value = val
      radio.checked = state.scoring === val
      radio.addEventListener('change', () => {
        state.scoring = val
        recompute()
      })
      lab.appendChild(radio)
      lab.appendChild(document.createTextNode(label))
      fieldset.appendChild(lab)
    }
    controlsEl.appendChild(fieldset)

    const exportBtn = document.createElement('button')
    exportBtn.type = 'button'
    exportBtn.className = 'btn'
    exportBtn.textContent = '导出实验 JSON'
    exportBtn.addEventListener('click', exportExperiment)
    controlsEl.appendChild(exportBtn)

    const saveBtn = document.createElement('button')
    saveBtn.type = 'button'
    saveBtn.className = 'btn'
    saveBtn.textContent = '保存到记录'
    saveBtn.addEventListener('click', saveToNotebook)
    controlsEl.appendChild(saveBtn)
  }

  buildControls()
  renderEditors()
  recompute()
  renderCompare()
}
