import { rnnStep, lstmStep, gruStep, rnnBackward } from '../algorithms/sequence.js'
import { createRng } from '../core/random.js'
import { renderHeatmap, divergingColor } from '../viz/heatmap.js'
import { renderInspector } from '../components/inspector.js'

const fmt = (x) => (Number.isFinite(x) ? x.toFixed(3) : '—')
const MODEL_LABELS = { rnn: 'RNN', lstm: 'LSTM', gru: 'GRU' }

function mat(rows, cols, fn) {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, fn))
}
function vec(n, fn) {
  return Array.from({ length: n }, fn)
}

export function renderSequence(container) {
  const state = {
    model: 'rnn',
    input: [0, 1, 2, 1],
    embedDim: 2,
    hiddenDim: 3,
    seed: 42,
    pointer: 0,
    computed: null
  }

  const root = document.createElement('div')
  root.className = 'module seq-module'
  container.appendChild(root)
  root.innerHTML = `
    <div class="module-head">
      <h1>M02 序列模型</h1>
      <p class="module-desc">RNN / LSTM / GRU 真实单步前向；观察隐状态热力图、门值与每个时间步的真实反向传播梯度范数。</p>
    </div>
    <div class="seq-layout">
      <aside class="seq-controls"></aside>
      <section class="seq-main"></section>
      <aside class="seq-inspector"></aside>
    </div>
  `

  const controlsEl = root.querySelector('.seq-controls')
  const mainEl = root.querySelector('.seq-main')
  const inspectorEl = root.querySelector('.seq-inspector')

  function makeEmbedding() {
    const rng = createRng(state.seed + 12345)
    const vocab = Math.max(1, Math.max(...state.input) + 1)
    return Array.from({ length: vocab }, () => Array.from({ length: state.embedDim }, () => (rng() * 2 - 1) * 0.5))
  }

  function makeWeights() {
    const rng = createRng(state.seed)
    const dx = state.embedDim
    const dh = state.hiddenDim
    const r = (scale) => (rng() * 2 - 1) * scale
    if (state.model === 'rnn') {
      return { Wxh: mat(dh, dx, () => r(0.4)), Whh: mat(dh, dh, () => r(0.4)), bh: vec(dh, () => r(0.1)) }
    } else if (state.model === 'lstm') {
      return { Wx: mat(4 * dh, dx, () => r(0.4)), Wh: mat(4 * dh, dh, () => r(0.4)), b: vec(4 * dh, () => r(0.1)) }
    }
    return { Wx: mat(3 * dh, dx, () => r(0.4)), Wh: mat(3 * dh, dh, () => r(0.4)), b: vec(6 * dh, () => r(0.1)) }
  }

  function forwardAll(embedded, model, W) {
    const dh = state.hiddenDim
    const h0 = new Array(dh).fill(0)
    const history = [{ step: 0, h: h0, c: null, gates: null }]
    let h = h0
    let c = new Array(dh).fill(0)
    for (let t = 0; t < embedded.length; t++) {
      const x = embedded[t]
      if (model === 'rnn') {
        h = rnnStep(x, h, W.Wxh, W.Whh, W.bh)
        history.push({ step: t + 1, h, c: null, gates: null })
      } else if (model === 'lstm') {
        const r = lstmStep(x, h, c, W.Wx, W.Wh, W.b)
        h = r.h
        c = r.c
        history.push({ step: t + 1, h: r.h, c: r.c, gates: { i: r.i, f: r.f, g: r.g, o: r.o } })
      } else {
        const r = gruStep(x, h, W.Wx, W.Wh, W.b)
        h = r.h
        history.push({ step: t + 1, h: r.h, c: null, gates: { r: r.r, z: r.z, n: r.n } })
      }
    }
    return history
  }

  function lossOf(history) {
    return history.reduce((s, step) => s + step.h.reduce((a, x) => a + x, 0), 0)
  }

  function computeNorms(embedded, model, W) {
    const dh = state.hiddenDim
    const h0 = new Array(dh).fill(0)
    if (model === 'rnn') {
      const v = new Array(dh).fill(1)
      const { dLdx } = rnnBackward(embedded, h0, W.Wxh, W.Whh, W.bh, v)
      return dLdx.map((g) => Math.sqrt(g.reduce((a, x) => a + x * x, 0)))
    }
    // LSTM/GRU：数值梯度（有限差分）
    const eps = 1e-4
    const norms = []
    for (let t = 0; t < embedded.length; t++) {
      const g = []
      for (let d = 0; d < embedded[0].length; d++) {
        const sp = embedded.map((x) => [...x])
        sp[t][d] += eps
        const sm = embedded.map((x) => [...x])
        sm[t][d] -= eps
        g.push((lossOf(forwardAll(sp, model, W)) - lossOf(forwardAll(sm, model, W))) / (2 * eps))
      }
      norms.push(Math.sqrt(g.reduce((a, x) => a + x * x, 0)))
    }
    return norms
  }

  function compute() {
    const E = makeEmbedding()
    const W = makeWeights()
    const embedded = state.input.map((id) => E[id % E.length])
    const history = forwardAll(embedded, state.model, W)
    const gradNorms = computeNorms(embedded, state.model, W)
    state.computed = { E, W, embedded, history, gradNorms }
    if (state.pointer > history.length - 1) state.pointer = history.length - 1
    renderMain()
    renderInspectorPane()
  }

  function renderMain() {
    mainEl.innerHTML = ''
    if (!state.computed) return
    const { embedded, history, gradNorms } = state.computed

    // 时间展开
    const tTitle = document.createElement('div')
    tTitle.className = 'chart-title'
    tTitle.textContent = '时间展开（点击某步查看；高亮 = 当前步）'
    mainEl.appendChild(tTitle)
    const unroll = document.createElement('div')
    unroll.className = 'seq-unroll'
    history.forEach((s) => {
      const cell = document.createElement('button')
      cell.type = 'button'
      cell.className = 'seq-step' + (s.step === state.pointer ? ' active' : '')
      cell.textContent = s.step === 0 ? 't0\nh₀' : `t${s.step}\n[${s.h.map(fmt).join(',')}]`
      cell.style.whiteSpace = 'pre-line'
      cell.addEventListener('click', () => {
        state.pointer = s.step
        renderMain()
        renderInspectorPane()
      })
      unroll.appendChild(cell)
    })
    mainEl.appendChild(unroll)

    // 状态热力图（与时间轴联动）
    const hmTitle = document.createElement('div')
    hmTitle.className = 'chart-title'
    hmTitle.textContent = '状态热力图（行 = 隐状态维，列 = 时间步，高亮 = 当前步）'
    mainEl.appendChild(hmTitle)
    const hmHost = document.createElement('div')
    const dh = state.hiddenDim
    const T = history.length - 1
    const hmValues = Array.from({ length: dh }, (_, i) => history.map((s) => s.h[i]))
    renderHeatmap(hmHost, {
      rowLabels: Array.from({ length: dh }, (_, i) => `h${i + 1}`),
      colLabels: Array.from({ length: T + 1 }, (_, j) => `t${j}`),
      values: hmValues,
      format: fmt,
      color: divergingColor,
      highlightCol: state.pointer
    })
    mainEl.appendChild(hmHost)

    // 反向传播梯度范数
    const gTitle = document.createElement('div')
    gTitle.className = 'chart-title'
    gTitle.textContent = `反向传播梯度范数（每个时间步）${state.model === 'rnn' ? '（解析 BPTT）' : '（数值梯度）'}`
    mainEl.appendChild(gTitle)
    const gBars = document.createElement('div')
    gBars.className = 'grad-bars'
    const maxNorm = Math.max(...gradNorms, 1e-9)
    gradNorms.forEach((n, t) => {
      const row = document.createElement('div')
      row.className = 'grad-row'
      const lab = document.createElement('span')
      lab.className = 'grad-label'
      lab.textContent = `t${t + 1}`
      const bar = document.createElement('div')
      bar.className = 'grad-track'
      const fill = document.createElement('span')
      fill.className = 'grad-fill'
      fill.style.width = `${(n / maxNorm) * 100}%`
      bar.appendChild(fill)
      const val = document.createElement('span')
      val.className = 'num'
      val.textContent = fmt(n)
      row.appendChild(lab)
      row.appendChild(bar)
      row.appendChild(val)
      gBars.appendChild(row)
    })
    mainEl.appendChild(gBars)

    // 当前步门值
    const cur = history[state.pointer]
    if (cur.gates) {
      const gateTitle = document.createElement('div')
      gateTitle.className = 'chart-title'
      gateTitle.textContent = `t${cur.step} 门值`
      mainEl.appendChild(gateTitle)
      const gateGrid = document.createElement('div')
      gateGrid.className = 'gate-grid'
      for (const [name, vals] of Object.entries(cur.gates)) {
        const g = document.createElement('div')
        g.className = 'gate-cell'
        g.innerHTML = `<span class="gate-name">${name}</span><span class="num">[${vals.map(fmt).join(', ')}]</span>`
        gateGrid.appendChild(g)
      }
      mainEl.appendChild(gateGrid)
    }
  }

  function renderInspectorPane() {
    if (!state.computed) {
      inspectorEl.innerHTML = ''
      const hint = document.createElement('div')
      hint.className = 'inspector-hint'
      hint.textContent = '点击时间步查看该步计算。'
      inspectorEl.appendChild(hint)
      return
    }
    const { E, embedded, history, gradNorms } = state.computed
    const cur = history[state.pointer]
    const steps = []

    // 嵌入
    const embSteps = embedded.map((x, i) => ({
      label: `token ${state.input[i]} → E[${state.input[i]}]`,
      value: `[${x.map(fmt).join(', ')}]`
    }))
    steps.push({
      title: '输入嵌入',
      value: `维度 ${state.embedDim}`,
      formula: `x_t = E[token_t]（词表 ${E.length} × ${state.embedDim}）`,
      inputs: embSteps
    })

    if (cur.step === 0) {
      steps.push({ title: '初始隐状态', value: `[${cur.h.map(fmt).join(', ')}]`, formula: 'h₀ = 0 向量' })
    } else {
      const t = cur.step - 1
      const formula = {
        rnn: 'h_t = tanh(W_hh h_{t-1} + W_xh x_t + b_h)',
        lstm: 'i,f,g,o = σ/σ/tanh/σ(...)；c_t = f⊙c + i⊙g；h_t = o⊙tanh(c_t)',
        gru: 'r,z = σ(...)；n = tanh(...)；h_t = (1-z)⊙n + z⊙h_{t-1}'
      }[state.model]
      steps.push({
        title: `第 ${t + 1} 步前向`,
        value: `[${cur.h.map(fmt).join(', ')}]`,
        formula,
        inputs: [
          { label: '输入嵌入 x_t', value: `[${embedded[t].map(fmt).join(', ')}]` },
          { label: '上一步隐状态', value: `[${history[t].h.map(fmt).join(', ')}]` }
        ]
      })
    }

    // 梯度
    if (cur.step >= 1) {
      steps.push({
        title: '该步梯度范数',
        value: fmt(gradNorms[cur.step - 1]),
        formula: `||dL/dx_t||（L = Σ_t 1·h_t）`,
        meta: state.model === 'rnn' ? '解析反向传播（BPTT）计算' : '数值梯度（有限差分 ε=1e-4）'
      })
    }

    renderInspector(inspectorEl, steps)
  }

  function buildControls() {
    controlsEl.innerHTML = ''

    // 模型选择（绿色选中描边）
    const modelGroup = document.createElement('div')
    modelGroup.className = 'control-group'
    const ml = document.createElement('div')
    ml.className = 'control-label'
    ml.textContent = '模型'
    modelGroup.appendChild(ml)
    const modelRow = document.createElement('div')
    modelRow.className = 'btn-row'
    const renderModelButtons = () => {
      modelRow.innerHTML = ''
      for (const [key, label] of Object.entries(MODEL_LABELS)) {
        const b = document.createElement('button')
        b.type = 'button'
        b.className = 'btn' + (state.model === key ? ' selected' : '')
        b.textContent = label
        b.addEventListener('click', () => {
          state.model = key
          state.pointer = 0
          renderModelButtons()
          compute()
        })
        modelRow.appendChild(b)
      }
    }
    renderModelButtons()
    modelGroup.appendChild(modelRow)
    controlsEl.appendChild(modelGroup)

    // 输入序列
    const inputGroup = document.createElement('div')
    inputGroup.className = 'control-group'
    const il = document.createElement('div')
    il.className = 'control-label'
    il.textContent = '输入序列（token id，逗号分隔）'
    inputGroup.appendChild(il)
    const input = document.createElement('input')
    input.type = 'text'
    input.value = state.input.join(', ')
    input.addEventListener('input', () => {
      state.input = input.value.split(/[,，\s]+/).map((x) => Math.round(Number(x))).filter((x) => Number.isFinite(x) && x >= 0)
      state.pointer = 0
      compute()
    })
    inputGroup.appendChild(input)
    controlsEl.appendChild(inputGroup)

    // 参数
    const params = [
      ['随机种子 seed', 'seed', 0, 999, 1],
      ['Embedding 维度', 'embedDim', 1, 8, 1],
      ['隐藏状态维度', 'hiddenDim', 1, 8, 1]
    ]
    for (const [label, key, min, max, step] of params) {
      const group = document.createElement('div')
      group.className = 'control-group'
      const l = document.createElement('div')
      l.className = 'control-label'
      l.textContent = `${label} = ${state[key]}`
      group.appendChild(l)
      const row = document.createElement('div')
      row.className = 'slider-row'
      const range = document.createElement('input')
      range.type = 'range'
      range.min = String(min)
      range.max = String(max)
      range.step = String(step)
      range.value = state[key]
      const num = document.createElement('input')
      num.type = 'number'
      num.step = String(step)
      num.min = String(min)
      num.max = String(max)
      num.value = state[key]
      const set = (v) => {
        state[key] = v
        range.value = v
        num.value = v
        l.textContent = `${label} = ${v}`
      }
      range.addEventListener('input', () => {
        set(Number(range.value))
        state.pointer = 0
        compute()
      })
      num.addEventListener('input', () => {
        const v = Number(num.value)
        if (Number.isFinite(v)) {
          set(v)
          state.pointer = 0
          compute()
        }
      })
      row.appendChild(range)
      row.appendChild(num)
      group.appendChild(row)
      controlsEl.appendChild(group)
    }

    // 步进控制
    const stepGroup = document.createElement('div')
    stepGroup.className = 'control-group'
    const sl = document.createElement('div')
    sl.className = 'control-label'
    sl.textContent = '步进控制'
    stepGroup.appendChild(sl)
    const sRow = document.createElement('div')
    sRow.className = 'btn-row'
    const mk = (label, fn) => {
      const b = document.createElement('button')
      b.type = 'button'
      b.className = 'btn'
      b.textContent = label
      b.addEventListener('click', fn)
      sRow.appendChild(b)
    }
    mk('前进一步', () => {
      if (state.pointer < state.computed.history.length - 1) state.pointer += 1
      renderMain()
      renderInspectorPane()
    })
    mk('后退', () => {
      if (state.pointer > 0) state.pointer -= 1
      renderMain()
      renderInspectorPane()
    })
    mk('重置', () => {
      state.pointer = 0
      renderMain()
      renderInspectorPane()
    })
    stepGroup.appendChild(sRow)
    controlsEl.appendChild(stepGroup)
  }

  buildControls()
  compute()
}
