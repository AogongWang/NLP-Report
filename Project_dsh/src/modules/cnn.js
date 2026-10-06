import { conv1d, reluArr, maxPool } from '../algorithms/cnn.js'
import { softmax } from '../core/math.js'
import { hashString, createRng } from '../core/random.js'
import { renderHeatmap, divergingColor } from '../viz/heatmap.js'
import { renderInspector } from '../components/inspector.js'

const fmt = (x) => (Number.isFinite(x) ? x.toFixed(3) : '—')

function charEmbedding(ch, dim) {
  const rng = createRng(hashString(ch))
  return Array.from({ length: dim }, () => Math.round((rng() * 2 - 1) * 100) / 100)
}

/** 生成互不相同的默认卷积核（避免两个核卷积值相同导致分类概率恒为 0.5） */
function defaultKernel(width, dim, variant) {
  return Array.from({ length: width }, (_, w) =>
    Array.from({ length: dim }, (_, d) => {
      if (variant === 0) return w % dim === d ? 1 : 0
      return w === width - 1 ? 1 : w % dim === d ? 1 : 0
    })
  )
}

export function renderCnn(container) {
  const state = {
    text: '这个电影不错',
    embedDim: 2,
    kernels: [
      { width: 2, values: defaultKernel(2, 2, 0) },
      { width: 3, values: defaultKernel(3, 2, 1) }
    ],
    fcWeights: [[1, 0.5], [0.5, 1]],
    fcBias: [0, 0],
    classNames: ['正面', '负面'],
    result: null,
    stale: false,
    selectedWindow: -1,
    animRunning: false,
    animTimer: null
  }

  const root = document.createElement('div')
  root.className = 'module cnn-module'
  container.appendChild(root)
  root.innerHTML = `
    <div class="module-head">
      <h1>M03 Text-CNN</h1>
      <p class="module-desc">文本 → Embedding → 多窗口卷积 → ReLU → max-over-time 池化 → 分类概率。动画在词向量矩阵上滑动高亮窗口。</p>
    </div>
    <div class="cnn-layout">
      <aside class="cnn-controls"></aside>
      <section class="cnn-main"></section>
      <aside class="cnn-inspector"></aside>
    </div>
  `

  const controlsEl = root.querySelector('.cnn-controls')
  const mainEl = root.querySelector('.cnn-main')
  const inspectorEl = root.querySelector('.cnn-inspector')

  function markStale() {
    state.stale = true
    stopAnimation()
    renderControlsStatus()
  }

  function stopAnimation() {
    state.animRunning = false
    if (state.animTimer) {
      clearInterval(state.animTimer)
      state.animTimer = null
    }
  }

  function compute() {
    const tokens = [...state.text.trim()]
    if (tokens.length === 0) return
    const vocab = [...new Set(tokens)]
    const idOf = new Map(vocab.map((ch, i) => [ch, i]))
    const embedding = vocab.map((ch) => charEmbedding(ch, state.embedDim))
    const tokenIds = tokens.map((ch) => idOf.get(ch))
    const emb = tokenIds.map((id) => embedding[id])

    const kernelResults = state.kernels.map((k) => {
      const raw = conv1d(emb, k.values)
      const relu = reluArr(raw)
      const pooled = maxPool(relu)
      const pooledWindow = pooled > 0 ? relu.indexOf(pooled) : -1
      return { width: k.width, raw, relu, pooled, pooledWindow }
    })
    const pooledVec = kernelResults.map((r) => r.pooled)
    const scores = state.fcWeights.map((row, c) => row.reduce((s, w, j) => s + w * pooledVec[j], 0) + (state.fcBias[c] || 0))
    const probs = softmax(scores)

    state.result = { tokens, vocab, idOf, embedding, tokenIds, emb, kernelResults, pooledVec, scores, probs }
    state.stale = false
    state.selectedWindow = -1
    stopAnimation()
    renderControlsStatus()
    renderMain()
    renderInspectorPane()
  }

  function playAnimation() {
    if (!state.result) return
    stopAnimation()
    state.animRunning = true
    const kr = state.result.kernelResults[0]
    let wi = -1
    const step = () => {
      if (wi < kr.raw.length - 1) {
        wi += 1
        state.selectedWindow = wi
      } else {
        state.selectedWindow = kr.pooledWindow
        state.animRunning = false
        clearInterval(state.animTimer)
        state.animTimer = null
      }
      renderMain()
      renderInspectorPane()
    }
    step()
    state.animTimer = setInterval(step, 650)
  }

  function renderMain() {
    mainEl.innerHTML = ''
    if (!state.result) {
      const hint = document.createElement('div')
      hint.className = 'inspector-hint'
      hint.textContent = '点击「重新计算」查看结果。'
      mainEl.appendChild(hint)
      return
    }
    const { tokens, emb, kernelResults, scores, probs } = state.result
    const k0 = state.kernels[0]

    // 词向量矩阵（按序列排列，卷积在其上滑动）
    const embTitle = document.createElement('div')
    embTitle.className = 'chart-title'
    embTitle.textContent = `词向量矩阵（序列 ${tokens.length} 步 × ${state.embedDim} 维，红框 = 当前卷积窗口）`
    mainEl.appendChild(embTitle)
    const embHost = document.createElement('div')
    const hlRows = state.selectedWindow >= 0
      ? Array.from({ length: k0.width }, (_, w) => state.selectedWindow + w)
      : []
    renderHeatmap(embHost, {
      rowLabels: tokens,
      colLabels: Array.from({ length: state.embedDim }, (_, d) => `d${d + 1}`),
      values: emb,
      format: fmt,
      color: divergingColor,
      highlightRows: hlRows
    })
    mainEl.appendChild(embHost)

    // 各卷积核窗口
    kernelResults.forEach((kr, ki) => {
      const sec = document.createElement('div')
      sec.className = 'kernel-sec'
      const title = document.createElement('div')
      title.className = 'chart-title'
      title.textContent = `卷积核 ${ki + 1}（窗口=${kr.width}）：滑动 → ReLU → 池化`
      sec.appendChild(title)

      const wins = document.createElement('div')
      wins.className = 'conv-windows'
      kr.raw.forEach((rawV, wi) => {
        const win = document.createElement('div')
        const isSelected = ki === 0 && state.selectedWindow === wi
        const isPooled = kr.pooledWindow === wi
        win.className = 'conv-window' + (isSelected ? ' selected' : '') + (isPooled ? ' pooled' : '')
        win.innerHTML = `<span class="win-tokens">${tokens.slice(wi, wi + kr.width).join(' ')}</span><span class="num win-val">${fmt(rawV)} → ReLU ${fmt(kr.relu[wi])}</span>`
        win.addEventListener('click', () => {
          stopAnimation()
          state.selectedWindow = wi
          renderMain()
          renderInspectorPane()
        })
        wins.appendChild(win)
      })
      sec.appendChild(wins)

      const poolInfo = document.createElement('div')
      poolInfo.className = 'pool-info'
      poolInfo.textContent = `max-over-time 池化值 = ${fmt(kr.pooled)}`
      sec.appendChild(poolInfo)
      mainEl.appendChild(sec)
    })

    // 拼接向量
    const concatTitle = document.createElement('div')
    concatTitle.className = 'chart-title'
    concatTitle.textContent = '拼接向量（各卷积核池化值拼接）'
    mainEl.appendChild(concatTitle)
    const concatRow = document.createElement('div')
    concatRow.className = 'concat-row'
    kernelResults.forEach((kr, ki) => {
      const cell = document.createElement('div')
      cell.className = 'concat-cell'
      cell.innerHTML = `<span class="concat-label">核${ki + 1} 池化</span><span class="num">${fmt(kr.pooled)}</span>`
      concatRow.appendChild(cell)
    })
    mainEl.appendChild(concatRow)

    // 分类层权重
    const wTitle = document.createElement('div')
    wTitle.className = 'chart-title'
    wTitle.textContent = '分类层权重（类别 × 卷积核）与偏置'
    mainEl.appendChild(wTitle)
    const wHost = document.createElement('div')
    renderHeatmap(wHost, {
      rowLabels: state.classNames,
      colLabels: state.kernels.map((_, ki) => `核${ki + 1}`),
      values: state.fcWeights,
      format: fmt,
      color: divergingColor
    })
    mainEl.appendChild(wHost)
    const biasRow = document.createElement('div')
    biasRow.className = 'dim'
    biasRow.textContent = `偏置 b = [${state.fcBias.map(fmt).join(', ')}]`
    mainEl.appendChild(biasRow)

    // 得分计算
    const scoreTitle = document.createElement('div')
    scoreTitle.className = 'chart-title'
    scoreTitle.textContent = '得分计算（权重 · 拼接向量 + 偏置）→ softmax 概率'
    mainEl.appendChild(scoreTitle)
    state.classNames.forEach((name, c) => {
      const row = document.createElement('div')
      row.className = 'score-calc'
      const terms = state.fcWeights[c].map((w, j) => `${fmt(w)}×${fmt(state.result.pooledVec[j])}`)
      row.innerHTML = `<span class="cls-name">${name}</span><span class="num">${terms.join(' + ')} + ${fmt(state.fcBias[c])} = ${fmt(scores[c])} → ${fmt(probs[c])}</span>`
      mainEl.appendChild(row)
    })

    // 分类条形图
    const clsTitle = document.createElement('div')
    clsTitle.className = 'chart-title'
    clsTitle.textContent = '分类概率（softmax）'
    mainEl.appendChild(clsTitle)
    const bars = document.createElement('div')
    bars.className = 'cls-bars'
    state.classNames.forEach((name, c) => {
      const bar = document.createElement('div')
      bar.className = 'cls-bar'
      bar.innerHTML = `<span class="cls-name">${name}</span><span class="num">分数 ${fmt(scores[c])} · 概率 ${fmt(probs[c])}</span>`
      bars.appendChild(bar)
    })
    mainEl.appendChild(bars)
  }

  function renderInspectorPane() {
    inspectorEl.innerHTML = ''
    if (!state.result) {
      const hint = document.createElement('div')
      hint.className = 'inspector-hint'
      hint.textContent = '点击某个窗口查看逐项乘加。'
      inspectorEl.appendChild(hint)
      return
    }
    const wi = state.selectedWindow
    if (wi < 0) {
      const hint = document.createElement('div')
      hint.className = 'inspector-hint'
      hint.textContent = '点击卷积窗口（或播放动画）查看该窗口的逐项乘加。'
      inspectorEl.appendChild(hint)
      return
    }
    const ki = 0
    const kr = state.result.kernelResults[ki]
    const { tokens, emb } = state.result
    const k = state.kernels[ki]
    const steps = []
    const inputs = []
    let sum = 0
    for (let w = 0; w < k.width; w++) {
      const tv = emb[wi + w]
      const kv = k.values[w]
      const prod = tv.reduce((s, x, d) => s + x * kv[d], 0)
      sum += prod
      inputs.push({ label: `token[${wi + w}] · kernel[${w}]`, value: `[${tv.join(',')}]·[${kv.join(',')}] = ${fmt(prod)}` })
    }
    steps.push({
      title: '卷积激活（池化前）',
      value: fmt(kr.relu[wi]),
      formula: 'ReLU( Σ_w x[wi+w] · kernel[w] )',
      inputs,
      meta: `窗口位置 ${wi}，窗口词「${tokens.slice(wi, wi + k.width).join(' ')}」，卷积和 = ${fmt(sum)}`
    })
    steps.push({
      title: 'max-over-time 池化',
      value: fmt(kr.pooled),
      formula: 'pool = max(激活值)',
      inputs: kr.relu.map((a, i) => ({ label: `窗口 ${i} 激活`, value: fmt(a) })),
      meta: kr.pooledWindow === wi ? '该窗口是最大激活，被池化选中' : '该窗口不是最大值'
    })
    renderInspector(inspectorEl, steps)
  }

  function renderControlsStatus() {
    const status = controlsEl.querySelector('.recalc-status')
    if (status) {
      status.textContent = state.stale ? '参数已修改，结果已过期' : ''
      status.className = 'run-status' + (state.stale ? ' stale' : '')
    }
  }

  function buildControls() {
    controlsEl.innerHTML = ''

    const textGroup = document.createElement('div')
    textGroup.className = 'control-group'
    const tl = document.createElement('div')
    tl.className = 'control-label'
    tl.textContent = '输入文本（字符级切分）'
    textGroup.appendChild(tl)
    const textInput = document.createElement('input')
    textInput.type = 'text'
    textInput.value = state.text
    textInput.addEventListener('input', () => {
      state.text = textInput.value
      markStale()
    })
    textGroup.appendChild(textInput)
    controlsEl.appendChild(textGroup)

    const dimGroup = document.createElement('div')
    dimGroup.className = 'control-group'
    const dl = document.createElement('div')
    dl.className = 'control-label'
    dl.textContent = `Embedding 维度 = ${state.embedDim}`
    dimGroup.appendChild(dl)
    const dimRange = document.createElement('input')
    dimRange.type = 'range'
    dimRange.min = '1'
    dimRange.max = '8'
    dimRange.value = state.embedDim
    dimRange.addEventListener('input', () => {
      state.embedDim = Number(dimRange.value)
      dl.textContent = `Embedding 维度 = ${state.embedDim}`
      markStale()
    })
    dimGroup.appendChild(dimRange)
    controlsEl.appendChild(dimGroup)

    state.kernels.forEach((k, ki) => {
      const group = document.createElement('div')
      group.className = 'control-group'
      const kl = document.createElement('div')
      kl.className = 'control-label'
      kl.textContent = `卷积核 ${ki + 1}（窗口=${k.width}）`
      group.appendChild(kl)
      k.values.forEach((row, w) => {
        const rowDiv = document.createElement('div')
        rowDiv.className = 'init-row'
        row.forEach((v, d) => {
          const num = document.createElement('input')
          num.type = 'number'
          num.step = 'any'
          num.value = v
          num.addEventListener('input', () => {
            k.values[w][d] = Number(num.value) || 0
            markStale()
          })
          rowDiv.appendChild(num)
        })
        group.appendChild(rowDiv)
      })
      controlsEl.appendChild(group)
    })

    const recalc = document.createElement('button')
    recalc.type = 'button'
    recalc.className = 'btn run-btn'
    recalc.textContent = '重新计算'
    recalc.addEventListener('click', compute)
    controlsEl.appendChild(recalc)
    const status = document.createElement('div')
    status.className = 'run-status recalc-status'
    controlsEl.appendChild(status)

    const animRow = document.createElement('div')
    animRow.className = 'btn-row'
    const playBtn = document.createElement('button')
    playBtn.type = 'button'
    playBtn.className = 'btn'
    playBtn.textContent = '播放卷积动画'
    playBtn.addEventListener('click', playAnimation)
    const stopBtn = document.createElement('button')
    stopBtn.type = 'button'
    stopBtn.className = 'btn'
    stopBtn.textContent = '停止'
    stopBtn.addEventListener('click', () => {
      stopAnimation()
      state.selectedWindow = -1
      renderMain()
      renderInspectorPane()
    })
    animRow.appendChild(playBtn)
    animRow.appendChild(stopBtn)
    controlsEl.appendChild(animRow)

    const note = document.createElement('p')
    note.className = 'dim'
    note.textContent = '修改参数后点「重新计算」；动画在词向量矩阵上滑动高亮第一个卷积核的窗口。'
    controlsEl.appendChild(note)
  }

  buildControls()
  compute()
}
