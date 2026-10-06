import * as d3 from 'd3'
import {
  trainSkipGram,
  trainCBOW,
  trainGloVe,
  cooccurrence,
  buildVocab,
  pca2D,
  cosineSimilarity
} from '../algorithms/embeddings.js'
import { renderLineChart } from '../viz/linechart.js'
import { renderScatter } from '../viz/scatter.js'

const fmt = (x) => (Number.isFinite(x) ? x.toFixed(4) : '—')

const DEFAULT_CORPUS = [
  '我 爱 自然 语言 处理',
  '自然 语言 处理 很 有趣',
  '我 爱 编程',
  '机器 学习 很 有趣',
  '深度 学习 处理 图像',
  '我 爱 机器 学习'
].join('\n')

const ALGORITHMS = [
  ['skipgram', 'Skip-gram'],
  ['cbow', 'CBOW'],
  ['glove', 'GloVe']
]

function tokenizeCorpus(text) {
  return text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l !== '')
    .map((l) => (l.includes(' ') ? l.split(/\s+/) : [...l]))
}

function nearest(vectors, id2word, vec, k, exclude = new Set()) {
  return vectors
    .map((v, i) => ({ word: id2word[i], sim: cosineSimilarity(vec, v) }))
    .filter((x) => !exclude.has(x.word))
    .sort((a, b) => b.sim - a.sim)
    .slice(0, k)
}

export function renderEmbeddings(container) {
  const state = {
    corpus: DEFAULT_CORPUS,
    algorithm: 'skipgram',
    dim: 8,
    window: 2,
    epochs: 20,
    lr: 0.05,
    seed: 42,
    result: null,
    selected: new Set(), // 当前选中的词下标
    scatterApi: null
  }

  const root = document.createElement('div')
  root.className = 'module emb-module'
  container.appendChild(root)
  root.innerHTML = `
    <div class="module-head">
      <h1>M01 词向量</h1>
      <p class="module-desc">CBOW / Skip-gram 负采样与 GloVe 真实训练；查询相似词、类比与二维投影（可缩放/框选）。</p>
    </div>
    <div class="emb-layout">
      <aside class="emb-controls"></aside>
      <section class="emb-main"></section>
      <aside class="emb-inspector"></aside>
    </div>
  `

  const controlsEl = root.querySelector('.emb-controls')
  const mainEl = root.querySelector('.emb-main')
  const inspectorEl = root.querySelector('.emb-inspector')

  function train() {
    const corpus = tokenizeCorpus(state.corpus)
    if (corpus.length === 0) return
    const base = { dim: state.dim, window: state.window, epochs: state.epochs, lr: state.lr, seed: state.seed }
    if (state.algorithm === 'glove') {
      const { word2id, vocabSize } = buildVocab(corpus)
      const { sparse } = cooccurrence(corpus, word2id, state.window)
      const r = trainGloVe(sparse, vocabSize, base)
      state.result = { vectors: r.vectors, word2id, id2word: [...word2id.keys()], losses: r.losses, algorithm: 'GloVe' }
    } else {
      const fn = state.algorithm === 'cbow' ? trainCBOW : trainSkipGram
      const r = fn(corpus, { ...base, negative: 3 })
      state.result = { vectors: r.W, word2id: r.word2id, id2word: r.id2word, losses: r.losses, algorithm: state.algorithm === 'cbow' ? 'CBOW' : 'Skip-gram' }
    }
    state.selected = new Set()
    renderMain()
    renderInspectorPane()
  }

  function renderMain() {
    mainEl.innerHTML = ''
    if (!state.result) {
      const hint = document.createElement('div')
      hint.className = 'inspector-hint'
      hint.textContent = '点击「训练」开始。'
      mainEl.appendChild(hint)
      return
    }
    const { vectors, id2word, losses } = state.result

    const lossTitle = document.createElement('div')
    lossTitle.className = 'chart-title'
    lossTitle.textContent = `训练损失（${state.result.algorithm}）`
    mainEl.appendChild(lossTitle)
    const chartHost = document.createElement('div')
    renderLineChart(chartHost, {
      data: losses.map((y, i) => ({ x: i + 1, y })),
      width: 620,
      height: 180,
      xLabel: 'epoch',
      yLabel: '损失'
    })
    mainEl.appendChild(chartHost)

    const scTitle = document.createElement('div')
    scTitle.className = 'chart-title'
    scTitle.textContent = '二维投影（PCA）'
    mainEl.appendChild(scTitle)
    const scHost = document.createElement('div')
    const pts = pca2D(vectors)
    state.scatterApi = renderScatter(scHost, {
      points: pts,
      labels: id2word,
      width: 620,
      height: 340,
      onSelect: (indices) => {
        state.selected = new Set(indices)
        renderInspectorPane()
      }
    })
    mainEl.appendChild(scHost)
    const hint = document.createElement('div')
    hint.className = 'scatter-hint'
    hint.textContent = '拖拽平移 · 滚轮缩放 · Shift 拖拽框选 · 点击选中查看原始向量'
    mainEl.appendChild(hint)
  }

  function showWord(index) {
    state.selected = new Set([index])
    if (state.scatterApi) state.scatterApi.focusPoint(index)
    renderInspectorPane()
  }

  function renderInspectorPane() {
    inspectorEl.innerHTML = ''
    const wrap = document.createElement('div')
    wrap.className = 'emb-query'

    // 词查询
    const qTitle = document.createElement('div')
    qTitle.className = 'control-label'
    qTitle.textContent = '相似词查询'
    wrap.appendChild(qTitle)
    const qInput = document.createElement('input')
    qInput.type = 'text'
    qInput.placeholder = '输入词'
    wrap.appendChild(qInput)
    const qBtn = document.createElement('button')
    qBtn.type = 'button'
    qBtn.className = 'btn'
    qBtn.textContent = '查询'
    wrap.appendChild(qBtn)
    const qResult = document.createElement('div')
    qResult.className = 'neighbor-list'
    wrap.appendChild(qResult)

    const doQuery = () => {
      const w = qInput.value.trim()
      if (!state.result) return
      const { vectors, word2id, id2word } = state.result
      const id = word2id.get(w)
      qResult.innerHTML = ''
      if (id === undefined) {
        qResult.textContent = w ? '词不在词表中' : ''
        return
      }
      nearest(vectors, id2word, vectors[id], 6, new Set([w])).forEach((n) => {
        const li = document.createElement('div')
        li.className = 'neighbor-item'
        li.innerHTML = `<span>${n.word}</span><span class="num">${fmt(n.sim)}</span>`
        li.addEventListener('click', () => showWord(word2id.get(n.word)))
        qResult.appendChild(li)
      })
    }
    qBtn.addEventListener('click', doQuery)
    qInput.addEventListener('change', doQuery)

    // 类比
    const aTitle = document.createElement('div')
    aTitle.className = 'control-label'
    aTitle.textContent = '类比（A - B + C ≈ D）'
    wrap.appendChild(aTitle)
    const aRow = document.createElement('div')
    aRow.className = 'init-row'
    const aInputs = {}
    for (const key of ['a', 'b', 'c']) {
      const inp = document.createElement('input')
      inp.type = 'text'
      inp.placeholder = key.toUpperCase()
      aInputs[key] = inp
      aRow.appendChild(inp)
    }
    wrap.appendChild(aRow)
    const aBtn = document.createElement('button')
    aBtn.type = 'button'
    aBtn.className = 'btn'
    aBtn.textContent = '计算类比'
    wrap.appendChild(aBtn)
    const aResult = document.createElement('div')
    aResult.className = 'neighbor-list'
    wrap.appendChild(aResult)
    aBtn.addEventListener('click', () => {
      if (!state.result) return
      const { vectors, word2id, id2word } = state.result
      const ia = word2id.get(aInputs.a.value.trim())
      const ib = word2id.get(aInputs.b.value.trim())
      const ic = word2id.get(aInputs.c.value.trim())
      aResult.innerHTML = ''
      if (ia === undefined || ib === undefined || ic === undefined) {
        aResult.textContent = 'A/B/C 需都在词表中'
        return
      }
      const vec = vectors[ia].map((v, d) => v - vectors[ib][d] + vectors[ic][d])
      nearest(vectors, id2word, vec, 3, new Set([aInputs.a.value.trim(), aInputs.b.value.trim(), aInputs.c.value.trim()])).forEach((n) => {
        const li = document.createElement('div')
        li.className = 'neighbor-item'
        li.innerHTML = `<span>${n.word}</span><span class="num">${fmt(n.sim)}</span>`
        aResult.appendChild(li)
      })
    })

    // 选中词详情（原始向量）
    const sTitle = document.createElement('div')
    sTitle.className = 'control-label'
    sTitle.textContent = '选中词原始向量'
    wrap.appendChild(sTitle)
    const detail = document.createElement('div')
    detail.className = 'vector-list'
    wrap.appendChild(detail)

    if (state.result && state.selected.size > 0) {
      const { vectors, id2word } = state.result
      if (state.selected.size === 1) {
        const i = [...state.selected][0]
        const word = id2word[i]
        const vec = vectors[i]
        detail.innerHTML = `<div class="cls-name">${word}</div><div class="vec-dim">[${vec.map(fmt).join(', ')}]</div>`
        const nears = nearest(vectors, id2word, vec, 5, new Set([word]))
        const nl = document.createElement('div')
        nl.className = 'neighbor-list'
        nears.forEach((n) => {
          const li = document.createElement('div')
          li.className = 'neighbor-item'
          li.innerHTML = `<span>${n.word}</span><span class="num">${fmt(n.sim)}</span>`
          nl.appendChild(li)
        })
        detail.appendChild(nl)
      } else {
        const words = [...state.selected].map((i) => id2word[i])
        detail.innerHTML = `<div>已选中 ${state.selected.size} 个词：</div><div>${words.join('、')}</div>`
      }
    } else {
      detail.innerHTML = '<div class="dim">点击散点或查询结果查看原始向量。</div>'
    }

    inspectorEl.appendChild(wrap)
  }

  function buildControls() {
    controlsEl.innerHTML = ''

    const corpusGroup = document.createElement('div')
    corpusGroup.className = 'control-group'
    const cl = document.createElement('div')
    cl.className = 'control-label'
    cl.textContent = '语料（一行一句）'
    corpusGroup.appendChild(cl)
    const ta = document.createElement('textarea')
    ta.rows = 5
    ta.value = state.corpus
    ta.addEventListener('input', () => {
      state.corpus = ta.value
    })
    corpusGroup.appendChild(ta)
    controlsEl.appendChild(corpusGroup)

    // 算法选择（绿色选中描边）
    const algGroup = document.createElement('div')
    algGroup.className = 'control-group'
    const al = document.createElement('div')
    al.className = 'control-label'
    al.textContent = '算法'
    algGroup.appendChild(al)
    const algRow = document.createElement('div')
    algRow.className = 'btn-row'
    const renderAlgButtons = () => {
      algRow.innerHTML = ''
      for (const [key, label] of ALGORITHMS) {
        const b = document.createElement('button')
        b.type = 'button'
        b.className = 'btn' + (state.algorithm === key ? ' selected' : '')
        b.textContent = label
        b.addEventListener('click', () => {
          state.algorithm = key
          renderAlgButtons()
        })
        algRow.appendChild(b)
      }
    }
    renderAlgButtons()
    algGroup.appendChild(algRow)
    controlsEl.appendChild(algGroup)

    // 参数
    const params = [
      ['训练轮数 epochs', 'epochs', 1, 100, 1],
      ['学习率 lr', 'lr', 0.001, 1, 0.005],
      ['随机种子 seed', 'seed', 0, 999, 1],
      ['维度 dim', 'dim', 2, 16, 1]
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
      range.addEventListener('input', () => set(Number(range.value)))
      num.addEventListener('input', () => {
        const v = Number(num.value)
        if (Number.isFinite(v)) set(v)
      })
      row.appendChild(range)
      row.appendChild(num)
      group.appendChild(row)
      controlsEl.appendChild(group)
    }

    const trainBtn = document.createElement('button')
    trainBtn.type = 'button'
    trainBtn.className = 'btn run-btn'
    trainBtn.textContent = '训练'
    trainBtn.addEventListener('click', train)
    controlsEl.appendChild(trainBtn)
  }

  buildControls()
  renderMain()
  renderInspectorPane()
}
