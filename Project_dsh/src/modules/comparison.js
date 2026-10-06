import {
  tokenize,
  buildVocabulary,
  bowVector,
  trainNaiveBayes,
  predictNaiveBayes,
  trainLinearSVM,
  svmDecisionScores
} from '../algorithms/classifiers.js'
import {
  trainRNNClassifier,
  predictRNNClassifier,
  trainCNNClassifier,
  predictCNNClassifier
} from '../algorithms/textclassifiers.js'
import { confusionMatrix, accuracy, macroF1 } from '../core/metrics.js'

const fmt = (x) => (Number.isFinite(x) ? x.toFixed(3) : '—')

// 内置二分类数据集：训练集（清晰 + 简单否定/转折）+ 测试集（新否定/转折组合）
const DATASET = [
  { text: '这 部 电影 太 棒 了', label: '正面', split: 'train' },
  { text: '画面 精美 音乐 好听', label: '正面', split: 'train' },
  { text: '主演 演技 精湛', label: '正面', split: 'train' },
  { text: '剧情 拖沓 无聊', label: '负面', split: 'train' },
  { text: '浪费 时间 难看', label: '负面', split: 'train' },
  { text: '太 无聊 中途 退场', label: '负面', split: 'train' },
  { text: '这 部 电影 不 好看', label: '负面', split: 'train' },
  { text: '演员 并 不 差', label: '正面', split: 'train' },
  { text: '虽然 画面 精美 但 故事 糟糕', label: '负面', split: 'train' },
  { text: '虽然 开头 无聊 但 结局 精彩', label: '正面', split: 'train' },
  { text: '剧情 并 不 精彩', label: '负面', split: 'test' },
  { text: '并 不 难看', label: '正面', split: 'test' },
  { text: '演技 精湛 但 剧情 空洞', label: '负面', split: 'test' },
  { text: '节奏 慢 但 内容 深刻', label: '正面', split: 'test' },
  { text: '特效 华丽 但 故事 空洞', label: '负面', split: 'test' },
  { text: '票房 一般 但 口碑 极好', label: '正面', split: 'test' }
]

const MODEL_KEYS = ['NB', 'SVM', 'RNN', 'CNN']
const MODEL_NAMES = { NB: '朴素贝叶斯', SVM: '线性 SVM', RNN: 'RNN', CNN: 'Text-CNN' }
const MODEL_NOTES = { NB: '对数概率', SVM: '决策分数（非概率）', RNN: 'softmax 概率', CNN: 'softmax 概率' }

/** 根据模型归纳偏置 + 样本语言现象生成误分类原因 */
function reasonFor(modelKey, sample) {
  const hasNeg = sample.includes('不')
  const hasBut = /但|可是|却/.test(sample)
  const base = {
    NB: '词袋模型按词频统计、忽略词序',
    SVM: '线性决策边界只拟合词频特征',
    RNN: 'RNN 能建模顺序，但训练数据太少、易过拟合',
    CNN: '卷积只捕捉局部 n-gram，难以处理长距离依赖'
  }[modelKey]
  if (hasNeg) return `${base}，未正确处理否定词「不」对情感的反转`
  if (hasBut) return `${base}，未正确处理转折「但」后情感起主导作用`
  return `${base}，对该类样本泛化不足`
}

export function renderComparison(container) {
  const state = {
    trained: null,
    input: '虽然 画面 一般 但 故事 动人'
  }

  const root = document.createElement('div')
  root.className = 'module cmp-module'
  container.appendChild(root)
  root.innerHTML = `
    <div class="module-head">
      <h1>M05 多模型对照</h1>
      <p class="module-desc">训练集/测试集固定划分，四模型在测试集上的预测、指标与短板对照。</p>
    </div>
    <div class="cmp-layout">
      <aside class="cmp-controls"></aside>
      <section class="cmp-main"></section>
      <aside class="cmp-inspector"></aside>
    </div>
  `

  const controlsEl = root.querySelector('.cmp-controls')
  const mainEl = root.querySelector('.cmp-main')
  const inspectorEl = root.querySelector('.cmp-inspector')

  function train() {
    const trainSamples = DATASET.filter((s) => s.split === 'train')
    const trainTexts = trainSamples.map((s) => s.text)
    const trainLabels = trainSamples.map((s) => s.label)
    const trainTokens = trainTexts.map((s) => tokenize(s))
    const vocab = buildVocabulary(trainTokens)
    const idx = new Map(vocab.map((w, i) => [w, i]))
    const trainX = trainTokens.map((toks) => bowVector(toks, vocab, idx))
    const classes = [...new Set(trainLabels)]
    const pos = classes[0]

    const nb = trainNaiveBayes(trainTokens, trainLabels, { alpha: 1 })
    const svm = trainLinearSVM(trainX, trainLabels.map((l) => (l === pos ? 1 : -1)), { lr: 0.02, epochs: 500, C: 1, seed: 42 })
    const rnn = trainRNNClassifier(trainTexts, trainLabels, { dim: 6, hidden: 6, epochs: 40, lr: 0.5, seed: 42 }).model
    const cnn = trainCNNClassifier(trainTexts, trainLabels, { dim: 6, epochs: 40, lr: 0.5, seed: 42 }).model

    const allLabels = DATASET.map((s) => s.label)
    const preds = {
      NB: DATASET.map((s) => predictNaiveBayes(nb, tokenize(s.text)).label),
      SVM: DATASET.map((s) => (svmDecisionScores(svm, [bowVector(tokenize(s.text), vocab, idx)])[0] >= 0 ? pos : classes[1])),
      RNN: DATASET.map((s) => predictRNNClassifier(rnn, s.text).label),
      CNN: DATASET.map((s) => predictCNNClassifier(cnn, s.text).label)
    }

    const details = {
      NB: DATASET.map((s) => predictNaiveBayes(nb, tokenize(s.text))),
      SVM: DATASET.map((s) => svmDecisionScores(svm, [bowVector(tokenize(s.text), vocab, idx)])[0]),
      RNN: DATASET.map((s) => predictRNNClassifier(rnn, s.text)),
      CNN: DATASET.map((s) => predictCNNClassifier(cnn, s.text))
    }
    state.trained = { nb, svm, rnn, cnn, vocab, idx, classes, pos, labels: allLabels, preds, details }
    renderMain()
    renderInspectorPane()
  }

  function renderMain() {
    mainEl.innerHTML = ''
    if (!state.trained) {
      const hint = document.createElement('div')
      hint.className = 'inspector-hint'
      hint.textContent = '点击「训练模型」开始。'
      mainEl.appendChild(hint)
      return
    }
    const { labels, preds, classes, details } = state.trained
    const testIdx = DATASET.map((s, i) => (s.split === 'test' ? i : -1)).filter((i) => i >= 0)
    const testLabels = testIdx.map((i) => labels[i])

    // 评测指标（测试集）
    const tTitle = document.createElement('div')
    tTitle.className = 'chart-title'
    tTitle.textContent = '评测指标（测试集，6 条）'
    mainEl.appendChild(tTitle)
    const table = document.createElement('table')
    table.className = 'preview-table'
    const thead = document.createElement('tr')
    thead.innerHTML = '<th>模型</th><th>准确率</th><th>Macro-F1</th><th>说明</th>'
    table.appendChild(thead)
    MODEL_KEYS.forEach((k) => {
      const p = testIdx.map((i) => preds[k][i])
      const tr = document.createElement('tr')
      tr.innerHTML = `<td>${MODEL_NAMES[k]}</td><td class="num">${fmt(accuracy(testLabels, p))}</td><td class="num">${fmt(macroF1(testLabels, p, classes))}</td><td>${MODEL_NOTES[k]}</td>`
      table.appendChild(tr)
    })
    mainEl.appendChild(table)

    // 批量样本对比（全部样本，标注训练/测试）
    const bTitle = document.createElement('div')
    bTitle.className = 'chart-title'
    bTitle.textContent = '批量样本对比（✓ 正确 · ✗ 错误）'
    mainEl.appendChild(bTitle)
    const bTable = document.createElement('table')
    bTable.className = 'cmp-batch'
    const bHead = document.createElement('tr')
    bHead.innerHTML = '<th>样本</th><th>真实</th><th>划分</th>' + MODEL_KEYS.map((k) => `<th>${MODEL_NAMES[k]}</th>`).join('')
    bTable.appendChild(bHead)
    DATASET.forEach((s, i) => {
      const tr = document.createElement('tr')
      const cells = [`<td>${s.text}</td><td>${s.label}</td><td class="dim">${s.split === 'train' ? '训练' : '测试'}</td>`]
      MODEL_KEYS.forEach((k) => {
        const ok = preds[k][i] === labels[i]
        cells.push(ok ? '<td class="ok">✓</td>' : `<td class="bad">✗ ${preds[k][i]}</td>`)
      })
      tr.innerHTML = cells.join('')
      bTable.appendChild(tr)
    })
    mainEl.appendChild(bTable)

    // 模型短板（测试集误分类，含预测数据与原因）
    const wTitle = document.createElement('div')
    wTitle.className = 'chart-title'
    wTitle.textContent = '模型短板（测试集误分类 · 详细原因）'
    mainEl.appendChild(wTitle)
    const wList = document.createElement('div')
    wList.className = 'weakness-list'
    MODEL_KEYS.forEach((k) => {
      const wrongIdx = testIdx.filter((i) => preds[k][i] !== labels[i])
      const modelCard = document.createElement('div')
      modelCard.className = 'weakness-model'
      const mh = document.createElement('div')
      mh.className = 'weakness-model-head'
      mh.textContent = `${MODEL_NAMES[k]}（误分类 ${wrongIdx.length} 条）`
      modelCard.appendChild(mh)
      if (wrongIdx.length === 0) {
        const p = document.createElement('div')
        p.className = 'dim'
        p.textContent = '测试集全对'
        modelCard.appendChild(p)
      } else {
        wrongIdx.forEach((i) => {
          const item = document.createElement('div')
          item.className = 'weakness-detail'
          let dataStr = ''
          if (k === 'SVM') {
            dataStr = `决策分数 = ${fmt(details.SVM[i])}`
          } else {
            const d = details[k][i]
            dataStr = `P(${d.classes[0]}) = ${fmt(d.probs[0])}，P(${d.classes[1]}) = ${fmt(d.probs[1])}`
          }
          const reason = reasonFor(k, DATASET[i].text)
          item.innerHTML =
            `<div class="dim">${DATASET[i].text}（真实 ${labels[i]} → 预测 ${preds[k][i]}）</div>` +
            `<div class="num">${dataStr}</div>` +
            `<div class="weakness-reason">原因：${reason}</div>`
          modelCard.appendChild(item)
        })
      }
      wList.appendChild(modelCard)
    })
    mainEl.appendChild(wList)

    // 混淆矩阵（朴素贝叶斯，测试集）
    const cmTitle = document.createElement('div')
    cmTitle.className = 'chart-title'
    cmTitle.textContent = '混淆矩阵（朴素贝叶斯 · 测试集）'
    mainEl.appendChild(cmTitle)
    const { matrix, labels: cmLabels } = confusionMatrix(testLabels, testIdx.map((i) => preds.NB[i]), classes)
    const cmTable = document.createElement('table')
    cmTable.className = 'cm-table'
    const cmHead = document.createElement('tr')
    cmHead.innerHTML = '<th>真实\\预测</th>' + cmLabels.map((l) => `<th>${l}</th>`).join('')
    cmTable.appendChild(cmHead)
    matrix.forEach((row, i) => {
      const tr = document.createElement('tr')
      tr.innerHTML = `<th>${cmLabels[i]}</th>` + row.map((v) => `<td class="num">${v}</td>`).join('')
      cmTable.appendChild(tr)
    })
    mainEl.appendChild(cmTable)
  }

  function renderInspectorPane() {
    inspectorEl.innerHTML = ''
    const wrap = document.createElement('div')
    const t = document.createElement('div')
    t.className = 'control-label'
    t.textContent = '自主输入 · 四模型预测'
    wrap.appendChild(t)
    const input = document.createElement('input')
    input.type = 'text'
    input.value = state.input
    input.addEventListener('input', () => {
      state.input = input.value
    })
    wrap.appendChild(input)
    const btn = document.createElement('button')
    btn.type = 'button'
    btn.className = 'btn'
    btn.textContent = '预测'
    wrap.appendChild(btn)
    const result = document.createElement('div')
    result.className = 'cmp-result'
    wrap.appendChild(result)

    btn.addEventListener('click', () => {
      result.innerHTML = ''
      if (!state.trained) {
        result.textContent = '请先训练模型'
        return
      }
      const { nb, svm, rnn, cnn, vocab, idx, classes, pos } = state.trained
      const toks = tokenize(state.input)
      const nbPred = predictNaiveBayes(nb, toks)
      const svmScore = svmDecisionScores(svm, [bowVector(toks, vocab, idx)])[0]
      const svmLabel = svmScore >= 0 ? pos : classes[1]
      const rnnPred = predictRNNClassifier(rnn, state.input)
      const cnnPred = predictCNNClassifier(cnn, state.input)

      const mk = (name, label, conf, detail) => {
        const d = document.createElement('div')
        d.className = 'cmp-model'
        d.innerHTML = `<div class="cmp-model-head"><span class="cls-name">${name}</span><span class="num">${label}</span></div><div class="dim">${detail}</div><div class="dim">置信度 ${conf}</div>`
        result.appendChild(d)
      }
      mk('朴素贝叶斯', nbPred.label, fmt(Math.max(...nbPred.probs)), `各类概率 [${nbPred.probs.map(fmt).join(', ')}]`)
      mk('线性 SVM', svmLabel, `${fmt(Math.abs(svmScore))}（未校准）`, `决策分数 ${fmt(svmScore)}`)
      mk('RNN', rnnPred.label, fmt(Math.max(...rnnPred.probs)), `概率 [${rnnPred.probs.map(fmt).join(', ')}]`)
      mk('Text-CNN', cnnPred.label, fmt(Math.max(...cnnPred.probs)), `概率 [${cnnPred.probs.map(fmt).join(', ')}]`)
    })

    inspectorEl.appendChild(wrap)
  }

  function buildControls() {
    controlsEl.innerHTML = ''
    const group = document.createElement('div')
    group.className = 'control-group'
    const l = document.createElement('div')
    l.className = 'control-label'
    l.textContent = '数据集：10 条训练 + 6 条测试（词级，含否定/转折）'
    group.appendChild(l)
    const btn = document.createElement('button')
    btn.type = 'button'
    btn.className = 'btn run-btn'
    btn.textContent = '训练模型'
    btn.addEventListener('click', train)
    group.appendChild(btn)
    controlsEl.appendChild(group)
    const note = document.createElement('p')
    note.className = 'dim'
    note.textContent = '词表只在训练集上拟合（不偷看测试集）。输入用空格分词；SVM 置信度为分数绝对值（未校准）。'
    controlsEl.appendChild(note)
  }

  buildControls()
  renderMain()
  renderInspectorPane()
}
