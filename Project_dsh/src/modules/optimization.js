import * as d3 from 'd3'
import { makeBinaryDataset, trainLogistic2D, lossSurface2D } from '../algorithms/optimization.js'
import { renderLineChart } from '../viz/linechart.js'
import { renderInspector } from '../components/inspector.js'
import { createStateMachine, STATES } from '../core/state-machine.js'
import { saveRecord } from '../data/notebook.js'

const PRESETS = {
  slow: { label: '慢收敛', lr: 0.002, epochs: 200 },
  normal: { label: '正常收敛', lr: 0.1, epochs: 60 },
  oscillate: { label: '震荡', lr: 200, epochs: 50 }
}

function lossColor(v, minV, maxV) {
  const t = maxV === minV ? 0 : Math.max(0, Math.min(1, (v - minV) / (maxV - minV)))
  const r = Math.round(255 - 210 * t)
  const g = Math.round(255 - 130 * t)
  const b = Math.round(255 - 30 * t)
  return `rgb(${r},${g},${b})`
}

export function renderOptimization(container) {
  const state = {
    lr: PRESETS.normal.lr,
    epochs: PRESETS.normal.epochs,
    init: [0, 0],
    seed: 42,
    batchSize: 100,
    dataset: null,
    result: null,
    selected: -1,
    sm: createStateMachine()
  }
  state.dataset = makeBinaryDataset(100, state.seed)

  const root = document.createElement('div')
  root.className = 'module optimization-module'
  container.appendChild(root)
  root.innerHTML = `
    <div class="module-head">
      <h1>M06 损失与优化</h1>
      <p class="module-desc">真实执行前向、交叉熵、梯度与参数更新。曲线每个点都来自真实迭代；二维等高线展示参数轨迹。</p>
    </div>
    <div class="opt-layout">
      <aside class="opt-controls"></aside>
      <section class="opt-main">
        <div class="opt-chart-wrap"><div class="chart-title">训练损失曲线（每步真实评估）</div></div>
        <div class="opt-contour-wrap"><div class="chart-title">二维损失等高线 + 优化轨迹</div></div>
      </section>
      <aside class="opt-inspector"></aside>
    </div>
  `

  const controlsEl = root.querySelector('.opt-controls')
  const chartEl = root.querySelector('.opt-chart-wrap')
  const contourEl = root.querySelector('.opt-contour-wrap')
  const inspectorEl = root.querySelector('.opt-inspector')

  const statusEl = document.createElement('div')
  statusEl.className = 'run-status'

  function updateStatus() {
    const s = state.sm.state
    statusEl.textContent =
      s === STATES.stale ? '⚠ 参数已变，结果已过期（请重新运行）'
        : s === STATES.completed ? '✓ 已完成'
        : s === STATES.running ? '运行中…'
        : '未运行'
    statusEl.className = 'run-status' + (s === STATES.stale ? ' stale' : '')
  }
  state.sm.subscribe(updateStatus)

  function markStale() {
    state.sm.markStale()
    updateStatus()
  }

  function run() {
    state.sm.transition(STATES.validating)
    state.sm.transition(STATES.running)
    const { X, y } = state.dataset
    state.result = trainLogistic2D(X, y, { lr: state.lr, epochs: state.epochs, init: state.init, batchSize: state.batchSize, seed: state.seed })
    state.selected = -1
    state.sm.transition(STATES.completed)
    renderResults()
    renderInspectorPane()
  }

  function saveToNotebook() {
    if (!state.result) return
    saveRecord({
      module: 'M06 优化',
      name: `优化 · lr=${state.lr} batch=${state.batchSize}`,
      config: {
        lr: state.lr,
        epochs: state.epochs,
        init: state.init,
        batchSize: state.batchSize,
        seed: state.seed
      },
      result: {
        weights: state.result.weights,
        finalLoss: state.result.losses[state.result.losses.length - 1],
        trajectory: state.result.trajectory,
        divergedAt: state.result.divergedAt
      }
    })
    window.alert('已保存到实验记录')
  }

  function renderResults() {
    const { losses } = state.result
    const data = losses.map((loss, i) => ({ x: i, y: loss }))
    const host = document.createElement('div')
    renderLineChart(host, {
      data,
      width: 620,
      height: 260,
      xLabel: '迭代步',
      yLabel: '交叉熵损失',
      selected: state.selected,
      onPointClick: (i) => {
        state.selected = i
        renderResults()
        renderInspectorPane()
      }
    })
    const t = chartEl.querySelector('.chart-title')
    chartEl.innerHTML = ''
    chartEl.appendChild(t)
    chartEl.appendChild(host)
    renderContour()
  }

  function renderContour() {
    const host = document.createElement('div')
    host.className = 'contour-svg'
    const width = 620
    const height = 320
    const margin = { top: 12, right: 16, bottom: 40, left: 44 }
    const innerW = width - margin.left - margin.right
    const innerH = height - margin.top - margin.bottom

    const pts = state.result.trajectory
    const w0s = pts.map((p) => p[0])
    const w1s = pts.map((p) => p[1])
    const pad = 1
    const lo0 = Math.min(-1, d3.min(w0s)) - pad
    const hi0 = Math.max(1, d3.max(w0s)) + pad
    const lo1 = Math.min(-1, d3.min(w1s)) - pad
    const hi1 = Math.max(1, d3.max(w1s)) + pad

    const N = 48
    const g0 = Array.from({ length: N }, (_, i) => lo0 + (i / (N - 1)) * (hi0 - lo0))
    const g1 = Array.from({ length: N }, (_, i) => lo1 + (i / (N - 1)) * (hi1 - lo1))
    const Z = lossSurface2D(state.dataset.X, state.dataset.y, g0, g1)
    const allZ = Z.flat()
    const minZ = d3.min(allZ)
    const maxZ = d3.max(allZ)

    const xScale = d3.scaleLinear().domain([lo0, hi0]).range([0, innerW])
    const yScale = d3.scaleLinear().domain([lo1, hi1]).range([innerH, 0])
    const cw = innerW / N
    const ch = innerH / N

    const svg = d3.select(host).append('svg').attr('width', width).attr('height', height)
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        g.append('rect')
          .attr('x', xScale(g0[j]) - cw / 2)
          .attr('y', yScale(g1[i]) - ch / 2)
          .attr('width', cw + 0.5)
          .attr('height', ch + 0.5)
          .attr('fill', lossColor(Z[i][j], minZ, maxZ))
      }
    }

    const trajLine = d3.line().x((d) => xScale(d[0])).y((d) => yScale(d[1]))
    g.append('path').datum(pts).attr('fill', 'none').attr('stroke', 'var(--accent)').attr('stroke-width', 1.6).attr('d', trajLine)
    g.selectAll('circle.traj').data(pts).join('circle')
      .attr('class', 'traj')
      .attr('cx', (d) => xScale(d[0]))
      .attr('cy', (d) => yScale(d[1]))
      .attr('r', (d, i) => (i === state.selected ? 5 : i === 0 ? 4 : 2))
      .attr('fill', (d, i) => (i === state.selected ? 'var(--danger)' : i === 0 ? 'var(--warning)' : 'var(--accent)'))

    g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(6))
    g.append('g').call(d3.axisLeft(yScale).ticks(6))
    g.append('text').attr('x', innerW / 2).attr('y', innerH + 34).attr('text-anchor', 'middle').attr('class', 'axis-label').text('w0')
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -innerH / 2).attr('y', -34).attr('text-anchor', 'middle').attr('class', 'axis-label').text('w1')

    const t = contourEl.querySelector('.chart-title')
    contourEl.innerHTML = ''
    contourEl.appendChild(t)
    contourEl.appendChild(host)
  }

  function renderInspectorPane() {
    if (!state.result) {
      inspectorEl.innerHTML = ''
      const hint = document.createElement('div')
      hint.className = 'inspector-hint'
      hint.textContent = '点击曲线点查看该步的配置与数值。'
      inspectorEl.appendChild(hint)
      return
    }
    const steps = []
    if (state.selected >= 0) {
      const i = state.selected
      const loss = state.result.losses[i]
      const w = state.result.trajectory[i]
      steps.push({
        title: '迭代步',
        value: String(i),
        formula: `学习率 ${state.lr} · 初始 [${state.init.join(', ')}]`,
        inputs: [
          { label: '损失', value: Number.isFinite(loss) ? loss.toFixed(6) : '∞（发散）' },
          { label: '参数', value: `[${w[0].toFixed(4)}, ${w[1].toFixed(4)}]` }
        ],
        meta: '损失为该步参数在全部 100 个样本上的真实均值交叉熵'
      })
    } else {
      const last = state.result.losses[state.result.losses.length - 1]
      steps.push({
        title: '训练结果',
        value: Number.isFinite(last) ? last.toFixed(6) : '∞（发散）',
        formula: `学习率 ${state.lr} · ${state.result.losses.length - 1} 步`,
        inputs: [
          { label: '最终参数', value: `[${state.result.weights[0].toFixed(4)}, ${state.result.weights[1].toFixed(4)}]` },
          { label: '初始损失', value: state.result.losses[0].toFixed(6) }
        ],
        meta: state.result.divergedAt != null ? `第 ${state.result.divergedAt} 步出现非有限值` : '未出现非有限值'
      })
    }
    renderInspector(inspectorEl, steps)
  }

  function sliderField(label, key, min, max, step, onChange) {
    const group = document.createElement('div')
    group.className = 'control-group'
    const lab = document.createElement('div')
    lab.className = 'control-label'
    lab.textContent = label
    group.appendChild(lab)
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
      lab.textContent = `${label} = ${v}`
      if (onChange) onChange(v)
      markStale()
    }
    range.addEventListener('input', () => set(Number(range.value)))
    num.addEventListener('input', () => {
      const v = Number(num.value)
      if (Number.isFinite(v)) set(v)
    })
    row.appendChild(range)
    row.appendChild(num)
    group.appendChild(row)
    return group
  }

  function buildControls() {
    controlsEl.innerHTML = ''

    const presetGroup = document.createElement('div')
    presetGroup.className = 'control-group'
    const presetLabel = document.createElement('div')
    presetLabel.className = 'control-label'
    presetLabel.textContent = '验证案例预设'
    presetGroup.appendChild(presetLabel)
    const btnRow = document.createElement('div')
    btnRow.className = 'btn-row'
    for (const p of Object.values(PRESETS)) {
      const b = document.createElement('button')
      b.type = 'button'
      b.className = 'btn'
      b.textContent = p.label
      b.addEventListener('click', () => {
        state.lr = p.lr
        state.epochs = p.epochs
        markStale()
        buildControls()
      })
      btnRow.appendChild(b)
    }
    presetGroup.appendChild(btnRow)
    controlsEl.appendChild(presetGroup)

    controlsEl.appendChild(sliderField('学习率 lr', 'lr', 0.001, 1, 0.001))
    controlsEl.appendChild(sliderField('迭代数 epochs', 'epochs', 10, 500, 1))
    controlsEl.appendChild(sliderField('批大小 batchSize', 'batchSize', 1, 100, 1))
    controlsEl.appendChild(sliderField('随机种子 seed', 'seed', 0, 999, 1, () => {
      state.dataset = makeBinaryDataset(100, state.seed)
    }))

    const initGroup = document.createElement('div')
    initGroup.className = 'control-group'
    const initLabel = document.createElement('div')
    initLabel.className = 'control-label'
    initLabel.textContent = '初始参数 [w0, w1]'
    initGroup.appendChild(initLabel)
    const initRow = document.createElement('div')
    initRow.className = 'init-row'
    state.init.forEach((v, i) => {
      const num = document.createElement('input')
      num.type = 'number'
      num.step = 'any'
      num.value = v
      num.addEventListener('input', () => {
        state.init[i] = Number(num.value) || 0
        markStale()
      })
      initRow.appendChild(num)
    })
    initGroup.appendChild(initRow)
    controlsEl.appendChild(initGroup)

    const runBtn = document.createElement('button')
    runBtn.type = 'button'
    runBtn.className = 'btn run-btn'
    runBtn.textContent = '运行训练'
    runBtn.addEventListener('click', run)
    controlsEl.appendChild(runBtn)

    const saveBtn = document.createElement('button')
    saveBtn.type = 'button'
    saveBtn.className = 'btn'
    saveBtn.textContent = '保存到记录'
    saveBtn.addEventListener('click', saveToNotebook)
    controlsEl.appendChild(saveBtn)

    controlsEl.appendChild(statusEl)
  }

  buildControls()
  updateStatus()
  run()
}
