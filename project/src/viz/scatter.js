import * as d3 from 'd3'

/**
 * 交互式散点图（PCA 投影用）。
 * 支持：拖拽平移、滚轮缩放（横纵轴同步跟随）、Shift 框选、点或名字点击选中、标签防重叠。
 * focusPoint 可将某点平移居中。
 */
export function renderScatter(container, opts = {}) {
  const { points, labels, width = 620, height = 340, onSelect = () => {} } = opts
  const margin = { top: 12, right: 12, bottom: 32, left: 44 }
  const innerW = width - margin.left - margin.right
  const innerH = height - margin.top - margin.bottom

  const xScale = d3.scaleLinear()
    .domain([d3.min(points, (d) => d[0]), d3.max(points, (d) => d[0])])
    .nice()
    .range([0, innerW])
  const yScale = d3.scaleLinear()
    .domain([d3.min(points, (d) => d[1]), d3.max(points, (d) => d[1])])
    .nice()
    .range([innerH, 0])

  let xCur = xScale.copy()
  let yCur = yScale.copy()
  let selected = new Set()

  const textWidth = (label) => [...String(label)].length * 9 + 6

  container.innerHTML = ''
  const svg = d3.select(container).append('svg')
    .attr('width', width).attr('height', height)
    .style('cursor', 'grab')

  const plotG = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)
  const xAxisG = plotG.append('g').attr('transform', `translate(0,${innerH})`)
  const yAxisG = plotG.append('g')

  const boxRect = svg.append('rect')
    .attr('fill', 'var(--accent-soft)')
    .attr('stroke', 'var(--accent)')
    .attr('stroke-dasharray', '4 2')
    .attr('opacity', 0)
    .style('pointer-events', 'none')

  function renderAxes() {
    xAxisG.call(d3.axisBottom(xCur).ticks(5))
    yAxisG.call(d3.axisLeft(yCur).ticks(5))
  }

  function selectPoint(i, additive) {
    if (additive) {
      if (selected.has(i)) selected.delete(i)
      else selected.add(i)
    } else {
      selected = new Set([i])
    }
    onSelect([...selected])
    render()
  }

  function render() {
    // 标签防重叠（当前像素空间）
    const placed = []
    const visible = new Set()
    const order = points.map((_, i) => i).sort((a, b) => xCur(points[a][0]) - xCur(points[b][0]))
    for (const i of order) {
      const lx = xCur(points[i][0]) + 6
      const ly = yCur(points[i][1]) + 4
      const rect = { x: lx, y: ly - 11, w: textWidth(labels[i]), h: 15 }
      let overlap = false
      for (const p of placed) {
        if (rect.x < p.x + p.w && rect.x + rect.w > p.x && rect.y < p.y + p.h && rect.y + rect.h > p.y) {
          overlap = true
          break
        }
      }
      if (!overlap) {
        placed.push(rect)
        visible.add(i)
      }
      if (selected.has(i)) visible.add(i)
    }

    // 点
    plotG.selectAll('circle.p').data(points).join('circle')
      .attr('class', 'p')
      .attr('cx', (d) => xCur(d[0]))
      .attr('cy', (d) => yCur(d[1]))
      .attr('r', (d, i) => (selected.has(i) ? 6 : 4))
      .attr('fill', (d, i) => (selected.has(i) ? 'var(--danger)' : 'var(--accent)'))
      .attr('opacity', 0.9)
      .style('cursor', 'pointer')
      .on('click', (event, d) => selectPoint(points.indexOf(d), event.shiftKey))

    // 名字（可点击）
    plotG.selectAll('text.lab').data(points).join('text')
      .attr('class', 'lab')
      .attr('x', (d) => xCur(d[0]) + 6)
      .attr('y', (d) => yCur(d[1]) + 4)
      .text((d, i) => labels[i])
      .attr('font-size', 12)
      .attr('fill', 'var(--text)')
      .attr('display', (d, i) => (visible.has(i) ? null : 'none'))
      .style('cursor', 'pointer')
      .on('click', (event, d) => selectPoint(points.indexOf(d), false))
  }

  // 缩放/平移（横纵轴随 rescale 一起动；Shift 拖拽留给框选）
  const zoom = d3.zoom()
    .scaleExtent([0.5, 30])
    .clickDistance(6)
    .filter((event) => event.type === 'wheel' || !event.shiftKey)
    .on('zoom', (event) => {
      xCur = event.transform.rescaleX(xScale)
      yCur = event.transform.rescaleY(yScale)
      renderAxes()
      render()
    })
  svg.call(zoom)

  // Shift 框选
  let boxStart = null
  const toDataX = (sx) => xCur.invert(sx - margin.left)
  const toDataY = (sy) => yCur.invert(sy - margin.top)

  svg.on('mousedown', (event) => {
    if (!event.shiftKey) return
    boxStart = d3.pointer(event, svg.node())
    boxRect.attr('x', boxStart[0]).attr('y', boxStart[1]).attr('width', 0).attr('height', 0).attr('opacity', 0.5)
  })
  svg.on('mousemove', (event) => {
    if (!boxStart) return
    const p = d3.pointer(event, svg.node())
    boxRect
      .attr('x', Math.min(boxStart[0], p[0]))
      .attr('y', Math.min(boxStart[1], p[1]))
      .attr('width', Math.abs(p[0] - boxStart[0]))
      .attr('height', Math.abs(p[1] - boxStart[1]))
  })
  svg.on('mouseup', (event) => {
    if (!boxStart) return
    const p = d3.pointer(event, svg.node())
    const x0 = Math.min(boxStart[0], p[0])
    const x1 = Math.max(boxStart[0], p[0])
    const y0 = Math.min(boxStart[1], p[1])
    const y1 = Math.max(boxStart[1], p[1])
    boxStart = null
    boxRect.attr('opacity', 0)

    const dx0 = toDataX(x0)
    const dx1 = toDataX(x1)
    const dy0 = toDataY(y1)
    const dy1 = toDataY(y0)
    selected = new Set()
    points.forEach((pt, i) => {
      if (pt[0] >= dx0 && pt[0] <= dx1 && pt[1] >= dy0 && pt[1] <= dy1) selected.add(i)
    })
    onSelect([...selected])
    render()
  })

  renderAxes()
  render()

  return {
    setSelected: (i) => {
      selected = new Set([i])
      render()
    },
    focusPoint: (i) => {
      selected = new Set([i])
      render()
      const cx = xScale(points[i][0])
      const cy = yScale(points[i][1])
      const t = d3.zoomTransform(svg.node())
      svg.transition().duration(350).call(
        zoom.transform,
        d3.zoomIdentity.translate(innerW / 2 - t.k * cx, innerH / 2 - t.k * cy).scale(t.k)
      )
    }
  }
}
