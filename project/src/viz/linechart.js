import * as d3 from 'd3'

/**
 * 可复用折线图（SVG + D3 坐标轴）。
 * 支持点击数据点回调、选中高亮、非有限值断线。
 */
export function renderLineChart(container, opts = {}) {
  const {
    data,
    width = 640,
    height = 300,
    xLabel = '',
    yLabel = '',
    format = (d) => (Number.isFinite(d) ? d.toFixed(3) : '∞'),
    onPointClick = null,
    selected = -1
  } = opts

  const indexed = data.map((d, i) => ({ x: d.x, y: d.y, i }))
  const margin = { top: 16, right: 16, bottom: 40, left: 52 }
  const innerW = width - margin.left - margin.right
  const innerH = height - margin.top - margin.bottom

  const xs = indexed.map((d) => d.x)
  const finiteYs = indexed.filter((d) => Number.isFinite(d.y)).map((d) => d.y)
  const xScale = d3.scaleLinear().domain([d3.min(xs) ?? 0, d3.max(xs) ?? 1]).range([0, innerW])
  let yMin = d3.min(finiteYs) ?? 0
  let yMax = d3.max(finiteYs) ?? 1
  const pad = (yMax - yMin) * 0.06 || 0.1
  if (yMin > 0) yMin = 0
  const yScale = d3.scaleLinear().domain([yMin - pad, yMax + pad]).range([innerH, 0])

  container.innerHTML = ''
  const svg = d3.select(container).append('svg')
    .attr('width', width)
    .attr('height', height)
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('role', 'img')

  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

  g.append('g').attr('transform', `translate(0,${innerH})`)
    .call(d3.axisBottom(xScale).ticks(Math.min(10, xs.length)).tickFormat((d) => String(d)))
  g.append('g').call(d3.axisLeft(yScale).ticks(6))

  g.append('text').attr('x', innerW / 2).attr('y', innerH + 34)
    .attr('text-anchor', 'middle').attr('class', 'axis-label').text(xLabel)
  g.append('text').attr('transform', 'rotate(-90)').attr('x', -innerH / 2).attr('y', -40)
    .attr('text-anchor', 'middle').attr('class', 'axis-label').text(yLabel)

  const line = d3.line()
    .defined((d) => Number.isFinite(d.y))
    .x((d) => xScale(d.x))
    .y((d) => yScale(d.y))
  g.append('path').datum(indexed).attr('fill', 'none')
    .attr('stroke', 'var(--accent)').attr('stroke-width', 1.6).attr('d', line)

  g.selectAll('circle.point').data(indexed).join('circle')
    .attr('class', 'point')
    .attr('cx', (d) => xScale(d.x))
    .attr('cy', (d) => (Number.isFinite(d.y) ? yScale(d.y) : -10))
    .attr('r', (d) => (d.i === selected ? 5 : 3))
    .attr('fill', (d) => (d.i === selected ? 'var(--danger)' : 'var(--accent)'))
    .style('cursor', onPointClick ? 'pointer' : 'default')
    .on('click', (event, d) => onPointClick && onPointClick(d.i))
    .append('title')
    .text((d) => `步 ${d.x}: ${format(d.y)}`)

  return svg
}
