/**
 * 可复用热力图组件（HTML 表格实现，适用于小规模教学矩阵）。
 * 大规模场景后续换 Canvas，接口保持不变（rowLabels/colLabels/values/onCellClick）。
 */

/** 权重顺序色带（0→白，1→强调色）；返回背景色与文字深浅 */
export function weightColor(w) {
  const c = Math.max(0, Math.min(1, w))
  const r = Math.round(255 + (8 - 255) * c)
  const g = Math.round(255 + (127 - 255) * c)
  const b = Math.round(255 + (140 - 255) * c)
  return { bg: `rgb(${r},${g},${b})`, dark: c < 0.55 }
}

/** 发散色带（-1→蓝，0→白，+1→红），用于以 0 为中心的数值（隐状态/梯度等） */
export function divergingColor(v) {
  const t = Math.max(-1, Math.min(1, v))
  const a = Math.abs(t)
  if (t >= 0) {
    const r = Math.round(255 + (200 - 255) * a)
    const g = Math.round(255 + (72 - 255) * a)
    const b = Math.round(255 + (72 - 255) * a)
    return { bg: `rgb(${r},${g},${b})`, dark: a < 0.45 }
  }
  const r = Math.round(255 + (66 - 255) * a)
  const g = Math.round(255 + (120 - 255) * a)
  const b = Math.round(255 + (220 - 255) * a)
  return { bg: `rgb(${r},${g},${b})`, dark: a < 0.45 }
}

/**
 * @param {HTMLElement} container
 * @param {object} opts { rowLabels, colLabels, values, format, color, onCellClick, onShiftClick, highlightCol, highlightRows, masked, selected }
 */
export function renderHeatmap(container, { rowLabels, colLabels, values, format, color, onCellClick, onShiftClick, highlightCol = -1, highlightRows = [], masked = null, selected = null }) {
  const table = document.createElement('table')
  table.className = 'heatmap'
  table.setAttribute('role', 'grid')

  const thead = document.createElement('thead')
  const hr = document.createElement('tr')
  const corner = document.createElement('th')
  corner.textContent = 'Q \\ K'
  hr.appendChild(corner)
  for (const c of colLabels) {
    const th = document.createElement('th')
    th.textContent = c
    th.scope = 'col'
    hr.appendChild(th)
  }
  thead.appendChild(hr)
  table.appendChild(thead)

  const tbody = document.createElement('tbody')
  values.forEach((row, i) => {
    const tr = document.createElement('tr')
    const th = document.createElement('th')
    th.textContent = rowLabels[i]
    th.scope = 'row'
    tr.appendChild(th)
    row.forEach((v, j) => {
      const td = document.createElement('td')
      const isMasked = masked ? masked[i][j] === false : false
      td.textContent = isMasked ? '×' : format(v)
      const { bg, dark } = color(v)
      td.style.background = isMasked ? 'var(--danger)' : bg
      td.style.color = isMasked ? '#ffffff' : dark ? '#0b1f28' : '#ffffff'
      if (isMasked) td.classList.add('masked')
      if (selected && selected[0] === i && selected[1] === j) td.classList.add('selected')
      if (j === highlightCol) td.classList.add('hl-col')
      if (highlightRows.includes(i)) td.classList.add('hl-row')
      td.tabIndex = 0
      td.setAttribute('role', 'gridcell')
      td.setAttribute('aria-label', `${rowLabels[i]} → ${colLabels[j]}: ${format(v)}`)
      const activate = (event) => {
        if (event.shiftKey && onShiftClick) onShiftClick(i, j)
        else if (onCellClick) onCellClick(i, j)
      }
      td.addEventListener('click', activate)
      td.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          activate(e)
        }
      })
      tr.appendChild(td)
    })
    tbody.appendChild(tr)
  })
  table.appendChild(tbody)

  container.innerHTML = ''
  container.appendChild(table)
}
