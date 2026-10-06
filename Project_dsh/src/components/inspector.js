/**
 * 计算检查器（I01 计算显微镜）。
 * 把「一个数字 → 所属步骤 → 公式 → 输入数值 → 形状/索引 → 上游来源」以可读形式展示。
 * 全部用 textContent 写入，用户输入不会拼进可执行 HTML。
 */

/**
 * @param {HTMLElement} container
 * @param {Array} steps 每项 { title, value, formula, inputs?: [{label, value}], meta?: string }
 */
export function renderInspector(container, steps) {
  container.innerHTML = ''
  const wrap = document.createElement('div')
  wrap.className = 'inspector'

  const head = document.createElement('div')
  head.className = 'inspector-head'
  head.textContent = '计算检查器'
  wrap.appendChild(head)

  for (const step of steps) {
    const sec = document.createElement('section')
    sec.className = 'inspector-step'

    const titleRow = document.createElement('div')
    titleRow.className = 'inspector-row'
    const title = document.createElement('span')
    title.className = 'inspector-title'
    title.textContent = step.title
    const value = document.createElement('span')
    value.className = 'inspector-value num'
    value.textContent = step.value
    titleRow.appendChild(title)
    titleRow.appendChild(value)
    sec.appendChild(titleRow)

    if (step.formula) {
      const formula = document.createElement('div')
      formula.className = 'inspector-formula num'
      formula.textContent = step.formula
      sec.appendChild(formula)
    }

    if (step.inputs && step.inputs.length) {
      const ul = document.createElement('ul')
      ul.className = 'inspector-inputs'
      for (const input of step.inputs) {
        const li = document.createElement('li')
        const label = document.createElement('span')
        label.textContent = input.label
        const val = document.createElement('span')
        val.className = 'num'
        val.textContent = input.value
        li.appendChild(label)
        li.appendChild(val)
        ul.appendChild(li)
      }
      sec.appendChild(ul)
    }

    if (step.meta) {
      const meta = document.createElement('div')
      meta.className = 'inspector-meta'
      meta.textContent = step.meta
      sec.appendChild(meta)
    }

    wrap.appendChild(sec)
  }

  container.innerHTML = ''
  container.appendChild(wrap)
}
