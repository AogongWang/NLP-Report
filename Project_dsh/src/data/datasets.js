/**
 * 数据集解析与校验（§8.1）。
 * 支持 CSV / JSONL / TXT；CSV 正确处理引号、转义、内嵌分隔符与换行。
 * JSONL 逐行报错（带行号）；提供标签分布 / 长度分布 / 重复检查。
 */

/** 解析 CSV（RFC 4180 风格）。返回二维数组（行 × 列）。 */
export function parseCSV(text, { delimiter = ',', skipEmptyLines = true } = {}) {
  const rows = []
  let row = []
  let field = ''
  let inQuotes = false
  const n = text.length
  let i = 0
  const pushRow = () => {
    row.push(field)
    field = ''
    if (!skipEmptyLines || row.some((x) => x !== '')) rows.push(row)
    row = []
  }
  while (i < n) {
    const c = text[i]
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"'
          i += 2
          continue
        }
        inQuotes = false
        i += 1
        continue
      }
      field += c
      i += 1
      continue
    }
    if (c === '"') {
      inQuotes = true
      i += 1
      continue
    }
    if (c === delimiter) {
      row.push(field)
      field = ''
      i += 1
      continue
    }
    if (c === '\n') {
      pushRow()
      i += 1
      continue
    }
    if (c === '\r') {
      if (text[i + 1] === '\n') i += 1
      pushRow()
      i += 1
      continue
    }
    field += c
    i += 1
  }
  if (field !== '' || row.length > 0) pushRow()
  return rows
}

/** 解析 JSONL。返回 { records, errors:[{line, message}] }。 */
export function parseJSONL(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== '')
  const records = []
  const errors = []
  lines.forEach((line, idx) => {
    try {
      records.push(JSON.parse(line))
    } catch (e) {
      errors.push({ line: idx + 1, message: e.message })
    }
  })
  return { records, errors }
}

/** 解析 TXT 语料：每个非空行为一个句子/文档（词向量训练常用「一行一句」格式）。 */
export function parseTXT(text) {
  return text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l !== '')
}

/**
 * 分析样本：标签分布、长度分布、重复检查。
 * @param {Array} samples 每项含 text/label 字段（字段名可配）
 */
export function analyzeSamples(samples, { textField = 'text', labelField = 'label' } = {}) {
  const labelCount = new Map()
  const lengths = []
  const seen = new Map()
  const duplicatePairs = []
  samples.forEach((s, i) => {
    const label = s[labelField] ?? '(无标签)'
    labelCount.set(label, (labelCount.get(label) || 0) + 1)
    const len = (s[textField] || '').length
    lengths.push(len)
    const key = s[textField]
    if (seen.has(key)) duplicatePairs.push([seen.get(key), i])
    else seen.set(key, i)
  })
  const sum = lengths.reduce((a, b) => a + b, 0)
  return {
    count: samples.length,
    labelDistribution: Object.fromEntries(labelCount),
    lengthStats:
      lengths.length > 0
        ? {
            min: Math.min(...lengths),
            max: Math.max(...lengths),
            avg: Math.round((sum / lengths.length) * 100) / 100
          }
        : { min: 0, max: 0, avg: 0 },
    duplicatePairs
  }
}

/** 校验样本结构，返回 { valid, errors }。 */
export function validateSamples(samples, { textField = 'text' } = {}) {
  const errors = []
  samples.forEach((s, i) => {
    if (s == null || typeof s !== 'object') {
      errors.push({ index: i, message: '非对象' })
      return
    }
    if (typeof s[textField] !== 'string' || s[textField].trim() === '') {
      errors.push({ index: i, message: `缺少或为空字段 "${textField}"` })
    }
  })
  return { valid: errors.length === 0, errors }
}
