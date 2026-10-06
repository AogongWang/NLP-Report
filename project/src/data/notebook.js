/**
 * 实验记录 notebook（I04）：命名、保存、查看、删除、导出、导入。
 * 存储用 localStorage（文件协议下可用），带配额/受限异常降级到内存。
 */

const KEY = 'tensorscope.notebook.records'

// 内存降级缓存（localStorage 不可用时）
let memoryFallback = null

function read() {
  try {
    const raw = localStorage.getItem(KEY)
    if (raw == null) return []
    const arr = JSON.parse(raw)
    return Array.isArray(arr) ? arr : []
  } catch {
    return memoryFallback || []
  }
}

function persist(records) {
  try {
    localStorage.setItem(KEY, JSON.stringify(records))
    memoryFallback = null
  } catch {
    memoryFallback = records // 降级到内存
  }
}

/** 保存一条实验记录（返回带 id/savedAt 的完整记录） */
export function saveRecord(record) {
  const records = read()
  const full = {
    id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
    savedAt: new Date().toISOString(),
    ...record
  }
  records.unshift(full)
  persist(records)
  return full
}

/** 列出全部记录（新的在前） */
export function listRecords() {
  return read()
}

/** 删除一条记录 */
export function deleteRecord(id) {
  persist(read().filter((r) => r.id !== id))
}

/** 清空全部记录 */
export function clearRecords() {
  persist([])
}

/** 导出为 JSON 文本 */
export function exportRecordsText() {
  return JSON.stringify(read(), null, 2)
}

/** 导入 JSON 文本（返回导入条数） */
export function importRecordsText(text) {
  const arr = JSON.parse(text)
  if (!Array.isArray(arr)) throw new Error('不是有效的记录数组')
  persist(arr)
  return arr.length
}
