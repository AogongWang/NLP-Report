/**
 * 本地导出工具：通过 Blob + <a download> 触发浏览器下载。
 * 不依赖服务器，file:// 下也可用（I04 实验笔记导出）。
 */

export function downloadText(filename, text, mime = 'application/json') {
  const blob = new Blob([text], { type: `${mime};charset=utf-8` })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export function downloadJson(filename, obj) {
  downloadText(filename, JSON.stringify(obj, null, 2))
}
