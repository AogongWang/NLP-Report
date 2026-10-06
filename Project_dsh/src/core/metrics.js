/**
 * 分类指标（M05）：混淆矩阵、准确率、精确率、召回率、F1、Macro-F1。
 * 全部由真实标签与预测计算，明确零分母规则（零分母返回 0）。
 */

export function confusionMatrix(yTrue, yPred, labels) {
  const idx = new Map(labels.map((l, i) => [l, i]))
  const n = labels.length
  const matrix = Array.from({ length: n }, () => new Array(n).fill(0))
  for (let i = 0; i < yTrue.length; i++) {
    const r = idx.get(yTrue[i])
    const c = idx.get(yPred[i])
    if (r !== undefined && c !== undefined) matrix[r][c] += 1
  }
  return { matrix, labels }
}

export function accuracy(yTrue, yPred) {
  if (yTrue.length === 0) return 0
  let correct = 0
  for (let i = 0; i < yTrue.length; i++) if (yTrue[i] === yPred[i]) correct += 1
  return correct / yTrue.length
}

export function precisionRecallF1(yTrue, yPred, label) {
  let tp = 0
  let fp = 0
  let fn = 0
  for (let i = 0; i < yTrue.length; i++) {
    const t = yTrue[i] === label
    const p = yPred[i] === label
    if (t && p) tp += 1
    else if (!t && p) fp += 1
    else if (t && !p) fn += 1
  }
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0
  return { precision, recall, f1 }
}

export function macroF1(yTrue, yPred, labels) {
  if (labels.length === 0) return 0
  const f1s = labels.map((l) => precisionRecallF1(yTrue, yPred, l).f1)
  return f1s.reduce((a, b) => a + b, 0) / f1s.length
}
