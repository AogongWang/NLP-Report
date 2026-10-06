import { sigmoid, clamp } from '../core/math.js'
import { createRng, shuffle } from '../core/random.js'

/**
 * M06 损失与优化：二维参数 Logistic 回归。
 * 损失 = 均值二分类交叉熵（natural log）；梯度解析计算；支持真实梯度下降轨迹与二维损失曲面。
 * 曲线每个点都来自真实迭代，不使用 exp(-t) 等伪造曲线。
 */

function predict2D(w, x) {
  return sigmoid(w[0] * x[0] + w[1] * x[1])
}

/** 均值交叉熵损失 */
export function logisticLoss2D(w, X, y) {
  const n = X.length
  if (n === 0) return NaN
  let total = 0
  for (let i = 0; i < n; i++) {
    const p = clamp(predict2D(w, X[i]), 1e-12, 1 - 1e-12)
    total += y[i] === 1 ? -Math.log(p) : -Math.log(1 - p)
  }
  return total / n
}

/** 解析梯度 dL/dw = (1/n) Σ (p_i - y_i) x_i */
export function logisticGrad2D(w, X, y) {
  const n = X.length
  let g0 = 0
  let g1 = 0
  for (let i = 0; i < n; i++) {
    const p = predict2D(w, X[i])
    const err = p - y[i]
    g0 += err * X[i][0]
    g1 += err * X[i][1]
  }
  return [g0 / n, g1 / n]
}

/** 二维损失曲面 Z[i][j] = loss(w0s[j], w1s[i])（用于等高线） */
export function lossSurface2D(X, y, w0s, w1s) {
  return w1s.map((w1) => w0s.map((w0) => logisticLoss2D([w0, w1], X, y)))
}

/**
 * 小批量梯度下降，记录每轮（epoch）真实损失与参数轨迹。
 * batchSize 为 null 或 ≥ N 时等价于全批量；seed 控制每轮样本打乱顺序。
 * 返回 { weights, trajectory, losses, divergedAt }。
 */
export function trainLogistic2D(X, y, { lr = 0.1, epochs = 100, init = [0, 0], batchSize = null, seed = 42 } = {}) {
  let w = [...init]
  const trajectory = [[...w]]
  const losses = [logisticLoss2D(w, X, y)]
  let divergedAt = null
  const N = X.length
  const bs = batchSize == null || batchSize >= N ? N : Math.max(1, batchSize)
  const rng = createRng(seed)
  const allIdx = Array.from({ length: N }, (_, i) => i)

  for (let e = 0; e < epochs; e++) {
    const order = bs === N ? allIdx : shuffle(rng, allIdx)
    for (let start = 0; start < N; start += bs) {
      const idx = order.slice(start, start + bs)
      const g = logisticGrad2D(w, idx.map((i) => X[i]), idx.map((i) => y[i]))
      w = [w[0] - lr * g[0], w[1] - lr * g[1]]
    }
    const loss = logisticLoss2D(w, X, y)
    if (!Number.isFinite(loss)) {
      divergedAt = e + 1
      losses.push(Infinity)
      trajectory.push([...w])
      break
    }
    losses.push(loss)
    trajectory.push([...w])
  }
  return { weights: w, trajectory, losses, divergedAt }
}

/** 线性可分二维二分类数据集（可复现，用固定 seed） */
export function makeBinaryDataset(n = 100, seed = 42) {
  const rng = createRng(seed)
  const X = []
  const y = []
  for (let i = 0; i < n; i++) {
    const x0 = rng() * 2 - 1
    const x1 = rng() * 2 - 1
    X.push([x0, x1])
    y.push(x0 + x1 > 0 ? 1 : 0)
  }
  return { X, y }
}
