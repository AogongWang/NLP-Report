/**
 * 数值稳定的数学核心 —— 全部算法模块共用的底层数学。
 * 采用双精度；softmax / log-sum-exp / sigmoid / 交叉熵均做稳定处理。
 * 该文件禁止操作 DOM，供浏览器与 Node 测试共享。
 */

/** 数值夹取 */
export function clamp(x, lo, hi) {
  return Math.min(hi, Math.max(lo, x))
}

/** 稳定 softmax：softmax(x)_i = exp(x_i - max) / Σ exp(x_j - max) */
export function softmax(logits) {
  if (!Array.isArray(logits) || logits.length === 0) return []
  const max = Math.max(...logits)
  // 全 -Infinity 时（全部被掩码屏蔽），避免 NaN
  if (max === -Infinity) return logits.map(() => 0)
  const exps = logits.map((x) => Math.exp(x - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sum)
}

/** 稳定 log-sum-exp：log Σ exp(x_i) */
export function logSumExp(arr) {
  if (!Array.isArray(arr) || arr.length === 0) return -Infinity
  const max = Math.max(...arr)
  if (max === -Infinity) return -Infinity
  return max + Math.log(arr.reduce((s, x) => s + Math.exp(x - max), 0))
}

/** 稳定 sigmoid（正负分支避免 exp 溢出） */
export function sigmoid(x) {
  if (x >= 0) {
    const z = Math.exp(-x)
    return 1 / (1 + z)
  }
  const z = Math.exp(x)
  return z / (1 + z)
}

/** 二分类交叉熵（natural log）。label ∈ {0, 1} */
export function crossEntropy(probTrue, label) {
  const p = clamp(probTrue, 1e-12, 1 - 1e-12)
  return label === 1 ? -Math.log(p) : -Math.log(1 - p)
}

/** 多分类交叉熵（natural log）。prob 为各类概率分布，label 为真实类下标 */
export function categoricalCrossEntropy(prob, label) {
  const p = clamp(prob[label], 1e-12, 1)
  return -Math.log(p)
}

/** 向量点积 */
export function dot(a, b) {
  const n = Math.min(a.length, b.length)
  let s = 0
  for (let i = 0; i < n; i++) s += a[i] * b[i]
  return s
}

/** L2 范数 */
export function norm(a) {
  return Math.sqrt(dot(a, a))
}

/** 余弦相似度；零向量返回 0（与课程约定一致） */
export function cosineSimilarity(a, b) {
  const na = norm(a)
  const nb = norm(b)
  if (na === 0 || nb === 0) return 0
  return dot(a, b) / (na * nb)
}

/** 数值稳定的 scaled 点积注意力分数：q·k / sqrt(dk) */
export function scaledDotScore(q, k) {
  return dot(q, k) / Math.sqrt(q.length)
}

/** 矩阵乘法 A[m×n] @ B[n×p] → C[m×p]（二维数组表示矩阵） */
export function matmul(A, B) {
  const m = A.length
  const n = B.length
  if (n === 0) return []
  const p = B[0].length
  const C = Array.from({ length: m }, () => new Array(p).fill(0))
  for (let i = 0; i < m; i++) {
    for (let k = 0; k < n; k++) {
      const a = A[i][k]
      if (a !== 0) {
        for (let j = 0; j < p; j++) C[i][j] += a * B[k][j]
      }
    }
  }
  return C
}

/** 矩阵转置 A[m×n] → Aᵀ[n×m] */
export function transpose(A) {
  const m = A.length
  if (m === 0) return []
  const n = A[0].length
  const T = Array.from({ length: n }, () => new Array(m))
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) T[j][i] = A[i][j]
  return T
}
