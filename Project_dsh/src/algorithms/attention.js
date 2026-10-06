import { softmax, matmul, transpose } from '../core/math.js'

/**
 * 注意力前向计算（M04）。
 * 三种打分函数 + mask + softmax + 加权输出，全部为真实向量/矩阵计算。
 * 约定：
 *   Q: [n_q, d_k]   K: [n_k, d_k]   V: [n_k, d_v]
 *   scores/weights: [n_q, n_k]
 *   mask: [n_q, n_k] 布尔矩阵，false 表示屏蔽；可为 null。
 */

/** 点积打分：score(i,j) = q_i · k_j */
export function dotProductScores(Q, K) {
  return matmul(Q, transpose(K))
}

/** 缩放点积打分：score(i,j) = (q_i · k_j) / sqrt(d_k) */
export function scaledDotScores(Q, K) {
  const dk = Q[0].length
  const scale = Math.sqrt(dk)
  return matmul(Q, transpose(K)).map((row) => row.map((s) => s / scale))
}

/** 加性打分：score(i,j) = v^T tanh(W_q q_i + W_k k_j + b) */
export function additiveScores(Q, K, { Wq, Wk, b, v }) {
  const Qp = matmul(Q, Wq) // [n_q, d_h]
  const Kp = matmul(K, Wk) // [n_k, d_h]
  const nq = Qp.length
  const nk = Kp.length
  const dh = v.length
  const scores = Array.from({ length: nq }, () => new Array(nk))
  for (let i = 0; i < nq; i++) {
    for (let j = 0; j < nk; j++) {
      let s = 0
      for (let h = 0; h < dh; h++) {
        s += v[h] * Math.tanh(Qp[i][h] + Kp[j][h] + b[h])
      }
      scores[i][j] = s
    }
  }
  return scores
}

/** 应用掩码：mask[i][j] === false 的位置置为 -Infinity */
export function applyMask(scores, mask) {
  if (!mask) return scores
  return scores.map((row, i) => row.map((s, j) => (mask[i][j] === false ? -Infinity : s)))
}

/** 每行 softmax 得注意力权重（全屏蔽行返回全 0，不产生 NaN） */
export function attentionWeights(scores) {
  return scores.map((row) => softmax(row))
}

/** 加权输出 O = A @ V */
export function attentionOutput(weights, V) {
  return matmul(weights, V)
}

/**
 * 完整注意力前向。
 * @param {number[][]} Q 查询
 * @param {number[][]} K 键
 * @param {number[][]} V 值
 * @param {object} opts { scoreFn, additive, mask }
 *   - scoreFn: 打分函数（默认 scaledDotScores），additive 存在时优先用加性
 *   - additive: { Wq, Wk, b, v } 加性注意力参数
 *   - mask: [n_q, n_k] 布尔矩阵
 * @returns {{scores, masked, weights, output, invalidRows}}
 *   scores 原始分数；masked 掩码后分数；weights 掩码后权重；output 加权输出；
 *   invalidRows 为「整行全屏蔽」的行下标（权重按约定为全 0）。
 */
export function attention(Q, K, V, { scoreFn = scaledDotScores, additive = null, mask = null } = {}) {
  const scores = additive ? additiveScores(Q, K, additive) : scoreFn(Q, K)
  const masked = applyMask(scores, mask)
  const weights = attentionWeights(masked)
  const output = attentionOutput(weights, V)
  const invalidRows = weights
    .map((row, i) => (row.every((w) => w === 0) ? i : -1))
    .filter((i) => i !== -1)
  return { scores, masked, weights, output, invalidRows }
}
