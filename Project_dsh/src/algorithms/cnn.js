import { softmax, dot } from '../core/math.js'

/**
 * M03 Text-CNN 前向：文本 → Embedding → 多窗口卷积 → ReLU → max-over-time 池化 → 拼接 → 分类得分。
 * 卷积核覆盖整个 embedding 维度、沿序列方向滑动（1D 卷积，不是图像二维卷积）。
 * 短文本：有效窗口数为 0 时该卷积核贡献 0（不引入纯 padding 窗口主导分类）。
 */

/** token id 序列 → embedding 矩阵 [seq_len, dim] */
export function embed(tokens, embeddingMatrix) {
  return tokens.map((id) => embeddingMatrix[id] || embeddingMatrix[0])
}

/** 一维卷积（沿序列滑动），kernel 形状 [window, dim]，返回特征图 [seq_len - window + 1] */
export function conv1d(emb, kernel) {
  const window = kernel.length
  const seqLen = emb.length
  const outLen = seqLen - window + 1
  const out = []
  for (let i = 0; i < outLen; i++) {
    let s = 0
    for (let w = 0; w < window; w++) {
      s += dot(emb[i + w], kernel[w])
    }
    out.push(s)
  }
  return out
}

export function relu(x) {
  return Math.max(0, x)
}
export function reluArr(arr) {
  return arr.map(relu)
}

/** max-over-time 池化：空特征图返回 0 */
export function maxPool(arr) {
  return arr.length ? Math.max(...arr) : 0
}

/**
 * 完整 Text-CNN 前向。
 * @param {number[]} tokens token id 序列
 * @param {number[][]} embeddingMatrix 词表 embedding [vocab, dim]
 * @param {number[][][]} kernels 多个卷积核，每个 [window, dim]
 * @param {number[][]} fcWeights 分类层 [classes, n_kernels]
 * @param {number[]} fcBias [classes]
 * @returns {{pooled, scores, probs, featureMaps}} probs 为 softmax 概率
 */
export function textCNN(tokens, embeddingMatrix, kernels, fcWeights, fcBias) {
  const emb = embed(tokens, embeddingMatrix)
  const featureMaps = kernels.map((k) => reluArr(conv1d(emb, k)))
  const pooled = featureMaps.map((fm) => maxPool(fm))
  const scores = fcWeights.map((row, c) => {
    let s = row.reduce((acc, w, j) => acc + w * pooled[j], 0)
    if (fcBias) s += fcBias[c] || 0
    return s
  })
  return { pooled, scores, probs: softmax(scores), featureMaps }
}
