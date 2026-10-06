import { sigmoid, cosineSimilarity, dot } from '../core/math.js'
import { createRng } from '../core/random.js'

/**
 * M01 词向量：CBOW / Skip-gram 负采样训练，GloVe 稀疏共现训练，及 PCA 二维投影。
 * 负采样使用词频 0.75 次幂的噪声分布与固定 seed，可复现。
 */

/** 构建词表：返回 { word2id, id2word, freq, vocabSize } */
export function buildVocab(corpus, minCount = 1) {
  const freq = new Map()
  for (const sentence of corpus) {
    for (const tok of sentence) freq.set(tok, (freq.get(tok) || 0) + 1)
  }
  const words = [...freq.entries()].filter(([, c]) => c >= minCount).sort((a, b) => b[1] - a[1])
  const word2id = new Map(words.map(([w], i) => [w, i]))
  const id2word = words.map(([w]) => w)
  return { word2id, id2word, freq, vocabSize: words.length }
}

/** 共现矩阵（GloVe 用）。返回稀疏记录 [{i, j, count}] 与矩阵。 */
export function cooccurrence(corpus, word2id, window = 2) {
  const V = word2id.size
  const cooc = Array.from({ length: V }, () => new Array(V).fill(0))
  for (const sentence of corpus) {
    const ids = sentence.map((w) => word2id.get(w)).filter((i) => i !== undefined)
    for (let i = 0; i < ids.length; i++) {
      for (let j = Math.max(0, i - window); j <= Math.min(ids.length - 1, i + window); j++) {
        if (i !== j) cooc[ids[i]][ids[j]] += 1
      }
    }
  }
  const sparse = []
  for (let i = 0; i < V; i++) for (let j = 0; j < V; j++) if (cooc[i][j] > 0) sparse.push({ i, j, count: cooc[i][j] })
  return { cooc, sparse }
}

/** Skip-gram 负采样训练。返回 { W, Wout, word2id, id2word, losses } */
export function trainSkipGram(corpus, { dim = 4, window = 2, negative = 3, lr = 0.05, epochs = 3, seed = 42 } = {}) {
  const { word2id, id2word, freq, vocabSize } = buildVocab(corpus)
  const rng = createRng(seed)
  const total = [...freq.values()].reduce((a, b) => a + Math.pow(b, 0.75), 0)
  const noiseProb = [...freq.values()].map((c) => Math.pow(c, 0.75) / total)

  const W = Array.from({ length: vocabSize }, () => Array.from({ length: dim }, () => (rng() - 0.5) / dim))
  const Wout = Array.from({ length: vocabSize }, () => Array.from({ length: dim }, () => 0))
  const losses = []

  const updatePair = (center, target, label) => {
    const vc = W[center]
    const uo = Wout[target]
    const p = sigmoid(dot(vc, uo))
    const err = p - label
    for (let d = 0; d < dim; d++) {
      const gv = err * uo[d]
      const gu = err * vc[d]
      vc[d] -= lr * gv
      uo[d] -= lr * gu
    }
    return label === 1 ? -Math.log(p + 1e-12) : -Math.log(1 - p + 1e-12)
  }

  for (let ep = 0; ep < epochs; ep++) {
    let epLoss = 0
    for (const sentence of corpus) {
      const ids = sentence.map((w) => word2id.get(w)).filter((i) => i !== undefined)
      for (let c = 0; c < ids.length; c++) {
        for (let o = Math.max(0, c - window); o <= Math.min(ids.length - 1, c + window); o++) {
          if (o === c) continue
          epLoss += updatePair(ids[c], ids[o], 1)
          for (let n = 0; n < negative; n++) {
            let neg = 0
            let r = rng()
            let acc = 0
            for (let k = 0; k < noiseProb.length; k++) {
              acc += noiseProb[k]
              if (r <= acc) {
                neg = k
                break
              }
            }
            if (neg !== ids[o]) epLoss += updatePair(ids[c], neg, 0)
          }
        }
      }
    }
    losses.push(epLoss)
  }
  return { W, Wout, word2id, id2word, losses }
}

/** CBOW 负采样训练：上下文向量求平均预测中心词。 */
export function trainCBOW(corpus, { dim = 4, window = 2, negative = 3, lr = 0.05, epochs = 3, seed = 42 } = {}) {
  const { word2id, id2word, freq, vocabSize } = buildVocab(corpus)
  const rng = createRng(seed)
  const total = [...freq.values()].reduce((a, b) => a + Math.pow(b, 0.75), 0)
  const noiseProb = [...freq.values()].map((c) => Math.pow(c, 0.75) / total)
  const W = Array.from({ length: vocabSize }, () => Array.from({ length: dim }, () => (rng() - 0.5) / dim))
  const Wout = Array.from({ length: vocabSize }, () => Array.from({ length: dim }, () => 0))
  const losses = []

  for (let ep = 0; ep < epochs; ep++) {
    let epLoss = 0
    for (const sentence of corpus) {
      const ids = sentence.map((w) => word2id.get(w)).filter((i) => i !== undefined)
      for (let c = 0; c < ids.length; c++) {
        const ctx = []
        for (let o = Math.max(0, c - window); o <= Math.min(ids.length - 1, c + window); o++) {
          if (o !== c) ctx.push(ids[o])
        }
        if (ctx.length === 0) continue
        // 上下文向量平均
        const h = new Array(dim).fill(0)
        for (const cid of ctx) for (let d = 0; d < dim; d++) h[d] += W[cid][d] / ctx.length
        // 正样本
        const p = sigmoid(dot(h, Wout[ids[c]]))
        const err = p - 1
        epLoss += -Math.log(p + 1e-12)
        for (const cid of ctx) for (let d = 0; d < dim; d++) W[cid][d] -= lr * err * Wout[ids[c]][d] / ctx.length
        for (let d = 0; d < dim; d++) Wout[ids[c]][d] -= lr * err * h[d]
        // 负样本
        for (let n = 0; n < negative; n++) {
          let neg = 0
          let r = rng()
          let acc = 0
          for (let k = 0; k < noiseProb.length; k++) {
            acc += noiseProb[k]
            if (r <= acc) {
              neg = k
              break
            }
          }
          if (neg === ids[c]) continue
          const pn = sigmoid(dot(h, Wout[neg]))
          const en = pn
          epLoss += -Math.log(1 - pn + 1e-12)
          for (const cid of ctx) for (let d = 0; d < dim; d++) W[cid][d] -= lr * en * Wout[neg][d] / ctx.length
          for (let d = 0; d < dim; d++) Wout[neg][d] -= lr * en * h[d]
        }
      }
    }
    losses.push(epLoss)
  }
  return { W, Wout, word2id, id2word, losses }
}

/**
 * GloVe：用稀疏共现记录做加权最小二乘梯度下降。
 * 目标 J = Σ f(X_ij)(w_i·w̃_j + b_i + b̃_j − log X_ij)²，f(x) = (x/x_max)^0.75（截断）。
 */
export function trainGloVe(sparse, vocabSize, { dim = 4, lr = 0.05, epochs = 10, xMax = 10, seed = 42 } = {}) {
  const rng = createRng(seed)
  const W = Array.from({ length: vocabSize }, () => Array.from({ length: dim }, () => (rng() - 0.5) / dim))
  const Wt = Array.from({ length: vocabSize }, () => Array.from({ length: dim }, () => (rng() - 0.5) / dim))
  const b = new Array(vocabSize).fill(0)
  const bt = new Array(vocabSize).fill(0)
  const losses = []
  for (let ep = 0; ep < epochs; ep++) {
    let totalLoss = 0
    for (const { i, j, count } of sparse) {
      const w = count < xMax ? Math.pow(count / xMax, 0.75) : 1
      const diff = dot(W[i], Wt[j]) + b[i] + bt[j] - Math.log(count)
      totalLoss += w * diff * diff
      for (let d = 0; d < dim; d++) {
        const gi = w * diff * Wt[j][d]
        const gj = w * diff * W[i][d]
        W[i][d] -= lr * gi
        Wt[j][d] -= lr * gj
      }
      b[i] -= lr * w * diff
      bt[j] -= lr * w * diff
    }
    losses.push(totalLoss)
  }
  // 最终向量 = W + Wt（GloVe 惯例）
  const vectors = W.map((row, i) => row.map((x, d) => x + Wt[i][d]))
  return { vectors, losses }
}

/** 二维 PCA 投影（用于词向量散点图） */
export function pca2D(vectors) {
  const n = vectors.length
  if (n === 0) return []
  const dim = vectors[0].length
  const mean = new Array(dim).fill(0)
  for (const v of vectors) for (let d = 0; d < dim; d++) mean[d] += v[d] / n
  const centered = vectors.map((v) => v.map((x, d) => x - mean[d]))
  const cov = Array.from({ length: dim }, () => new Array(dim).fill(0))
  for (const v of centered) for (let i = 0; i < dim; i++) for (let j = 0; j < dim; j++) cov[i][j] += (v[i] * v[j]) / n

  const powerIteration = (C) => {
    let u = new Array(dim).fill(1 / Math.sqrt(dim))
    for (let iter = 0; iter < 100; iter++) {
      const v = C.map((row) => dot(row, u))
      const norm = Math.sqrt(dot(v, v)) || 1
      u = v.map((x) => x / norm)
    }
    return u
  }
  const e1 = powerIteration(cov)
  // 收缩得到第二主成分
  const e1e1t = e1.map((a) => e1.map((b2) => a * b2))
  const cov2 = cov.map((row, i) => row.map((x, j) => x - e1e1t[i][j]))
  const e2 = powerIteration(cov2)

  return centered.map((v) => [dot(v, e1), dot(v, e2)])
}

/** 余弦相似度（复用 math） */
export { cosineSimilarity }
