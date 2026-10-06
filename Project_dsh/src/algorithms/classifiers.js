import { dot } from '../core/math.js'
import { createRng } from '../core/random.js'

/**
 * M05 分类器：多项式朴素贝叶斯（对数空间、加一平滑）与线性 SVM（hinge + L2）。
 * 朴素贝叶斯输出真实对数概率；SVM 输出决策分数（不是概率）。
 */

/** 基础可复现切分：有空格按空格预分词，否则字符级。 */
export function tokenize(text) {
  const t = String(text).trim()
  return t.includes(' ') ? t.split(/\s+/) : [...t]
}

/** 构建词表 */
export function buildVocabulary(samples) {
  const vocab = new Set()
  for (const tokens of samples) for (const t of tokens) vocab.add(t)
  return [...vocab]
}

/** 词袋向量（稠密） */
export function bowVector(tokens, vocab, index) {
  const idx = index || new Map(vocab.map((w, i) => [w, i]))
  const v = new Array(vocab.length).fill(0)
  for (const t of tokens) {
    const i = idx.get(t)
    if (i !== undefined) v[i] += 1
  }
  return v
}

/** 训练多项式朴素贝叶斯（对数空间 + alpha 平滑） */
export function trainNaiveBayes(samples, labels, { alpha = 1 } = {}) {
  const classes = [...new Set(labels)]
  const classCounts = {}
  const classTotals = {}
  const wordCounts = {}
  const vocab = new Set()
  for (const l of classes) {
    classCounts[l] = 0
    classTotals[l] = 0
    wordCounts[l] = {}
  }
  samples.forEach((tokens, i) => {
    const l = labels[i]
    classCounts[l] += 1
    for (const t of tokens) {
      vocab.add(t)
      wordCounts[l][t] = (wordCounts[l][t] || 0) + 1
      classTotals[l] += 1
    }
  })
  const total = samples.length
  const logPrior = {}
  const logLik = {}
  for (const l of classes) {
    logPrior[l] = Math.log(classCounts[l] / total)
    logLik[l] = {}
    const denom = classTotals[l] + alpha * vocab.size
    for (const w of vocab) {
      logLik[l][w] = Math.log(((wordCounts[l][w] || 0) + alpha) / denom)
    }
  }
  return { classes, logPrior, logLik, vocab: [...vocab] }
}

/** 朴素贝叶斯预测：返回各类对数概率与 softmax 归一化概率 */
export function predictNaiveBayes(model, tokens) {
  const logScores = model.classes.map((l) => {
    let s = model.logPrior[l]
    for (const t of tokens) {
      if (model.logLik[l][t] !== undefined) s += model.logLik[l][t]
    }
    return s
  })
  const max = Math.max(...logScores)
  const exps = logScores.map((s) => Math.exp(s - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  const probs = exps.map((e) => e / sum)
  const best = model.classes[logScores.indexOf(max)]
  return { label: best, probs, classes: model.classes, logScores }
}

/**
 * 训练线性 SVM（二分类，hinge + L2 正则）。
 * @param {number[][]} X 特征向量 [n, d]
 * @param {number[]} y 标签 ±1
 */
export function trainLinearSVM(X, y, { lr = 0.01, epochs = 200, C = 1, seed = 42 } = {}) {
  const d = X[0].length
  const rng = createRng(seed)
  const w = Array.from({ length: d }, () => (rng() - 0.5) * 0.01)
  let b = 0
  const losses = []
  for (let ep = 0; ep < epochs; ep++) {
    let totalLoss = 0
    for (let i = 0; i < X.length; i++) {
      const margin = y[i] * (dot(w, X[i]) + b)
      if (margin < 1) {
        for (let j = 0; j < d; j++) w[j] = w[j] - lr * (w[j] / C - y[i] * X[i][j])
        b = b - lr * -y[i]
      } else {
        for (let j = 0; j < d; j++) w[j] = w[j] - lr * (w[j] / C)
      }
      totalLoss += Math.max(0, 1 - margin) + (0.5 / C) * dot(w, w)
    }
    losses.push(totalLoss)
  }
  return { w, b, losses }
}

/** SVM 决策分数（不是概率） */
export function svmDecisionScores(model, X) {
  return X.map((x) => dot(model.w, x) + model.b)
}
