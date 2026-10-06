import { describe, it, expect } from 'vitest'
import {
  buildVocab,
  cooccurrence,
  trainSkipGram,
  trainCBOW,
  trainGloVe,
  pca2D,
  cosineSimilarity
} from '../../src/algorithms/embeddings.js'

const corpus = [
  ['我', '爱', '自然', '语言', '处理'],
  ['自然', '语言', '处理', '很', '有趣'],
  ['我', '爱', '编程']
]

describe('buildVocab', () => {
  it('构建词表与 id 映射', () => {
    const v = buildVocab(corpus)
    expect(v.vocabSize).toBeGreaterThan(0)
    expect(v.word2id.get('自然')).toBeDefined()
  })
})

describe('cooccurrence', () => {
  it('共现计数对称且非负', () => {
    const { word2id } = buildVocab(corpus)
    const { sparse, cooc } = cooccurrence(corpus, word2id, 2)
    expect(sparse.length).toBeGreaterThan(0)
    for (const { i, j, count } of sparse) {
      expect(count).toBeGreaterThan(0)
      expect(cooc[j][i]).toBe(cooc[i][j])
    }
  })
})

describe('trainSkipGram', () => {
  it('同一 seed 可复现', () => {
    const a = trainSkipGram(corpus, { dim: 4, epochs: 2, seed: 1 })
    const b = trainSkipGram(corpus, { dim: 4, epochs: 2, seed: 1 })
    expect(a.W).toEqual(b.W)
  })
  it('训练会真实更新向量（损失逐轮下降）', () => {
    const r = trainSkipGram(corpus, { dim: 4, epochs: 5, lr: 0.1, seed: 0 })
    expect(r.losses[r.losses.length - 1]).toBeLessThan(r.losses[0])
  })
})

describe('trainCBOW', () => {
  it('训练会真实更新参数（向量偏离初始值）', () => {
    const r = trainCBOW(corpus, { dim: 4, epochs: 5, lr: 0.1, seed: 0 })
    const init = trainCBOW(corpus, { dim: 4, epochs: 0, seed: 0 })
    expect(r.W).not.toEqual(init.W)
  })
  it('较大语料下损失逐轮下降', () => {
    const big = []
    for (let i = 0; i < 20; i++) big.push(...corpus.map((s) => [...s]))
    const r = trainCBOW(big, { dim: 8, epochs: 5, lr: 0.05, seed: 0 })
    expect(r.losses[r.losses.length - 1]).toBeLessThan(r.losses[0])
  })
})

describe('trainGloVe', () => {
  it('GloVe 目标逐轮下降', () => {
    const { word2id } = buildVocab(corpus)
    const { sparse } = cooccurrence(corpus, word2id, 2)
    const r = trainGloVe(sparse, word2id.size, { dim: 4, epochs: 10, lr: 0.05, seed: 0 })
    expect(r.losses[r.losses.length - 1]).toBeLessThan(r.losses[0])
    expect(r.vectors.length).toBe(word2id.size)
  })
})

describe('pca2D / cosineSimilarity', () => {
  it('PCA 输出二维坐标', () => {
    const vecs = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [1, 1, 0, 0]
    ]
    const pts = pca2D(vecs)
    expect(pts.length).toBe(3)
    expect(pts[0].length).toBe(2)
  })
  it('余弦相似度在原始向量上计算', () => {
    expect(cosineSimilarity([1, 0], [2, 0])).toBeCloseTo(1)
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0)
  })
})
