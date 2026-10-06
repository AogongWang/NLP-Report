import { describe, it, expect } from 'vitest'
import { embed, conv1d, reluArr, maxPool, textCNN } from '../../src/algorithms/cnn.js'

describe('embed', () => {
  it('token id → embedding 矩阵', () => {
    const emb = embed([0, 1], [
      [1, 0],
      [0, 1]
    ])
    expect(emb).toEqual([
      [1, 0],
      [0, 1]
    ])
  })
})

describe('conv1d', () => {
  it('卷积核沿序列滑动，覆盖 embedding 维', () => {
    const emb = [
      [1, 0],
      [0, 1]
    ]
    const kernel = [
      [1, 0],
      [0, 1]
    ]
    // 只有 1 个有效窗口：1*1 + 0*0 + 0*0 + 1*1 = 2
    expect(conv1d(emb, kernel)).toEqual([2])
  })
  it('窗口数 = 序列长 - 窗口 + 1', () => {
    const emb = [
      [1],
      [2],
      [3]
    ]
    const kernel = [[1], [1]]
    expect(conv1d(emb, kernel)).toEqual([3, 5])
  })
  it('短文本：有效窗口为 0，返回空', () => {
    const emb = [[1]]
    const kernel = [[1], [1]]
    expect(conv1d(emb, kernel)).toEqual([])
  })
})

describe('reluArr / maxPool', () => {
  it('ReLU 截断负数', () => {
    expect(reluArr([-1, 2, 0])).toEqual([0, 2, 0])
  })
  it('max-over-time 取最大', () => {
    expect(maxPool([0.2, 0.9, 0.1])).toBe(0.9)
  })
  it('空特征图池化为 0（短文本不主导分类）', () => {
    expect(maxPool([])).toBe(0)
  })
})

describe('textCNN 完整前向（手算）', () => {
  it('单卷积核 + 单分类', () => {
    const tokens = [0, 1]
    const embedding = [
      [1, 0],
      [0, 1]
    ]
    const kernels = [
      [
        [1, 0],
        [0, 1]
      ]
    ]
    const r = textCNN(tokens, embedding, kernels, [[1]], [0])
    expect(r.pooled).toEqual([2])
    expect(r.scores).toEqual([2])
    expect(r.probs[0]).toBeCloseTo(1)
  })

  it('ReLU 使负卷积贡献为 0', () => {
    const tokens = [0, 1]
    const embedding = [
      [1, 0],
      [0, 1]
    ]
    const kernels = [
      [
        [-1, 0],
        [0, -1]
      ]
    ] // 卷积结果 = -2 → ReLU → 0
    const r = textCNN(tokens, embedding, kernels, [[1]], [0])
    expect(r.pooled).toEqual([0])
    expect(r.scores).toEqual([0])
  })

  it('多卷积核 + 二分类概率和为 1', () => {
    const tokens = [0, 1, 0]
    const embedding = [
      [1, 0],
      [0, 1]
    ]
    const k1 = [
      [1, 0],
      [0, 1]
    ]
    const k2 = [[1, 0]]
    const r = textCNN(tokens, embedding, [k1, k2], [
      [1, 0],
      [0, 1]
    ], [0, 0])
    const sum = r.probs.reduce((a, b) => a + b, 0)
    expect(sum).toBeCloseTo(1, 8)
    expect(r.pooled.length).toBe(2)
  })
})
