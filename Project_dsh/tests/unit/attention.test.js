import { describe, it, expect } from 'vitest'
import {
  dotProductScores,
  scaledDotScores,
  additiveScores,
  applyMask,
  attentionWeights,
  attention
} from '../../src/algorithms/attention.js'

describe('dotProductScores', () => {
  it('score(i,j) = q_i · k_j', () => {
    const Q = [
      [1, 0],
      [0, 1]
    ]
    const K = [
      [1, 0],
      [0, 1]
    ]
    const s = dotProductScores(Q, K)
    expect(s[0][0]).toBeCloseTo(1)
    expect(s[0][1]).toBeCloseTo(0)
    expect(s[1][0]).toBeCloseTo(0)
    expect(s[1][1]).toBeCloseTo(1)
  })
})

describe('scaledDotScores', () => {
  it('除以 sqrt(dk)', () => {
    const Q = [[1, 1]]
    const K = [[1, 1]]
    const s = scaledDotScores(Q, K)
    expect(s[0][0]).toBeCloseTo(2 / Math.sqrt(2))
  })
})

describe('additiveScores', () => {
  it('v^T tanh(Wq·q + Wk·k + b)，单隐藏维手算', () => {
    const Q = [[0.5]]
    const K = [[0.3]]
    const s = additiveScores(Q, K, { Wq: [[1]], Wk: [[1]], b: [0], v: [1] })
    expect(s[0][0]).toBeCloseTo(Math.tanh(0.8))
  })
})

describe('applyMask', () => {
  it('false 位置置 -Infinity', () => {
    const scores = [[1, 2, 3]]
    const mask = [[true, false, true]]
    const m = applyMask(scores, mask)
    expect(m[0][0]).toBe(1)
    expect(m[0][1]).toBe(-Infinity)
    expect(m[0][2]).toBe(3)
  })
  it('无 mask 时原样返回', () => {
    const scores = [[1, 2]]
    expect(applyMask(scores, null)).toEqual(scores)
  })
})

describe('attentionWeights', () => {
  it('每行和为 1', () => {
    const w = attentionWeights([[1, 2, 3]])
    expect(w[0].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
  })
  it('被屏蔽位置权重为 0，有效行和约为 1', () => {
    const masked = applyMask([[1, 2, 3]], [[true, false, true]])
    const w = attentionWeights(masked)
    expect(w[0][1]).toBe(0)
    expect(w[0].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
  })
  it('整行全屏蔽 → 权重全 0 且无 NaN', () => {
    const masked = applyMask([[1, 2]], [[false, false]])
    const w = attentionWeights(masked)
    expect(w[0].every((x) => Number.isFinite(x))).toBe(true)
    expect(w[0].every((x) => x === 0)).toBe(true)
  })
})

describe('attention 前向', () => {
  it('输出 O = A @ V', () => {
    const Q = [[1, 0]]
    const K = [
      [1, 0],
      [0, 1]
    ]
    const V = [
      [2, 3],
      [4, 5]
    ]
    const { weights, output } = attention(Q, K, V, { scoreFn: dotProductScores })
    // scores = [1, 0] → weights = softmax([1,0]) ≈ [0.7311, 0.2689]
    expect(weights[0][0]).toBeCloseTo(0.7310586, 4)
    expect(weights[0][1]).toBeCloseTo(0.2689414, 4)
    expect(output[0][0]).toBeCloseTo(0.7310586 * 2 + 0.2689414 * 4, 4)
    expect(output[0][1]).toBeCloseTo(0.7310586 * 3 + 0.2689414 * 5, 4)
  })
  it('mask 前后权重与输出不同', () => {
    const Q = [[1, 0]]
    const K = [
      [1, 0],
      [0, 1]
    ]
    const V = [
      [2, 3],
      [4, 5]
    ]
    const noMask = attention(Q, K, V, { scoreFn: dotProductScores })
    const withMask = attention(Q, K, V, { scoreFn: dotProductScores, mask: [[true, false]] })
    expect(withMask.weights[0][1]).toBe(0)
    expect(withMask.output).not.toEqual(noMask.output)
  })
  it('整行全屏蔽 → invalidRows 标记该行', () => {
    const Q = [[1, 0]]
    const K = [
      [1, 0],
      [0, 1]
    ]
    const V = [
      [2, 3],
      [4, 5]
    ]
    const r = attention(Q, K, V, { scoreFn: dotProductScores, mask: [[false, false]] })
    expect(r.invalidRows).toEqual([0])
    expect(r.weights[0].every((x) => x === 0)).toBe(true)
  })
})
