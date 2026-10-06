import { describe, it, expect } from 'vitest'
import {
  softmax,
  logSumExp,
  sigmoid,
  crossEntropy,
  cosineSimilarity,
  scaledDotScore
} from '../../src/core/math.js'

describe('softmax（数值稳定）', () => {
  it('softmax([1,2,3]) 与手算值一致', () => {
    const r = softmax([1, 2, 3])
    expect(r[0]).toBeCloseTo(0.09003057, 6)
    expect(r[1]).toBeCloseTo(0.24472847, 6)
    expect(r[2]).toBeCloseTo(0.66524096, 6)
  })
  it('softmax([1000,1001,1002]) 不溢出且与上例接近', () => {
    const r = softmax([1000, 1001, 1002])
    expect(r[0]).toBeCloseTo(0.09003057, 6)
    expect(r[1]).toBeCloseTo(0.24472847, 6)
    expect(r[2]).toBeCloseTo(0.66524096, 6)
  })
  it('和为 1', () => {
    const r = softmax([1, 2, 3])
    expect(r.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
  })
  it('全 -Infinity 不产生 NaN', () => {
    const r = softmax([-Infinity, -Infinity])
    expect(r.every((x) => Number.isFinite(x))).toBe(true)
  })
})

describe('logSumExp', () => {
  it('空数组返回 -Infinity', () => {
    expect(logSumExp([])).toBe(-Infinity)
  })
  it('log(sum(exp([1,2,3])))', () => {
    const expected = Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3))
    expect(logSumExp([1, 2, 3])).toBeCloseTo(expected, 10)
  })
})

describe('sigmoid', () => {
  it('sigmoid(0) = 0.5', () => {
    expect(sigmoid(0)).toBeCloseTo(0.5, 10)
  })
  it('sigmoid(±大数) 不溢出', () => {
    expect(sigmoid(100)).toBeCloseTo(1, 10)
    expect(sigmoid(-100)).toBeCloseTo(0, 10)
  })
})

describe('crossEntropy（natural log）', () => {
  it('p=0.8, label=1 → -ln(0.8) ≈ 0.22314355', () => {
    expect(crossEntropy(0.8, 1)).toBeCloseTo(0.22314355, 6)
  })
  it('p=0.8, label=0 → -ln(0.2) ≈ 1.60943791', () => {
    expect(crossEntropy(0.8, 0)).toBeCloseTo(1.60943791, 6)
  })
})

describe('cosineSimilarity', () => {
  it('同向 = 1', () => {
    expect(cosineSimilarity([1, 2, 3], [2, 4, 6])).toBeCloseTo(1, 10)
  })
  it('正交 = 0', () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0, 10)
  })
  it('零向量返回 0（不 NaN）', () => {
    expect(cosineSimilarity([0, 0], [1, 1])).toBe(0)
  })
})

describe('scaledDotScore', () => {
  it('q·k / sqrt(dk)', () => {
    const q = [1, 2]
    const k = [3, 4]
    expect(scaledDotScore(q, k)).toBeCloseTo((1 * 3 + 2 * 4) / Math.sqrt(2), 10)
  })
})
