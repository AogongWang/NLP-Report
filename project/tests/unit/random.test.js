import { describe, it, expect } from 'vitest'
import { createRng, shuffle, randInt, hashString } from '../../src/core/random.js'

describe('createRng', () => {
  it('同一 seed 产生同一序列', () => {
    const a = createRng(42)
    const b = createRng(42)
    const seqA = Array.from({ length: 100 }, () => a())
    const seqB = Array.from({ length: 100 }, () => b())
    expect(seqA).toEqual(seqB)
  })
  it('不同 seed 序列不同', () => {
    const a = createRng(1)
    const b = createRng(2)
    expect(a()).not.toBe(b())
  })
  it('值域在 [0, 1)', () => {
    const rng = createRng(7)
    for (let i = 0; i < 1000; i++) {
      const x = rng()
      expect(x).toBeGreaterThanOrEqual(0)
      expect(x).toBeLessThan(1)
    }
  })
})

describe('shuffle', () => {
  it('同一 seed 结果一致', () => {
    const arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    expect(shuffle(createRng(99), arr)).toEqual(shuffle(createRng(99), arr))
  })
  it('不改变原数组', () => {
    const arr = [1, 2, 3]
    shuffle(createRng(1), arr)
    expect(arr).toEqual([1, 2, 3])
  })
  it('shuffle 是原数组的排列', () => {
    const arr = [1, 2, 3, 4, 5]
    const out = shuffle(createRng(3), arr)
    expect([...out].sort((a, b) => a - b)).toEqual([...arr].sort((a, b) => a - b))
  })
})

describe('randInt / hashString', () => {
  it('randInt 在闭区间内', () => {
    const rng = createRng(5)
    for (let i = 0; i < 100; i++) {
      const x = randInt(rng, 2, 5)
      expect(x).toBeGreaterThanOrEqual(2)
      expect(x).toBeLessThanOrEqual(5)
    }
  })
  it('hashString 确定性', () => {
    expect(hashString('hello')).toBe(hashString('hello'))
    expect(hashString('hello')).not.toBe(hashString('world'))
  })
})
