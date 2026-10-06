import { describe, it, expect } from 'vitest'
import {
  logisticLoss2D,
  logisticGrad2D,
  lossSurface2D,
  trainLogistic2D,
  makeBinaryDataset
} from '../../src/algorithms/optimization.js'

describe('logisticLoss2D / logisticGrad2D', () => {
  const X = [
    [0.5, -0.3],
    [1.0, 0.2],
    [-0.5, 0.8]
  ]
  const y = [1, 1, 0]

  it('损失为有限非负数', () => {
    const loss = logisticLoss2D([0.1, 0.2], X, y)
    expect(loss).toBeGreaterThan(0)
    expect(Number.isFinite(loss)).toBe(true)
  })

  it('解析梯度通过独立有限差分检查', () => {
    const w = [0.3, -0.4]
    const g = logisticGrad2D(w, X, y)
    const eps = 1e-5
    for (let d = 0; d < 2; d++) {
      const wp = [...w]
      wp[d] += eps
      const wm = [...w]
      wm[d] -= eps
      const numeric = (logisticLoss2D(wp, X, y) - logisticLoss2D(wm, X, y)) / (2 * eps)
      expect(g[d]).toBeCloseTo(numeric, 3)
    }
  })
})

describe('trainLogistic2D（真实迭代）', () => {
  it('正常学习率使损失下降', () => {
    const { X, y } = makeBinaryDataset(100, 1)
    const r = trainLogistic2D(X, y, { lr: 0.1, epochs: 200, init: [0, 0] })
    expect(r.losses[r.losses.length - 1]).toBeLessThan(r.losses[0])
    expect(r.divergedAt).toBeNull()
  })

  it('大学习率震荡：损失曲线出现明显上升（非单调）', () => {
    const { X, y } = makeBinaryDataset(100, 1)
    const r = trainLogistic2D(X, y, { lr: 200, epochs: 50, init: [0, 0] })
    const L = r.losses
    const oscillated = L.some((v, i) => i > 0 && v > L[i - 1] + 0.01)
    expect(oscillated).toBe(true)
  })

  it('小学习率慢收敛：损失下降但幅度小', () => {
    const { X, y } = makeBinaryDataset(100, 1)
    const r = trainLogistic2D(X, y, { lr: 0.002, epochs: 50, init: [0, 0] })
    const drop = r.losses[0] - r.losses[r.losses.length - 1]
    expect(drop).toBeGreaterThan(0)
    expect(drop).toBeLessThan(0.2)
  })

  it('轨迹长度 = 迭代数 + 1（含初值）', () => {
    const { X, y } = makeBinaryDataset(20, 1)
    const r = trainLogistic2D(X, y, { lr: 0.1, epochs: 30, init: [0, 0] })
    expect(r.trajectory.length).toBe(r.losses.length)
  })
})

describe('lossSurface2D', () => {
  it('返回正确形状', () => {
    const { X, y } = makeBinaryDataset(20, 1)
    const w0s = [-1, 0, 1]
    const w1s = [-1, 0, 1]
    const Z = lossSurface2D(X, y, w0s, w1s)
    expect(Z.length).toBe(w1s.length)
    expect(Z[0].length).toBe(w0s.length)
  })
})

describe('makeBinaryDataset', () => {
  it('同一 seed 可复现', () => {
    const a = makeBinaryDataset(50, 7)
    const b = makeBinaryDataset(50, 7)
    expect(a.X).toEqual(b.X)
    expect(a.y).toEqual(b.y)
  })
})
