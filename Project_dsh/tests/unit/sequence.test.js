import { describe, it, expect } from 'vitest'
import {
  rnnStep,
  rnnForward,
  lstmStep,
  gruStep,
  rnnBackward
} from '../../src/algorithms/sequence.js'

describe('rnnStep / rnnForward', () => {
  it('单步 h = tanh(W_hh h + W_xh x + b)', () => {
    // dh=1, dx=1: Wxh=[[2]], Whh=[[0.5]], bh=[0], x=[0.3], h=[0]
    const h = rnnStep([0.3], [0], [[2]], [[0.5]], [0])
    expect(h[0]).toBeCloseTo(Math.tanh(2 * 0.3))
  })
  it('连续运行与逐步运行一致', () => {
    const seq = [[0.1], [0.2], [0.3]]
    const { hs } = rnnForward(seq, [0], [[1]], [[0.5]], [0])
    let h = [0]
    const steps = [h]
    for (const x of seq) {
      h = rnnStep(x, h, [[1]], [[0.5]], [0])
      steps.push(h)
    }
    expect(hs).toEqual(steps)
  })
})

describe('lstmStep', () => {
  it('输出 h 与细胞状态 c 形状正确，门值在合理范围', () => {
    const x = [0.5, -0.2]
    const hPrev = [0.1, 0.1]
    const cPrev = [0, 0]
    // dh=2, dx=2 → 4*dh=8
    const Wx = Array.from({ length: 8 }, (_, i) => new Array(2).fill(0.01 * i))
    const Wh = Array.from({ length: 8 }, (_, i) => new Array(2).fill(0.01 * i))
    const b = new Array(8).fill(0)
    const r = lstmStep(x, hPrev, cPrev, Wx, Wh, b)
    expect(r.h.length).toBe(2)
    expect(r.c.length).toBe(2)
    for (const g of [r.i, r.f, r.o]) {
      g.forEach((v) => {
        expect(v).toBeGreaterThan(0)
        expect(v).toBeLessThan(1)
      })
    }
  })
})

describe('gruStep', () => {
  it('reset 门为 0 时 n 不含 h 项；h 由 z 门插值', () => {
    const dh = 1
    const Wx = [[0], [0], [0]]
    const Wh = [[0], [0], [0]]
    // b: [b_ir, b_iz, b_in, b_hr, b_hz, b_hn]；让 r=0, z=0 → h' = n = tanh(b_in)
    const b = [0, 0, 0.5, 0, 0, 0]
    const r = gruStep([1], [0.8], Wx, Wh, b)
    expect(r.r[0]).toBeCloseTo(0.5) // sigmoid(0)
    expect(r.z[0]).toBeCloseTo(0.5)
    // r=0.5 → n = tanh(0.5 + 0.5*(0+0)) = tanh(0.5)
    expect(r.n[0]).toBeCloseTo(Math.tanh(0.5))
    // h' = (1-z)*n + z*h = 0.5*tanh(0.5) + 0.5*0.8
    expect(r.h[0]).toBeCloseTo(0.5 * Math.tanh(0.5) + 0.5 * 0.8)
  })
})

describe('rnnBackward（真实 BPTT）', () => {
  it('dL/dx_t 通过有限差分检查', () => {
    const seq = [[0.3], [0.5]]
    const h0 = [0]
    const Wxh = [[1.0]]
    const Whh = [[0.6]]
    const bh = [0]
    const v = [1.0]

    function loss(xs) {
      const { hs } = rnnForward(xs, h0, Wxh, Whh, bh)
      // L = v·h1 + v·h2 = hs[1][0] + hs[2][0]
      return hs[1][0] + hs[2][0]
    }

    const { dLdx } = rnnBackward(seq, h0, Wxh, Whh, bh, v)
    const eps = 1e-5
    for (let t = 0; t < seq.length; t++) {
      const xsP = seq.map((x) => [...x])
      const xsM = seq.map((x) => [...x])
      xsP[t][0] += eps
      xsM[t][0] -= eps
      const numeric = (loss(xsP) - loss(xsM)) / (2 * eps)
      expect(dLdx[t][0]).toBeCloseTo(numeric, 3)
    }
  })
})
