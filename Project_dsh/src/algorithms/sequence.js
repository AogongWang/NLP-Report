import { sigmoid } from '../core/math.js'

/**
 * M02 序列模型：RNN / LSTM / GRU 真实单步前向，及 RNN 反向传播（BPTT）。
 * 门顺序与公式与页面一致；GRU 采用 PyTorch「reset-before」约定：
 *   n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn))
 */

function matvec(M, v) {
  return M.map((row) => row.reduce((s, w, j) => s + w * v[j], 0))
}

/** RNN 单步：h' = tanh(W_hh h + W_xh x + b_h) */
export function rnnStep(x, hPrev, Wxh, Whh, bh) {
  const wx = matvec(Wxh, x)
  const wh = matvec(Whh, hPrev)
  return wh.map((val, i) => Math.tanh(val + wx[i] + bh[i]))
}

/** RNN 整段前向，返回 { hs:[h0..hT], zs:[z_0..z_{T-1}] } */
export function rnnForward(seq, h0, Wxh, Whh, bh) {
  const hs = [h0]
  const zs = []
  let h = h0
  for (const x of seq) {
    const wx = matvec(Wxh, x)
    const wh = matvec(Whh, h)
    const z = wh.map((val, i) => val + wx[i] + bh[i])
    zs.push(z)
    h = z.map(Math.tanh)
    hs.push(h)
  }
  return { hs, zs }
}

/**
 * LSTM 单步（PyTorch 门顺序 i, f, g, o）：
 * i = σ(W_ii x + b_ii + W_hi h + b_hi)；f、g、o 同理；
 * c' = f ⊙ c + i ⊙ g；h' = o ⊙ tanh(c')。
 * 参数用分块 Wx[4d,d]、Wh[4d,d]、b[4d]。
 */
export function lstmStep(x, hPrev, cPrev, Wx, Wh, b) {
  const dh = cPrev.length
  const wx = matvec(Wx, x)
  const wh = matvec(Wh, hPrev)
  const g = wh.map((val, i) => val + wx[i] + b[i]) // 拼接门 [4d]
  const slice = (start) => g.slice(start, start + dh)
  const i = slice(0).map(sigmoid)
  const f = slice(dh).map(sigmoid)
  const gc = slice(2 * dh).map(Math.tanh)
  const o = slice(3 * dh).map(sigmoid)
  const c = cPrev.map((cv, k) => f[k] * cv + i[k] * gc[k])
  const h = c.map((cv, k) => o[k] * Math.tanh(cv))
  return { h, c, i, f, g: gc, o }
}

/** LSTM 整段前向（返回每步状态与门值，供热力图/检查器） */
export function lstmForward(seq, h0, c0, Wx, Wh, b) {
  const states = [{ h: h0, c: c0 }]
  let h = h0
  let c = c0
  for (const x of seq) {
    const r = lstmStep(x, h, c, Wx, Wh, b)
    states.push(r)
    h = r.h
    c = r.c
  }
  return states
}

/**
 * GRU 单步（PyTorch reset-before）：
 * r = σ(W_ir x + b_ir + W_hr h + b_hr)
 * z = σ(W_iz x + b_iz + W_hz h + b_hz)
 * n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn))
 * h' = (1 - z) ⊙ n + z ⊙ h
 * 参数用分块 Wx[3d,d]、Wh[3d,d]、b[6d]（b_ir,b_iz,b_in,b_hr,b_hz,b_hn）。
 */
export function gruStep(x, hPrev, Wx, Wh, b) {
  const dh = hPrev.length
  const wx = matvec(Wx, x) // [3d]
  const wh = matvec(Wh, hPrev) // [3d]
  const lin = (off) => {
    const out = []
    for (let k = 0; k < dh; k++) out.push(wx[off + k] + wh[off + k] + b[off + k] + b[3 * dh + off + k])
    return out
  }
  const r = lin(0).map(sigmoid)
  const z = lin(dh).map(sigmoid)
  // n 的 h 项要单独乘 r
  const n = []
  for (let k = 0; k < dh; k++) {
    const base = wx[2 * dh + k] + b[2 * dh + k]
    n.push(Math.tanh(base + r[k] * (wh[2 * dh + k] + b[5 * dh + k])))
  }
  const h = hPrev.map((hp, k) => (1 - z[k]) * n[k] + z[k] * hp)
  return { h, r, z, n }
}

/**
 * RNN 反向传播（BPTT）。
 * 损失 L = Σ_t v · h_{t+1}（线性读出）。返回 dL/dx_t（每步）与 dL/dh_0。
 * 用于观察梯度沿时间步的传播（早期步梯度范数可据此计算）。
 */
export function rnnBackward(seq, h0, Wxh, Whh, bh, v) {
  const { hs, zs } = rnnForward(seq, h0, Wxh, Whh, bh)
  const T = seq.length
  const dh = v.length
  const dLdh = Array.from({ length: T + 1 }, () => new Array(dh).fill(0))
  const dLdx = seq.map(() => new Array(seq[0].length).fill(0))

  for (let t = T - 1; t >= 0; t--) {
    // 读出层
    for (let k = 0; k < dh; k++) dLdh[t + 1][k] += v[k]
    // 过 tanh：dL/dz_t = dL/dh_{t+1} ⊙ (1 - h_{t+1}^2)
    const hNext = hs[t + 1]
    const dz = dLdh[t + 1].map((g, k) => g * (1 - hNext[k] * hNext[k]))
    // dL/dx_t = Wxh^T dz
    for (let j = 0; j < seq[t].length; j++) {
      let s = 0
      for (let k = 0; k < dh; k++) s += Wxh[k][j] * dz[k]
      dLdx[t][j] = s
    }
    // 传给 h_t = Whh^T dz
    for (let k = 0; k < dh; k++) {
      let s = 0
      for (let m = 0; m < dh; m++) s += Whh[m][k] * dz[m]
      dLdh[t][k] = s
    }
  }
  return { dLdx, dLdh0: dLdh[0] }
}
