import { softmax } from '../core/math.js'
import { createRng } from '../core/random.js'
import { tokenize, buildVocabulary } from './classifiers.js'
import { conv1d, reluArr, maxPool } from './cnn.js'

/**
 * M05 文本分类器：RNN 分类器与 Text-CNN 分类器的真实训练（全批量梯度下降）与预测。
 * 复用 sequence.js 的 RNN 单步前向与 cnn.js 的卷积/池化，训练时补全反向传播。
 */

const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0))
const vec = (n) => new Array(n).fill(0)
function matvec(M, v) {
  return M.map((row) => row.reduce((s, w, j) => s + w * v[j], 0))
}

function prepare(samples, labels) {
  const tokenized = samples.map((s) => tokenize(s))
  const vocab = buildVocabulary(tokenized)
  const word2id = new Map(vocab.map((w, i) => [w, i]))
  const classes = [...new Set(labels)]
  const class2id = new Map(classes.map((c, i) => [c, i]))
  const ids = tokenized.map((toks) => toks.map((t) => word2id.get(t)).filter((i) => i !== undefined))
  const y = labels.map((l) => class2id.get(l))
  return { ids, y, vocab, classes, word2id, class2id }
}

/** 共享：文本 → token id 序列 */
function toIds(model, text) {
  return tokenize(text).map((t) => model.word2id.get(t)).filter((i) => i !== undefined)
}

// ================= RNN 分类器 =================

export function trainRNNClassifier(samples, labels, { dim = 4, hidden = 4, epochs = 40, lr = 0.5, seed = 42 } = {}) {
  const { ids, y, vocab, classes, word2id } = prepare(samples, labels)
  const V = vocab.length
  const C = classes.length
  const rng = createRng(seed)
  const rand = (s) => (rng() * 2 - 1) * s

  const E = Array.from({ length: V }, () => Array.from({ length: dim }, () => rand(0.5)))
  const Wxh = Array.from({ length: hidden }, () => Array.from({ length: dim }, () => rand(0.5)))
  const Whh = Array.from({ length: hidden }, () => Array.from({ length: hidden }, () => rand(0.5)))
  const bh = vec(hidden)
  const Wout = Array.from({ length: C }, () => Array.from({ length: hidden }, () => rand(0.5)))
  const bout = vec(C)
  const losses = []

  for (let ep = 0; ep < epochs; ep++) {
    const gE = zeros(V, dim)
    const gWxh = zeros(hidden, dim)
    const gWhh = zeros(hidden, hidden)
    const gbh = vec(hidden)
    const gWout = zeros(C, hidden)
    const gbout = vec(C)
    let totalLoss = 0

    for (let n = 0; n < ids.length; n++) {
      const seq = ids[n]
      const hs = [vec(hidden)]
      const zs = []
      for (let t = 0; t < seq.length; t++) {
        const x = E[seq[t]]
        const wx = matvec(Wxh, x)
        const wh = matvec(Whh, hs[t])
        const z = wh.map((v, k) => v + wx[k] + bh[k])
        zs.push(z)
        hs.push(z.map(Math.tanh))
      }
      const hT = hs[hs.length - 1]
      const logits = Wout.map((row, c) => row.reduce((s, w, j) => s + w * hT[j], 0) + bout[c])
      const probs = softmax(logits)
      totalLoss += -Math.log(probs[y[n]] + 1e-12)

      const dlogits = probs.map((p, c) => p - (c === y[n] ? 1 : 0))
      const dhT = vec(hidden)
      for (let j = 0; j < hidden; j++) for (let c = 0; c < C; c++) dhT[j] += Wout[c][j] * dlogits[c]
      for (let c = 0; c < C; c++) {
        for (let j = 0; j < hidden; j++) gWout[c][j] += dlogits[c] * hT[j]
        gbout[c] += dlogits[c]
      }

      let dh = dhT.slice()
      for (let t = seq.length - 1; t >= 0; t--) {
        const hNext = hs[t + 1]
        const dz = dh.map((g, k) => g * (1 - hNext[k] * hNext[k]))
        const x = E[seq[t]]
        const hPrev = hs[t]
        for (let k = 0; k < hidden; k++) {
          gbh[k] += dz[k]
          for (let j = 0; j < dim; j++) {
            gWxh[k][j] += dz[k] * x[j]
            gE[seq[t]][j] += dz[k] * Wxh[k][j]
          }
          for (let m = 0; m < hidden; m++) gWhh[k][m] += dz[k] * hPrev[m]
        }
        const dhPrev = vec(hidden)
        for (let m = 0; m < hidden; m++) {
          let s = 0
          for (let k = 0; k < hidden; k++) s += Whh[k][m] * dz[k]
          dhPrev[m] = s
        }
        dh = dhPrev
      }
    }

    const sc = lr / ids.length
    for (let i = 0; i < V; i++) for (let j = 0; j < dim; j++) E[i][j] -= sc * gE[i][j]
    for (let k = 0; k < hidden; k++) {
      for (let j = 0; j < dim; j++) Wxh[k][j] -= sc * gWxh[k][j]
      for (let m = 0; m < hidden; m++) Whh[k][m] -= sc * gWhh[k][m]
      bh[k] -= sc * gbh[k]
    }
    for (let c = 0; c < C; c++) {
      for (let j = 0; j < hidden; j++) Wout[c][j] -= sc * gWout[c][j]
      bout[c] -= sc * gbout[c]
    }
    losses.push(totalLoss / ids.length)
  }

  const model = { E, Wxh, Whh, bh, Wout, bout, word2id, classes }
  return { model, losses }
}

export function predictRNNClassifier(model, text) {
  const ids = toIds(model, text)
  let h = vec(model.bh.length)
  for (let t = 0; t < ids.length; t++) {
    const x = model.E[ids[t]]
    const wx = matvec(model.Wxh, x)
    const wh = matvec(model.Whh, h)
    const z = wh.map((v, k) => v + wx[k] + model.bh[k])
    h = z.map(Math.tanh)
  }
  const logits = model.Wout.map((row, c) => row.reduce((s, w, j) => s + w * h[j], 0) + model.bout[c])
  const probs = softmax(logits)
  const best = model.classes[logits.indexOf(Math.max(...logits))]
  return { label: best, probs, classes: model.classes }
}

// ================= Text-CNN 分类器 =================

export function trainCNNClassifier(samples, labels, { dim = 4, widths = [2, 3], epochs = 40, lr = 0.5, seed = 42 } = {}) {
  const { ids, y, vocab, classes, word2id } = prepare(samples, labels)
  const V = vocab.length
  const C = classes.length
  const rng = createRng(seed)
  const rand = (s) => (rng() * 2 - 1) * s

  const E = Array.from({ length: V }, () => Array.from({ length: dim }, () => rand(0.5)))
  const kernels = widths.map((w) => Array.from({ length: w }, () => Array.from({ length: dim }, () => rand(0.5))))
  const nK = kernels.length
  const FC = Array.from({ length: C }, () => Array.from({ length: nK }, () => rand(0.5)))
  const b = vec(C)
  const losses = []

  for (let ep = 0; ep < epochs; ep++) {
    const gE = zeros(V, dim)
    const gKernels = kernels.map((k) => zeros(k.length, dim))
    const gFC = zeros(C, nK)
    const gb = vec(C)
    let totalLoss = 0

    for (let n = 0; n < ids.length; n++) {
      const seq = ids[n]
      const emb = seq.map((id) => E[id])
      const featureMaps = kernels.map((k) => {
        const raw = conv1d(emb, k)
        const relu = reluArr(raw)
        return { raw, relu, pooled: maxPool(relu), maxPos: relu.indexOf(maxPool(relu)) }
      })
      const pooledVec = featureMaps.map((f) => f.pooled)
      const logits = FC.map((row, c) => row.reduce((s, w, j) => s + w * pooledVec[j], 0) + b[c])
      const probs = softmax(logits)
      totalLoss += -Math.log(probs[y[n]] + 1e-12)

      const dlogits = probs.map((p, c) => p - (c === y[n] ? 1 : 0))
      const dpooled = vec(nK)
      for (let j = 0; j < nK; j++) for (let c = 0; c < C; c++) dpooled[j] += FC[c][j] * dlogits[c]
      for (let c = 0; c < C; c++) {
        for (let j = 0; j < nK; j++) gFC[c][j] += dlogits[c] * pooledVec[j]
        gb[c] += dlogits[c]
      }

      const demb = zeros(emb.length, dim)
      for (let k = 0; k < nK; k++) {
        const fm = featureMaps[k]
        const drelu = vec(fm.relu.length)
        if (fm.maxPos >= 0) drelu[fm.maxPos] = dpooled[k]
        const dconv = drelu.map((g, i) => (fm.raw[i] > 0 ? g : 0))
        for (let i = 0; i < dconv.length; i++) {
          for (let w = 0; w < kernels[k].length; w++) {
            for (let d = 0; d < dim; d++) {
              gKernels[k][w][d] += dconv[i] * emb[i + w][d]
              demb[i + w][d] += dconv[i] * kernels[k][w][d]
            }
          }
        }
      }
      for (let t = 0; t < emb.length; t++) {
        for (let d = 0; d < dim; d++) gE[seq[t]][d] += demb[t][d]
      }
    }

    const sc = lr / ids.length
    for (let i = 0; i < V; i++) for (let j = 0; j < dim; j++) E[i][j] -= sc * gE[i][j]
    for (let k = 0; k < nK; k++) {
      for (let w = 0; w < kernels[k].length; w++) for (let d = 0; d < dim; d++) kernels[k][w][d] -= sc * gKernels[k][w][d]
    }
    for (let c = 0; c < C; c++) {
      for (let j = 0; j < nK; j++) FC[c][j] -= sc * gFC[c][j]
      b[c] -= sc * gb[c]
    }
    losses.push(totalLoss / ids.length)
  }

  const model = { E, kernels, FC, b, word2id, classes, dim }
  return { model, losses }
}

export function predictCNNClassifier(model, text) {
  const ids = toIds(model, text)
  const emb = ids.map((id) => model.E[id])
  const pooledVec = model.kernels.map((k) => maxPool(reluArr(conv1d(emb, k))))
  const logits = model.FC.map((row, c) => row.reduce((s, w, j) => s + w * pooledVec[j], 0) + model.b[c])
  const probs = softmax(logits)
  const best = model.classes[logits.indexOf(Math.max(...logits))]
  return { label: best, probs, classes: model.classes }
}
