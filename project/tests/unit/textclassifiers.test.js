import { describe, it, expect } from 'vitest'
import {
  trainRNNClassifier,
  predictRNNClassifier,
  trainCNNClassifier,
  predictCNNClassifier
} from '../../src/algorithms/textclassifiers.js'

const samples = [
  '这 部 电影 太 棒 了',
  '剧情 拖沓 无聊',
  '画面 精美 音乐 好听',
  '浪费 时间 难看',
  '整体 不错 值得 一看',
  '剧情 混乱 看不懂',
  '主演 演技 精湛',
  '太 无聊 了'
]
const labels = ['正面', '负面', '正面', '负面', '正面', '负面', '正面', '负面']

describe('RNN 分类器', () => {
  it('训练损失逐轮下降', () => {
    const r = trainRNNClassifier(samples, labels, { dim: 4, hidden: 4, epochs: 40, lr: 0.5, seed: 0 })
    expect(r.losses[r.losses.length - 1]).toBeLessThan(r.losses[0])
  })
  it('同一 seed 可复现', () => {
    const a = trainRNNClassifier(samples, labels, { epochs: 5, seed: 1 })
    const b = trainRNNClassifier(samples, labels, { epochs: 5, seed: 1 })
    expect(a.model.E).toEqual(b.model.E)
  })
  it('预测返回合法类别与概率和为 1', () => {
    const { model } = trainRNNClassifier(samples, labels, { epochs: 20, seed: 0 })
    const p = predictRNNClassifier(model, '这 部 电影 很棒')
    expect(model.classes).toContain(p.label)
    expect(p.probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
  })
  it('训练集准确率较高', () => {
    const { model } = trainRNNClassifier(samples, labels, { dim: 4, hidden: 4, epochs: 40, lr: 0.5, seed: 0 })
    const preds = samples.map((s) => predictRNNClassifier(model, s).label)
    const acc = preds.filter((p, i) => p === labels[i]).length / labels.length
    expect(acc).toBeGreaterThanOrEqual(0.6)
  })
})

describe('CNN 分类器', () => {
  it('训练损失逐轮下降', () => {
    const r = trainCNNClassifier(samples, labels, { dim: 4, epochs: 40, lr: 0.5, seed: 0 })
    expect(r.losses[r.losses.length - 1]).toBeLessThan(r.losses[0])
  })
  it('预测概率和为 1', () => {
    const { model } = trainCNNClassifier(samples, labels, { epochs: 20, seed: 0 })
    const p = predictCNNClassifier(model, '画面 精美')
    expect(p.probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
  })
  it('训练集准确率较高', () => {
    const { model } = trainCNNClassifier(samples, labels, { dim: 4, epochs: 40, lr: 0.5, seed: 0 })
    const preds = samples.map((s) => predictCNNClassifier(model, s).label)
    const acc = preds.filter((p, i) => p === labels[i]).length / labels.length
    expect(acc).toBeGreaterThanOrEqual(0.6)
  })
})
