import { describe, it, expect } from 'vitest'
import {
  tokenize,
  buildVocabulary,
  bowVector,
  trainNaiveBayes,
  predictNaiveBayes,
  trainLinearSVM,
  svmDecisionScores
} from '../../src/algorithms/classifiers.js'
import { confusionMatrix, accuracy, precisionRecallF1, macroF1 } from '../../src/core/metrics.js'

const samples = [
  ['这', '部', '电影', '太', '棒', '了'],
  ['剧情', '拖沓', '无聊'],
  ['画面', '精美', '音乐', '好听'],
  ['浪费', '时间', '难看'],
  ['整体', '不错', '值得', '一看'],
  ['剧情', '混乱', '看不懂'],
  ['主演', '演技', '精湛'],
  ['太', '无聊', '了']
]
const labels = ['正面', '负面', '正面', '负面', '正面', '负面', '正面', '负面']

describe('tokenize / bowVector', () => {
  it('无空格按字符切分', () => {
    expect(tokenize('我爱NLP')).toEqual(['我', '爱', 'N', 'L', 'P'])
  })
  it('有空格按空格切分', () => {
    expect(tokenize('我 爱 NLP')).toEqual(['我', '爱', 'NLP'])
  })
})

describe('朴素贝叶斯', () => {
  it('训练后在训练集上准确率较高', () => {
    const model = trainNaiveBayes(samples, labels)
    const preds = samples.map((s) => predictNaiveBayes(model, s).label)
    const acc = accuracy(labels, preds)
    expect(acc).toBeGreaterThan(0.7)
  })
  it('预测概率和为 1', () => {
    const model = trainNaiveBayes(samples, labels)
    const p = predictNaiveBayes(model, ['这', '部', '电影'])
    expect(p.probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 8)
  })
})

describe('线性 SVM', () => {
  it('线性可分数据收敛到高准确率', () => {
    // 构造线性可分数据：x0 + x1 > 0 → +1
    const X = [
      [1, 1], [1, 0.5], [0.5, 1],
      [-1, -1], [-1, -0.5], [-0.5, -1]
    ]
    const y = [1, 1, 1, -1, -1, -1]
    const model = trainLinearSVM(X, y, { lr: 0.05, epochs: 500, C: 1, seed: 0 })
    const scores = svmDecisionScores(model, X)
    const preds = scores.map((s) => (s >= 0 ? 1 : -1))
    expect(preds).toEqual(y)
  })
  it('决策分数不是概率（可正可负）', () => {
    const X = [[1, 1], [-1, -1]]
    const y = [1, -1]
    const model = trainLinearSVM(X, y, { lr: 0.05, epochs: 200, seed: 0 })
    const scores = svmDecisionScores(model, X)
    expect(scores[0]).toBeGreaterThan(scores[1])
  })
})

describe('metrics', () => {
  it('混淆矩阵', () => {
    const { matrix, labels: ls } = confusionMatrix(['a', 'b', 'a', 'b'], ['a', 'b', 'b', 'b'], ['a', 'b'])
    expect(ls).toEqual(['a', 'b'])
    expect(matrix[0][0]).toBe(1) // a→a
    expect(matrix[0][1]).toBe(1) // a→b
  })
  it('precision/recall/F1 零分母为 0', () => {
    const r = precisionRecallF1(['a', 'a'], ['b', 'b'], 'a')
    expect(r.precision).toBe(0)
    expect(r.recall).toBe(0)
    expect(r.f1).toBe(0)
  })
  it('macroF1 在 [0,1]', () => {
    const f = macroF1(['a', 'b', 'a'], ['a', 'a', 'b'], ['a', 'b'])
    expect(f).toBeGreaterThanOrEqual(0)
    expect(f).toBeLessThanOrEqual(1)
  })
})
