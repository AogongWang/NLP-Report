/**
 * 六模块目录（单一数据源）。
 * 每个模块的算法实现与页面后续按 id 接入。
 */
export const catalog = [
  { id: 'embeddings', title: 'M01 词向量', desc: 'CBOW / Skip-gram / GloVe' },
  { id: 'sequence', title: 'M02 序列模型', desc: 'RNN / LSTM / GRU' },
  { id: 'cnn', title: 'M03 Text-CNN', desc: '卷积 / 池化 / 分类' },
  { id: 'attention', title: 'M04 注意力', desc: '点积 / 加性注意力' },
  { id: 'comparison', title: 'M05 多模型对照', desc: 'NB / SVM / RNN / CNN' },
  { id: 'optimization', title: 'M06 损失与优化', desc: '交叉熵 / 梯度 / 学习率' }
]
