/**
 * 实验首页内容（不含顶栏，顶栏由 main.js 统一渲染）。
 */
export function renderHome(content) {
  const el = document.createElement('div')
  el.className = 'home'
  el.innerHTML = `
    <main class="home-main">
      <h1>操作、计算、观察、比较、复现</h1>
      <p class="lede">一套纯前端、可离线打开的 NLP 原理仿真工具。算法真实、中间计算可逐项核对。</p>
      <section class="grid" aria-label="模块入口"></section>
    </main>
  `
  const grid = el.querySelector('.grid')
  const catalog = [
    ['embeddings', 'M01 词向量', 'CBOW / Skip-gram / GloVe 训练与向量空间'],
    ['sequence', 'M02 序列模型', 'RNN / LSTM / GRU 单步前向与梯度'],
    ['cnn', 'M03 Text-CNN', '卷积滑动、池化与分类概率'],
    ['attention', 'M04 注意力', '点积 / 加性注意力权重热力图'],
    ['comparison', 'M05 多模型对照', 'NB / SVM / RNN / CNN 同数据评测'],
    ['optimization', 'M06 损失与优化', '交叉熵、真实梯度与学习率实验']
  ]
  for (const [id, title, desc] of catalog) {
    const a = document.createElement('a')
    a.className = 'card'
    a.href = '#/' + id
    const h2 = document.createElement('h2')
    h2.textContent = title
    const p = document.createElement('p')
    p.textContent = desc
    a.appendChild(h2)
    a.appendChild(p)
    grid.appendChild(a)
  }
  content.appendChild(el)
}
