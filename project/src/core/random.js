/**
 * 可播种伪随机数生成器（mulberry32）。
 * 同一 seed 产生完全相同的序列，用于可复现的初始化 / 采样 / shuffle。
 * 随机性只用于初始化与抽样，不替代算法本身。
 */

/** 创建可播种 RNG，返回 [0,1) 的生成函数 */
export function createRng(seed) {
  let a = seed >>> 0
  return function next() {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** [lo, hi) 均匀分布 */
export function uniform(rng, lo = 0, hi = 1) {
  return lo + (hi - lo) * rng()
}

/** 标准正态（Box-Muller） */
export function randn(rng, mean = 0, std = 1) {
  let u = 0
  let v = 0
  while (u === 0) u = rng()
  while (v === 0) v = rng()
  const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  return mean + std * z
}

/** [lo, hi] 整数（含两端） */
export function randInt(rng, lo, hi) {
  return lo + Math.floor(rng() * (hi - lo + 1))
}

/** 从数组中等概率取一个元素 */
export function choice(rng, arr) {
  return arr[Math.floor(rng() * arr.length)]
}

/** Fisher-Yates shuffle，返回新数组，不改变原数组 */
export function shuffle(rng, arr) {
  const a = arr.slice()
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    const tmp = a[i]
    a[i] = a[j]
    a[j] = tmp
  }
  return a
}

/** FNV-1a 字符串哈希（用于从数据指纹/名称生成确定性 seed） */
export function hashString(str) {
  let h = 0x811c9dc5
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}
