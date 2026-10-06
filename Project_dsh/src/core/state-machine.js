/**
 * 实验状态机（§6.4）。
 * 生命周期：idle → validating → running → paused → completed
 *                                ↘ cancelling → cancelled
 *                                ↘ failed
 * 配置变化后结果过期：任意「有结果」的状态 → stale。
 * 页面据此区分「运行中 / 已完成 / 已过期」，避免拿旧结果冒充新参数结果。
 */

export const STATES = Object.freeze({
  idle: 'idle',
  validating: 'validating',
  running: 'running',
  paused: 'paused',
  completed: 'completed',
  cancelling: 'cancelling',
  cancelled: 'cancelled',
  failed: 'failed',
  stale: 'stale'
})

const TRANSITIONS = {
  idle: ['validating'],
  validating: ['running', 'failed', 'cancelling'],
  running: ['paused', 'completed', 'failed', 'cancelling'],
  paused: ['running', 'cancelling'],
  cancelling: ['cancelled', 'failed'],
  completed: ['validating', 'stale'],
  cancelled: ['validating', 'stale'],
  failed: ['validating', 'stale'],
  stale: ['validating']
}

/** 存在结果、可被配置变化标记为 stale 的状态 */
const HAS_RESULT = new Set(['running', 'paused', 'completed', 'failed', 'cancelled'])

export function createStateMachine(initial = STATES.idle) {
  let state = initial
  const listeners = new Set()

  function can(next) {
    return (TRANSITIONS[state] || []).includes(next)
  }

  function transition(next) {
    if (!can(next)) {
      throw new Error(`非法状态转换: ${state} → ${next}`)
    }
    state = next
    for (const fn of listeners) fn(state)
    return state
  }

  return {
    get state() {
      return state
    },
    can,
    transition,
    /** 配置变化：若当前有结果则标记为过期 */
    markStale() {
      if (HAS_RESULT.has(state)) transition(STATES.stale)
      return state
    },
    /** 订阅状态变化，返回取消订阅函数 */
    subscribe(fn) {
      listeners.add(fn)
      return () => listeners.delete(fn)
    }
  }
}
