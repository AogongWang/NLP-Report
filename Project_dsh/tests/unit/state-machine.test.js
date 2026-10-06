import { describe, it, expect } from 'vitest'
import { createStateMachine, STATES } from '../../src/core/state-machine.js'

describe('状态机生命周期', () => {
  it('正常流程 idle → validating → running → completed', () => {
    const sm = createStateMachine()
    sm.transition(STATES.validating)
    sm.transition(STATES.running)
    sm.transition(STATES.completed)
    expect(sm.state).toBe(STATES.completed)
  })

  it('running 可取消', () => {
    const sm = createStateMachine()
    sm.transition(STATES.validating)
    sm.transition(STATES.running)
    sm.transition(STATES.cancelling)
    sm.transition(STATES.cancelled)
    expect(sm.state).toBe(STATES.cancelled)
  })

  it('running 可失败', () => {
    const sm = createStateMachine()
    sm.transition(STATES.validating)
    sm.transition(STATES.running)
    sm.transition(STATES.failed)
    expect(sm.state).toBe(STATES.failed)
  })

  it('非法转换抛错', () => {
    const sm = createStateMachine()
    expect(() => sm.transition(STATES.running)).toThrow()
  })

  it('配置变化 → 有结果状态标记为 stale', () => {
    const sm = createStateMachine()
    sm.transition(STATES.validating)
    sm.transition(STATES.running)
    sm.transition(STATES.completed)
    sm.markStale()
    expect(sm.state).toBe(STATES.stale)
  })

  it('idle 状态 markStale 不变', () => {
    const sm = createStateMachine()
    sm.markStale()
    expect(sm.state).toBe(STATES.idle)
  })

  it('stale 后可重新 validating', () => {
    const sm = createStateMachine()
    sm.transition(STATES.validating)
    sm.transition(STATES.running)
    sm.transition(STATES.completed)
    sm.markStale()
    sm.transition(STATES.validating)
    expect(sm.state).toBe(STATES.validating)
  })

  it('subscribe 收到状态变化', () => {
    const sm = createStateMachine()
    const seen = []
    sm.subscribe((s) => seen.push(s))
    sm.transition(STATES.validating)
    expect(seen).toEqual([STATES.validating])
  })
})
