import { describe, it, expect } from 'vitest'
import { parseCSV, parseJSONL, parseTXT, analyzeSamples, validateSamples } from '../../src/data/datasets.js'

describe('parseCSV', () => {
  it('基本解析', () => {
    expect(parseCSV('a,b,c\n1,2,3')).toEqual([
      ['a', 'b', 'c'],
      ['1', '2', '3']
    ])
  })
  it('带引号的字段（内嵌分隔符）', () => {
    expect(parseCSV('"a,b",c')).toEqual([['a,b', 'c']])
  })
  it('转义引号 ""', () => {
    expect(parseCSV('"a""b",c')).toEqual([['a"b', 'c']])
  })
  it('引号内嵌换行', () => {
    expect(parseCSV('"a\nb",c')).toEqual([['a\nb', 'c']])
  })
  it('CRLF 换行', () => {
    expect(parseCSV('a,b\r\n1,2')).toEqual([
      ['a', 'b'],
      ['1', '2']
    ])
  })
  it('自定义分隔符（制表符）', () => {
    expect(parseCSV('a\tb\n1\t2', { delimiter: '\t' })).toEqual([
      ['a', 'b'],
      ['1', '2']
    ])
  })
})

describe('parseJSONL', () => {
  it('解析有效行', () => {
    const { records, errors } = parseJSONL('{"text":"a","label":"1"}\n{"text":"b","label":"0"}')
    expect(records).toHaveLength(2)
    expect(records[0].label).toBe('1')
    expect(errors).toHaveLength(0)
  })
  it('无效行报错并带行号', () => {
    const { errors } = parseJSONL('{"ok":1}\n{broken}')
    expect(errors).toHaveLength(1)
    expect(errors[0].line).toBe(2)
  })
})

describe('parseTXT', () => {
  it('按空行分隔文档', () => {
    expect(parseTXT('doc1 line\n\ndoc2 line')).toEqual(['doc1 line', 'doc2 line'])
  })
  it('无空行按行分隔', () => {
    expect(parseTXT('line1\nline2')).toEqual(['line1', 'line2'])
  })
})

describe('analyzeSamples', () => {
  const samples = [
    { text: '你好', label: '正' },
    { text: '世界', label: '负' },
    { text: '你好', label: '正' },
    { text: '很长的一段文本内容', label: '负' }
  ]
  it('标签分布', () => {
    const r = analyzeSamples(samples)
    expect(r.labelDistribution).toEqual({ 正: 2, 负: 2 })
  })
  it('重复检查', () => {
    const r = analyzeSamples(samples)
    expect(r.duplicatePairs).toEqual([[0, 2]])
  })
  it('长度统计', () => {
    const r = analyzeSamples(samples)
    expect(r.count).toBe(4)
    expect(r.lengthStats.min).toBe(2)
    expect(r.lengthStats.max).toBe(9)
  })
})

describe('validateSamples', () => {
  it('缺少 text 字段报错', () => {
    const r = validateSamples([{ label: '1' }, { text: 'ok', label: '0' }])
    expect(r.valid).toBe(false)
    expect(r.errors).toHaveLength(1)
    expect(r.errors[0].index).toBe(0)
  })
})
