import hljs from 'highlight.js/lib/core'

export function highlight(code: string): string {
  return hljs.highlightAuto(code).value
}
