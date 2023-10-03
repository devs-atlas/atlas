import hljs from 'highlight.js/lib/core'

import type { NextApiRequest, NextApiResponse } from 'next'

const highlight = (req: NextApiRequest, res: NextApiResponse) => {
  const { code, language } = req.body

  const highlightedCode = hljs.highlight(code, { language }).value

  res.status(200).json({ highlightedCode })
}

export default highlight
