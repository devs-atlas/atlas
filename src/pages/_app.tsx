import hljs from 'highlight.js/lib/core'
import javascript from 'highlight.js/lib/languages/javascript'
import python from 'highlight.js/lib/languages/python'
import 'highlight.js/styles/stackoverflow-light.css'

hljs.registerLanguage('javascript', javascript)
hljs.registerLanguage('python', python)

import type { AppProps } from 'next/app'
import '~/styles/globals.css'

const App = ({ Component, pageProps }: AppProps) => {
  return <Component {...pageProps} />
}

export default App
