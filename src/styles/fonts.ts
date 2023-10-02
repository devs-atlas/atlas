import { EB_Garamond, Fragment_Mono, Manrope, Raleway } from 'next/font/google'

const fragment = Fragment_Mono({ subsets: ['latin'], weight: '400' })
const garamond = EB_Garamond({ subsets: ['latin'] })
const raleway = Raleway({ subsets: ['latin'] })
const manrope = Manrope({ subsets: ['latin'] })

export { fragment, garamond, manrope, raleway }
