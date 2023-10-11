import { useEffect } from 'react'

// replace all backticks surrounding phrases w/ inline code style
const useInlineCodeStyling = (className: string = 'post-content') => {
  useEffect(() => {
    const elements = document.querySelectorAll(`.${className}`)

    elements.forEach((el) => {
      el.innerHTML = el.innerHTML.replace(
        /`([^`]+)`/g,
        '<span class="inline-code">$1</span>'
      )
    })
  }, [className])
}

export default useInlineCodeStyling
