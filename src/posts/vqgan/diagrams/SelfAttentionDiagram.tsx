import * as d3 from 'd3'
import { useEffect } from 'react'

const SelfAttentionDiagram = () => {
  useEffect(() => {
    const words = ['hi', 'my', 'name', 'is', 'john']
    const margin = { top: 50, right: 50, bottom: 50, left: 50 }
    const width = 400 - margin.left - margin.right
    const height = 400 - margin.top - margin.bottom

    const svgContainer = d3.select('#self-attention .diagram')

    if (!svgContainer.select('svg').empty()) {
      return
    }

    const svg = svgContainer
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`)

    const x = d3.scalePoint().domain(words).range([0, width])
    const y = d3.scalePoint().domain(words).range([0, height])

    svg.append('g').call(d3.axisTop(x))
    svg.append('g').call(d3.axisLeft(y))

    const cellSize = width / words.length

    words.forEach((_, i) => {
      words.forEach((_, j) => {
        const value = Math.random()
        svg
          .append('rect')
          .attr('x', i * cellSize)
          .attr('y', j * cellSize)
          .attr('width', cellSize)
          .attr('height', cellSize)
          .attr('fill', d3.interpolateGreys(value))
      })
    })

    // Legend
    const legendWidth = 20
    const legendHeight = 200
    const centerY = (height - legendHeight) / 2

    const legendSvg = d3
      .select('#self-attention .legend')
      .append('svg')
      .attr('width', legendWidth + 2 * margin.right)
      .attr('height', height + margin.top + margin.bottom)

    const legendScale = d3.scaleLinear().domain([0, 1]).range([legendHeight, 0])

    const gradient = legendSvg
      .append('defs')
      .append('linearGradient')
      .attr('id', 'gradient')
      .attr('x1', '0%')
      .attr('y1', '0%')
      .attr('x2', '0%')
      .attr('y2', '100%')

    gradient.append('stop').attr('offset', '0%').attr('stop-color', 'black')
    gradient.append('stop').attr('offset', '100%').attr('stop-color', 'white')

    legendSvg
      .append('rect')
      .attr('x', margin.right)
      .attr('y', centerY)
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#gradient)')

    const legendAxis = d3
      .axisRight(legendScale)
      .tickValues([0, 0.25, 0.5, 0.75, 1])
    legendSvg
      .append('g')
      .attr('transform', `translate(${legendWidth + margin.right}, ${centerY})`)
      .call(legendAxis)
  }, [])

  return (
    <div id="self-attention" className="diagram-container">
      <div className="diagram"></div>
      <div className="legend"></div>
    </div>
  )
}

export default SelfAttentionDiagram
