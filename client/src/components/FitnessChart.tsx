import React from 'react';
import './FitnessChart.css';

interface FitnessChartProps {
  history: number[];
}

const FitnessChart: React.FC<FitnessChartProps> = ({ history }) => {
  if (history.length === 0) {
    return null;
  }

  const width = 500;
  const height = 200;
  const padding = 40;

  const maxFitness = Math.max(...history);
  const minFitness = Math.min(...history);

  const getX = (index: number) => {
    if (history.length <= 1) return padding;
    return padding + (index / (history.length - 1)) * (width - padding * 2);
  };

  const getY = (value: number) => {
    if (maxFitness === minFitness) return height / 2;
    return height - padding - ((value - minFitness) / (maxFitness - minFitness)) * (height - padding * 2);
  };

  const path = history.map((value, index) => {
    const x = getX(index);
    const y = getY(value);
    return `${index === 0 ? 'M' : 'L'} ${x},${y}`;
  }).join(' ');

  return (
    <div className="fitness-chart">
      <h4>GA Fitness Over Generations</h4>
      <svg width={width} height={height}>
        {/* Y-axis */}
        <text x={padding - 10} y={padding} textAnchor="end">{maxFitness.toFixed(0)}</text>
        <text x={padding - 10} y={height - padding} textAnchor="end">{minFitness.toFixed(0)}</text>
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#ccc" />

        {/* X-axis */}
        <text x={padding} y={height - padding + 20}>0</text>
        <text x={width - padding} y={height - padding + 20} textAnchor="end">{history.length - 1}</text>
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#ccc" />

        {/* Path */}
        <path d={path} stroke="steelblue" fill="none" strokeWidth="2" />
      </svg>
    </div>
  );
};

export default FitnessChart;
