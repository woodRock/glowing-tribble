import React from 'react';
import type { RosterMetrics } from '../types';
import './RosterMetrics.css';

interface RosterMetricsProps {
  metrics: RosterMetrics | null;
}

const RosterMetricsDisplay: React.FC<RosterMetricsProps> = ({ metrics }) => {
  if (!metrics) {
    return null;
  }

  return (
    <div className="roster-metrics">
      <h4>Roster Score (Lower is Better)</h4>
      <div className="roster-metrics__total">
        Total Penalty: {metrics.totalPenalty.toFixed(2)}
      </div>
      <div className="roster-metrics__breakdown">
        <h5>Penalties:</h5>
        <ul>
          <li>Desired Hours Difference: {metrics.penalties.desiredHours.toFixed(2)}</li>
          <li>Consecutive Days Violations: {metrics.penalties.consecutiveDays}</li>
          <li>Clopen Violations: {metrics.penalties.clopen}</li>
        </ul>
      </div>
    </div>
  );
};

export default RosterMetricsDisplay;
