import React from 'react';
import type { RosterMetrics } from '../types';
import './RosterMetrics.css';

interface RosterMetricsProps {
  metrics: RosterMetrics | null;
}

const RosterMetricsDisplay: React.FC<RosterMetricsProps> = ({ metrics }) => {
  // if (!metrics) return null; // Removing early return to debug hook error

  const details = metrics?.details;
  const hasCritical = details && (details.uncoveredShifts > 0 || details.skillMismatches > 0 || details.availabilityMismatches > 0 || details.overlaps > 0 || details.minStaffForRoleViolations > 0);

  return metrics && details ? (
    <div className="roster-metrics">
      <h4>Roster Penalty: {metrics.totalPenalty.toFixed(0)}</h4>
      
      <div className="roster-metrics__sections">
        <div className={`roster-metrics__section ${hasCritical ? 'critical' : 'ok'}`}>
          <h5>Critical Violations</h5>
          <ul>
            <li>Uncovered Shifts: {details.uncoveredShifts}</li>
            <li>Skill Mismatches: {details.skillMismatches}</li>
            <li>Availability Issues: {details.availabilityMismatches}</li>
            <li>Shift Overlaps: {details.overlaps}</li>
            <li>Role Minimums Missed: {details.minStaffForRoleViolations}</li>
          </ul>
        </div>

        <div className="roster-metrics__section">
          <h5>Workload & Fatigue</h5>
          <ul>
            <li>Min Rest Violations: {details.minRestViolations}</li>
            <li>Max Workload Exceeded: {details.maxWorkloadViolations}</li>
            <li>Min Workload Missed: {details.minWorkloadViolations}</li>
            <li>Max Consec. Shifts: {details.maxConsecutiveShiftsViolations}</li>
            <li>Min Consec. Shifts: {details.minConsecutiveShiftsViolations}</li>
            <li>Min Days Off Missed: {details.minConsecutiveDaysOffViolations}</li>
          </ul>
        </div>

        <div className="roster-metrics__section">
          <h5>Preferences & Patterns</h5>
          <ul>
            <li>Weekend Pattern: {details.maxWeekendPatternViolations}</li>
            <li>Off Requests Penalty: {details.shiftOffRequestPenalties}</li>
            <li>On Requests Penalty: {details.shiftOnRequestPenalties}</li>
            <li>Cover Req Penalty: {details.coverRequirementPenalties}</li>
            <li>Desired Hours Penalty: {details.desiredHoursPenalties.toFixed(1)}</li>
          </ul>
        </div>
      </div>
    </div>
  ) : null;
};

export default RosterMetricsDisplay;
