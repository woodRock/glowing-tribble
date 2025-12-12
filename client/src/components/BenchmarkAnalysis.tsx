import React, { useMemo } from 'react';
import type { StaffMember, Shift } from '../types';
import './BenchmarkAnalysis.css';

interface BenchmarkAnalysisProps {
  staff: StaffMember[];
  shifts: Shift[];
  requests?: {
    shift_off: any[];
    shift_on: any[];
    cover: any[];
  };
}

const BenchmarkAnalysis: React.FC<BenchmarkAnalysisProps> = ({ staff, shifts, requests }) => {
  const analysis = useMemo(() => {
    if (!staff.length || !shifts.length) return null;

    // 1. Demand (Shift Hours)
    let totalDemandMinutes = 0;
    const roleDemand: Record<string, number> = {};

    shifts.forEach(s => {
      const start = new Date(s.startTime).getTime();
      const end = new Date(s.endTime).getTime();
      const duration = (end - start) / (1000 * 60); // minutes
      totalDemandMinutes += duration;

      roleDemand[s.role] = (roleDemand[s.role] || 0) + duration;
    });

    // 2. Supply (Staff Availability & Contracts)
    let totalSupplyMinutes = 0;
    const roleSupply: Record<string, number> = {};

    staff.forEach(s => {
      // Theoretical max based on contract/preferences
      // Default to 40 hours (2400 mins) if not specified, or use availability
      const desired = s.preferences.desiredHours ? s.preferences.desiredHours * 60 : 2400;
      
      // Calculate actual available time slots overlap (simplified: total slot duration)
      let availableMinutes = 0;
      s.preferences.availability.forEach(slot => {
          const start = new Date(slot.startTime).getTime();
          const end = new Date(slot.endTime).getTime();
          availableMinutes += (end - start) / (1000 * 60);
      });

      // Effective supply is min(contract cap, availability)
      const effectiveSupply = Math.min(desired, availableMinutes);
      totalSupplyMinutes += effectiveSupply;

      // Attribute supply to roles
      s.roles.forEach(role => {
        roleSupply[role] = (roleSupply[role] || 0) + effectiveSupply;
      });
    });

    // 3. Ratios
    const globalRatio = totalSupplyMinutes > 0 ? (totalDemandMinutes / totalSupplyMinutes) : Infinity;
    
    // Role Ratios (Demand / Supply) -> >1 means shortage
    const roleRatios: Record<string, number> = {};
    Object.keys(roleDemand).forEach(role => {
        const supply = roleSupply[role] || 0;
        roleRatios[role] = supply > 0 ? roleDemand[role] / supply : Infinity;
    });

    return {
      totalDemandHours: (totalDemandMinutes / 60).toFixed(1),
      totalSupplyHours: (totalSupplyMinutes / 60).toFixed(1),
      globalUtilization: (globalRatio * 100).toFixed(1),
      roleStats: Object.keys(roleDemand).map(role => ({
        role,
        demand: (roleDemand[role] / 60).toFixed(1),
        supply: ((roleSupply[role] || 0) / 60).toFixed(1),
        ratio: roleRatios[role]
      })).sort((a, b) => b.ratio - a.ratio), // Sort by scarcest first
      requestCounts: {
          off: requests?.shift_off?.length || 0,
          on: requests?.shift_on?.length || 0,
          cover: requests?.cover?.length || 0
      }
    };
  }, [staff, shifts, requests]);

  if (!analysis) return null;

  return (
    <div className="benchmark-analysis">
      <h3>Problem Analysis</h3>
      
      <div className="analysis-grid">
        <div className="analysis-card">
          <h4>Global Load</h4>
          <div className="metric">
            <span className="label">Demand:</span>
            <span className="value">{analysis.totalDemandHours} hrs</span>
          </div>
          <div className="metric">
            <span className="label">Supply:</span>
            <span className="value">{analysis.totalSupplyHours} hrs</span>
          </div>
          <div className={`metric ${Number(analysis.globalUtilization) > 90 ? 'danger' : Number(analysis.globalUtilization) > 75 ? 'warning' : 'success'}`}>
            <span className="label">Utilization:</span>
            <span className="value">{analysis.globalUtilization}%</span>
          </div>
        </div>

        <div className="analysis-card">
          <h4>Constraints</h4>
          <div className="metric">
            <span className="label">Shift-Off Req:</span>
            <span className="value">{analysis.requestCounts.off}</span>
          </div>
          <div className="metric">
            <span className="label">Shift-On Req:</span>
            <span className="value">{analysis.requestCounts.on}</span>
          </div>
          <div className="metric">
            <span className="label">Cover Req:</span>
            <span className="value">{analysis.requestCounts.cover}</span>
          </div>
        </div>
      </div>

      <div className="role-scarcity">
        <h4>Skill Bottlenecks</h4>
        <table>
          <thead>
            <tr>
              <th>Role</th>
              <th>Demand (h)</th>
              <th>Supply (h)</th>
              <th>Load</th>
            </tr>
          </thead>
          <tbody>
            {analysis.roleStats.map(r => (
              <tr key={r.role} className={r.ratio > 1 ? 'danger' : r.ratio > 0.8 ? 'warning' : ''}>
                <td>{r.role}</td>
                <td>{r.demand}</td>
                <td>{r.supply}</td>
                <td>{(r.ratio * 100).toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default BenchmarkAnalysis;
