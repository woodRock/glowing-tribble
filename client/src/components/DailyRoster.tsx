import React from 'react';
import type { Shift, StaffMember } from '../types.ts';
import ShiftSlot from './ShiftSlot';
import './DailyRoster.css';

interface DailyRosterProps {
  shifts: Shift[];
  staff: StaffMember[];
  onAssign: (shift: Shift, staffMember: StaffMember) => void;
}

const DailyRoster: React.FC<DailyRosterProps> = ({ shifts, staff, onAssign }) => {
  const getStaffMember = (staffId: string | undefined) => {
    if (!staffId) return null;
    return staff.find(s => s.id === staffId) || null;
  };

  return (
    <div className="daily-roster">
      {shifts.map(shift => (
        <ShiftSlot
          key={shift.id}
          shift={shift}
          staffMember={getStaffMember(shift.staffMemberId)}
        />
      ))}
    </div>
  );
};

export default DailyRoster;