import React from 'react';
import * as Types from '../types'; // Import all as Types
import './DailyRoster.css';
import { useDroppable } from '@dnd-kit/core';

// Import icons
import BarIcon from '../assets/icons/bar.svg';
import DutyManagerIcon from '../assets/icons/dutymanager.svg';
import MaitredIcon from '../assets/icons/maitred.svg';
import RestaurantIcon from '../assets/icons/restaurant.svg';

// Map roles to icons
const roleIcons: Record<Types.Role, string> = {
  'bar': BarIcon,
  'duty manager': DutyManagerIcon,
  'maitre\'d': MaitredIcon,
  'restaurant': RestaurantIcon,
};

interface DailyRosterProps {
  shifts: Types.Shift[];
  staff: Types.StaffMember[];
  selectedDate: Date;
  onShiftClick: (shift: Types.Shift) => void;
  conflicts: Types.Conflict[]; // New prop
}

const timeSlots = ['Morning (8am-12pm)', 'Afternoon (12pm-5pm)', 'Evening (5pm-1am)'];

const getTimeSlotCategory = (shift: Types.Shift): string => {
  const startHour = new Date(shift.startTime).getHours();
  if (startHour >= 8 && startHour < 12) return 'Morning (8am-12pm)';
  if (startHour >= 12 && startHour < 17) return 'Afternoon (12pm-5pm)';
  if (startHour >= 17 || startHour < 1) return 'Evening (5pm-1am)';
  return 'Other';
};

const DailyRoster: React.FC<DailyRosterProps> = ({ shifts, staff, selectedDate, onShiftClick, conflicts }) => {
  const getStaffMember = (staffId: string | undefined) => {
    if (!staffId) return null;
    return staff.find(s => s.id === staffId) || null;
  };

  const shiftsBySlot: { [slot: string]: Types.Shift[] } = {};
  timeSlots.forEach(slot => {
    shiftsBySlot[slot] = [];
  });

  const startOfDay = new Date(selectedDate);
  startOfDay.setHours(0, 0, 0, 0);
  const endOfDay = new Date(selectedDate);
  endOfDay.setHours(23, 59, 59, 999);

  shifts.forEach(shift => {
    const shiftStartTime = new Date(shift.startTime);
    if (shiftStartTime >= startOfDay && shiftStartTime <= endOfDay) {
      const slot = getTimeSlotCategory(shift);
      if (shiftsBySlot[slot]) {
        shiftsBySlot[slot].push(shift);
      }
    }
  });

  const getShiftConflicts = (shiftId: string, staffMemberId?: string): Types.Conflict[] => {
    const hardConflictTypes: Types.Conflict['type'][] = [
      'double_booking',
      'unassigned_shift',
      'role_mismatch',
      'availability_violation',
      'insufficient_rest',
      'exceeds_consecutive_days',
    ];

    return conflicts.filter(conflict =>
      hardConflictTypes.includes(conflict.type) &&
      (conflict.shiftId === shiftId || (staffMemberId && conflict.staffMemberId === staffMemberId))
    );
  };

  return (
    <div className="daily-roster">
      <h2>Roster for {selectedDate.toDateString()}</h2>
      <div className="daily-roster__grid">
        {timeSlots.map(slot => (
          <div key={slot} className="daily-roster__slot">
            <h3>{slot}</h3>
            {shiftsBySlot[slot].length > 0 ? (
              <>
                {shiftsBySlot[slot].map(shift => {
                  const { setNodeRef } = useDroppable({ id: shift.id });
                  const shiftConflicts = getShiftConflicts(shift.id, shift.staffMemberId);
                  const hasConflict = shiftConflicts.length > 0;
                  return (
                    <div
                      key={shift.id}
                      ref={setNodeRef}
                      className={`daily-roster__shift ${hasConflict ? 'daily-roster__shift--conflict' : ''}`}
                      onClick={() => onShiftClick(shift)}
                    >
                      <div className="daily-roster__shift-header">
                        <img src={roleIcons[shift.role]} alt={shift.role} className="daily-roster__shift-icon" />
                        <div className="daily-roster__shift-role">{shift.role}</div>
                        {hasConflict && (
                          <span className="daily-roster__conflict-indicator" title={shiftConflicts.map(c => c.message).join('\n')}>⚠️</span>
                        )}
                      </div>
                      <div className="daily-roster__shift-time">
                        {new Date(shift.startTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} -
                        {new Date(shift.endTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </div>
                      <div className="daily-roster__shift-staff">
                        {shift.staffMemberId ? (
                          <>
                            <img src={getStaffMember(shift.staffMemberId)?.avatar} alt={getStaffMember(shift.staffMemberId)?.name} className="daily-roster__staff-avatar" />
                            <span>{getStaffMember(shift.staffMemberId)?.name}</span>
                          </>
                        ) : (
                          'Unassigned'
                        )}
                      </div>
                    </div>
                  );
                })}
              </>
            ) : (
              <p>No shifts</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default DailyRoster;

