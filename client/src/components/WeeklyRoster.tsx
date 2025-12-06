import React from 'react';
import * as Types from '../types';
import './WeeklyRoster.css';
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

interface WeeklyRosterProps {
  shifts: Types.Shift[];
  staff: Types.StaffMember[];
  selectedDate: Date;
  onShiftClick: (shift: Types.Shift) => void;
  conflicts: Types.Conflict[];
}

const weekDays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

const WeeklyRoster: React.FC<WeeklyRosterProps> = ({ shifts, staff, selectedDate, onShiftClick, conflicts }) => {
  const getStaffMember = (staffId: string | undefined) => {
    if (!staffId) return null;
    return staff.find(s => s.id === staffId) || null;
  };

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

  const getWeekStart = (date: Date): Date => {
    const d = new Date(date);
    const day = d.getDay(); // 0 for Sunday, 6 for Saturday
    const diff = d.getDate() - day + (day === 0 ? -6 : 1); // Adjust to Monday
    d.setDate(diff);
    d.setHours(0, 0, 0, 0);
    return d;
  };

  const weekStart = getWeekStart(selectedDate);
  const daysOfWeek: Date[] = [];
  for (let i = 0; i < 7; i++) {
    const day = new Date(weekStart);
    day.setDate(weekStart.getDate() + i);
    daysOfWeek.push(day);
  }

  const shiftsByDay: { [key: string]: Types.Shift[] } = {};
  daysOfWeek.forEach(day => {
    shiftsByDay[day.toDateString()] = [];
  });

  shifts.forEach(shift => {
    const shiftStartTime = new Date(shift.startTime);
    const shiftDayString = shiftStartTime.toDateString();
    if (shiftsByDay[shiftDayString]) {
      shiftsByDay[shiftDayString].push(shift);
    }
  });

  return (
    <div className="weekly-roster">
      <h2>Weekly Roster for the week of {weekStart.toDateString()}</h2>
      <div className="weekly-roster__grid">
        {daysOfWeek.map(day => (
          <div key={day.toDateString()} className="weekly-roster__day">
            <h3>{weekDays[day.getDay()]} - {day.getDate()}/{day.getMonth() + 1}</h3>
            <div className="weekly-roster__day-content">
              {shiftsByDay[day.toDateString()].length > 0 ? (
                shiftsByDay[day.toDateString()].map(shift => {
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
                })
              ) : (
                <p>No shifts</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default WeeklyRoster;