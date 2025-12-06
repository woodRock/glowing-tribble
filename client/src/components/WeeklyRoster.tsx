import React from 'react';
import type { Shift, StaffMember } from '../types';
import './WeeklyRoster.css';

interface WeeklyRosterProps {
  shifts: Shift[];
  staff: StaffMember[];
}

const weekDays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
const timeSlots = ['Morning', 'Afternoon', 'Evening'];

const getTimeSlot = (shift: Shift): string => {
  const startHour = new Date(shift.startTime).getHours();
  if (startHour < 12) return 'Morning';
  if (startHour < 17) return 'Afternoon';
  return 'Evening';
};

const WeeklyRoster: React.FC<WeeklyRosterProps> = ({ shifts, staff }) => {
  const getStaffMember = (staffId: string | undefined) => {
    if (!staffId) return null;
    return staff.find(s => s.id === staffId) || null;
  };

  const shiftsByDayAndSlot: { [day: string]: { [slot: string]: Shift[] } } = {};
  weekDays.forEach(day => {
    shiftsByDayAndSlot[day] = {};
    timeSlots.forEach(slot => {
      shiftsByDayAndSlot[day][slot] = [];
    });
  });

  shifts.forEach(shift => {
    const dayIndex = new Date(shift.startTime).getDay();
    const day = weekDays[dayIndex === 0 ? 6 : dayIndex - 1]; // Adjust for Sunday being 0
    const slot = getTimeSlot(shift);
    if (day && slot) {
      shiftsByDayAndSlot[day][slot].push(shift);
    }
  });

  return (
    <div className="weekly-roster">
      <table className="weekly-roster__table">
        <thead>
          <tr>
            <th>Day</th>
            {timeSlots.map(slot => <th key={slot}>{slot}</th>)}
          </tr>
        </thead>
        <tbody>
          {weekDays.map(day => (
            <tr key={day}>
              <td>{day}</td>
              {timeSlots.map(slot => (
                <td key={slot}>
                  {shiftsByDayAndSlot[day][slot].map(shift => (
                    <div key={shift.id} className="weekly-roster__shift">
                      <div className="weekly-roster__shift-role">{shift.role}</div>
                      <div className="weekly-roster__shift-staff">
                        {getStaffMember(shift.staffMemberId)?.name || 'Unassigned'}
                      </div>
                    </div>
                  ))}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default WeeklyRoster;
