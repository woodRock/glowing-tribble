import React from 'react';
import type { Shift, StaffMember, Role } from '../types';
import './WeeklyRoster.css';
import { useDroppable } from '@dnd-kit/core'; // Import useDroppable

// Import icons
import BarIcon from '../assets/icons/bar.svg';
import DutyManagerIcon from '../assets/icons/dutymanager.svg';
import MaitredIcon from '../assets/icons/maitred.svg';
import RestaurantIcon from '../assets/icons/restaurant.svg';

// Map roles to icons
const roleIcons: Record<Role, string> = {
  'bar': BarIcon,
  'duty manager': DutyManagerIcon,
  'maitre\'d': MaitredIcon,
  'restaurant': RestaurantIcon,
};

interface WeeklyRosterProps {
  shifts: Shift[];
  staff: StaffMember[];
  onShiftClick: (shift: Shift) => void; // New prop
}

const weekDays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
const timeSlots = ['Morning', 'Afternoon', 'Evening'];

const getTimeSlot = (shift: Shift): string => {
  const startHour = new Date(shift.startTime).getHours();
  if (startHour < 12) return 'Morning';
  if (startHour < 17) return 'Afternoon';
  return 'Evening';
};

const WeeklyRoster: React.FC<WeeklyRosterProps> = ({ shifts, staff, onShiftClick }) => {
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
                  {shiftsByDayAndSlot[day][slot].map(shift => {
                    const { setNodeRef } = useDroppable({ id: shift.id });
                    return (
                      <div 
                        key={shift.id} 
                        ref={setNodeRef} // Set the droppable ref
                        className="weekly-roster__shift" 
                        onClick={() => onShiftClick(shift)}
                      >
                        <div className="weekly-roster__shift-header">
                          <img src={roleIcons[shift.role]} alt={shift.role} className="weekly-roster__shift-icon" />
                          <div className="weekly-roster__shift-role">{shift.role}</div>
                        </div>
                        <div className="weekly-roster__shift-time">
                          {new Date(shift.startTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} -
                          {new Date(shift.endTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
                        <div className="weekly-roster__shift-staff">
                          {shift.staffMemberId ? (
                            <>
                              <img src={getStaffMember(shift.staffMemberId)?.avatar} alt={getStaffMember(shift.staffMemberId)?.name} className="weekly-roster__staff-avatar" />
                              <span>{getStaffMember(shift.staffMemberId)?.name}</span>
                            </>
                          ) : (
                            'Unassigned'
                          )}
                        </div>
                      </div>
                    );
                  })}
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
