import React from 'react';
import type { Shift, StaffMember, Role } from '../types';
import './StaffRoster.css';

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

interface StaffRosterProps {
  staffMember: StaffMember;
  shifts: Shift[];
}

const weekDays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

const StaffRoster: React.FC<StaffRosterProps> = ({ staffMember, shifts }) => {
  const staffShifts = shifts.filter(shift => shift.staffMemberId === staffMember.id);

  const shiftsByDay: { [day: string]: Shift[] } = {};
  weekDays.forEach(day => {
    shiftsByDay[day] = [];
  });

  staffShifts.forEach(shift => {
    const dayIndex = new Date(shift.startTime).getDay();
    const day = weekDays[dayIndex];
    shiftsByDay[day].push(shift);
  });

  return (
    <div className="staff-roster">
      <h2>Roster for {staffMember.name}</h2>
      <div className="staff-roster__details">
        <img src={staffMember.avatar} alt={staffMember.name} className="staff-roster__avatar" />
        <p>Roles: {staffMember.roles.join(', ')}</p>
        <p>Desired Hours: {staffMember.preferences.desiredHours}</p>
      </div>
      <div className="staff-roster__shifts-grid">
        {weekDays.map(day => (
          <div key={day} className="staff-roster__day-slot">
            <h3>{day}</h3>
            {shiftsByDay[day].length > 0 ? (
              shiftsByDay[day].map(shift => (
                <div key={shift.id} className="staff-roster__shift">
                  <div className="staff-roster__shift-header">
                    <img src={roleIcons[shift.role]} alt={shift.role} className="staff-roster__shift-icon" />
                    <div className="staff-roster__shift-role">{shift.role}</div>
                  </div>
                  <div className="staff-roster__shift-time">
                    {new Date(shift.startTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} -
                    {new Date(shift.endTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              ))
            ) : (
              <p>No shifts</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default StaffRoster;
