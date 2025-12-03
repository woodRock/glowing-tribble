import React, { useState, useEffect } from 'react';
import type { Shift, StaffMember } from '../types.ts';
import './ShiftSlot.css';

interface ShiftSlotProps {
  shift: Shift;
  staffMember: StaffMember | null;
}

const ShiftSlot: React.FC<ShiftSlotProps> = ({ shift, staffMember }) => {
  const [icon, setIcon] = useState('');

  useEffect(() => {
    const importIcon = async () => {
      try {
        const iconModule = await import(`../assets/icons/${shift.role.replace(/'/g, '').replace(' ', '')}.svg`);
        setIcon(iconModule.default);
      } catch (e) {
        console.error(e);
      }
    };
    importIcon();
  }, [shift.role]);

  return (
    <div className="shift-slot">
      <div className="shift-info">
        {icon && <img src={icon} alt={`${shift.role} icon`} className="shift-role-icon" />}
        <div className="shift-details">
          <div className="shift-role">{shift.role}</div>
          <div className="shift-time">
            {new Date(shift.startTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} - 
            {new Date(shift.endTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </div>
        </div>
      </div>
      <div className="shift-assignment">
        {staffMember ? (
          <div className="assigned-staff">{staffMember.name}</div>
        ) : (
          <div className="unassigned-slot">Unassigned</div>
        )}
      </div>
    </div>
  );
};

export default ShiftSlot;
