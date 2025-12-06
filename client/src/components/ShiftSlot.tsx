import React, { useState, useEffect } from 'react';
import type { Shift, StaffMember } from '../types.ts';
import { useDroppable } from '@dnd-kit/core';
import StaffCard from './StaffCard';
import './ShiftSlot.css';

interface ShiftSlotProps {
  shift: Shift;
  staffMember: StaffMember | null;
}

const ShiftSlot: React.FC<ShiftSlotProps> = ({ shift, staffMember }) => {
  const { isOver, setNodeRef } = useDroppable({
    id: shift.id,
  });
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

  const slotClass = isOver ? "shift-slot shift-slot--over" : "shift-slot";

  return (
    <div ref={setNodeRef} className={slotClass}>
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
          <StaffCard
            id={staffMember.id}
            name={staffMember.name}
            avatarUrl={`https://api.dicebear.com/8.x/adventurer/svg?seed=${staffMember.id}`}
            isSelected={false}
            isDraggable={false}
          />
        ) : (
          <div className="unassigned-slot">Unassigned</div>
        )}
      </div>
    </div>
  );
};

export default ShiftSlot;


