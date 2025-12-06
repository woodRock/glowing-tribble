import React from 'react';
import { useDraggable } from '@dnd-kit/core';
import './StaffCard.css';

interface StaffCardProps {
  id: string;
  name: string;
  avatarUrl: string;
  isSelected: boolean;
  isDraggable: boolean;
}

const StaffCard: React.FC<StaffCardProps> = ({ id, name, avatarUrl, isSelected, isDraggable }) => {
  const { attributes, listeners, setNodeRef, transform } = useDraggable({
    id: id,
    disabled: !isDraggable,
  });

  const style = transform ? {
    transform: `translate3d(${transform.x}px, ${transform.y}px, 0)`,
  } : undefined;

  const cardClass = isSelected ? "staff-card-container staff-card-container--selected" : "staff-card-container";

  return (
    <div ref={setNodeRef} style={style} {...listeners} {...attributes} className={cardClass}>
      <div className="staff-card">
        <img src={avatarUrl} alt={name} className="staff-card__avatar" />
        <h3 className="staff-card__name">{name}</h3>
      </div>
    </div>
  );
};

export default StaffCard;


