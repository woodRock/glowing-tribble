import React from 'react';
import './StaffCard.css';

interface StaffCardProps {
  name: string;
  avatarUrl: string;
  isSelected: boolean;
}

const StaffCard: React.FC<StaffCardProps> = ({ name, avatarUrl, isSelected }) => {
  const cardClass = isSelected ? "staff-card-container staff-card-container--selected" : "staff-card-container";
  return (
    <div className={cardClass}>
      <div className="staff-card">
        <img src={avatarUrl} alt={name} className="staff-card__avatar" />
        <h3 className="staff-card__name">{name}</h3>
      </div>
    </div>
  );
};

export default StaffCard;
