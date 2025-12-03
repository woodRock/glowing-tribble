import React from 'react';
import StaffCard from './StaffCard';
import type { StaffMember } from '../types.ts';
import './TeamSlot.css';

interface TeamSlotProps {
  staffMember: StaffMember | null;
  role: string;
  onClick: () => void;
}

const TeamSlot: React.FC<TeamSlotProps> = ({ staffMember, role, onClick }) => {
  return (
    <div className="team-slot" onClick={onClick}>
      {staffMember ? (
        <StaffCard
          name={staffMember.name}
          avatarUrl={`https://api.dicebear.com/8.x/adventurer/svg?seed=${staffMember.id}`}
        />
      ) : (
        <div className="team-slot__placeholder">
          <div className="team-slot__role-icon">{/* Icon */}</div>
          <div className="team-slot__role-name">{role}</div>
        </div>
      )}
    </div>
  );
};

export default TeamSlot;