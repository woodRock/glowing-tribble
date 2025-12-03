import React from 'react';
import StaffCard from './StaffCard';
import type { StaffMember, Role } from '../types.ts';
import './AvailableStaff.css';

interface AvailableStaffProps {
  allStaff: StaffMember[];
  selectedStaffIds: string[];
  onSelectStaff: (staffMember: StaffMember) => void;
}

const AvailableStaff: React.FC<AvailableStaffProps> = ({ allStaff, selectedStaffIds, onSelectStaff }) => {
  const roles: Role[] = ["bar", "restaurant", "maitre'd", "duty manager"];

  const getStaffByRole = (role: Role) => {
    return allStaff.filter(staff => staff.roles.includes(role));
  };

  return (
    <div className="available-staff">
      {roles.map(role => (
        <div key={role} className="available-staff__role-group">
          <h3 className="available-staff__role-title">{role}</h3>
          <div className="available-staff__grid">
            {getStaffByRole(role).map(staff => (
              <div key={staff.id} onClick={() => onSelectStaff(staff)}>
                <StaffCard
                  name={staff.name}
                  avatarUrl={`https://api.dicebear.com/8.x/adventurer/svg?seed=${staff.id}`}
                  isSelected={selectedStaffIds.includes(staff.id)}
                />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

export default AvailableStaff;