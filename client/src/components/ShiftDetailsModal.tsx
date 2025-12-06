import React from 'react';
import type { Shift, StaffMember, Role } from '../types';
import './ShiftDetailsModal.css';

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

interface ShiftDetailsModalProps {
  shift: Shift | null;
  staff: StaffMember[];
  onClose: () => void;
}

const ShiftDetailsModal: React.FC<ShiftDetailsModalProps> = ({ shift, staff, onClose }) => {
  if (!shift) {
    return null;
  }

  const assignedStaff = shift.staffMemberId ? staff.find(s => s.id === shift.staffMemberId) : null;

  return (
    <div className="shift-details-modal-overlay" onClick={onClose}>
      <div className="shift-details-modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="shift-details-modal-close" onClick={onClose}>&times;</button>
        <h2>Shift Details</h2>
        <div className="shift-details-modal-body">
          <p><strong>ID:</strong> {shift.id}</p>
          <p>
            <strong>Role:</strong> 
            <img src={roleIcons[shift.role]} alt={shift.role} className="shift-details-modal-icon" />
            {shift.role}
          </p>
          <p>
            <strong>Time:</strong> {new Date(shift.startTime).toLocaleString()} - {new Date(shift.endTime).toLocaleString()}
          </p>
          <p>
            <strong>Assigned Staff:</strong> 
            {assignedStaff ? (
              <>
                <img src={assignedStaff.avatar} alt={assignedStaff.name} className="shift-details-modal-avatar" />
                {assignedStaff.name}
              </>
            ) : (
              'Unassigned'
            )}
          </p>
        </div>
      </div>
    </div>
  );
};

export default ShiftDetailsModal;
