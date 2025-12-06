import React, { useState, useEffect } from 'react';
import * as Types from '../types'; // Import all as Types
import './StaffManagement.css';

const allRoles: Types.Role[] = ["bar", "restaurant", "maitre'd", "duty manager"];

const StaffManagement: React.FC = () => {
  const [staff, setStaff] = useState<Types.StaffMember[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState<Partial<Types.StaffMember>>({
    name: '',
    roles: [],
    avatar: '',
    preferences: {
      availability: [],
      desiredHours: 0,
      prefersConsecutiveDaysOff: false,
    },
  });
  const [editingStaffId, setEditingStaffId] = useState<string | null>(null);

  const fetchStaff = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:4000/api/staff');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: Types.StaffMember[] = await response.json();
      setStaff(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStaff();
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    if (name.startsWith('preferences.')) {
      const prefName = name.split('.')[1];
      setFormData(prev => ({
        ...prev,
        preferences: {
          ...prev.preferences!,
          [prefName]: value,
        },
      }));
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
  };

  const handleCheckboxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { value, checked } = e.target;
    setFormData(prev => {
      const currentRoles = prev.roles || [];
      return {
        ...prev,
        roles: checked
          ? [...currentRoles, value as Types.Role]
          : currentRoles.filter(role => role !== value),
      };
    });
  };

  const handlePreferenceCheckboxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      preferences: {
        ...prev.preferences!,
        [name]: checked,
      },
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    // Basic validation
    if (!formData.name || formData.roles?.length === 0 || !formData.avatar) {
      setError('Please fill in all required fields (Name, Roles, Avatar).');
      return;
    }
    if (!formData.preferences?.desiredHours || formData.preferences.desiredHours <= 0) {
      setError('Desired hours must be a positive number.');
      return;
    }

    try {
      let response;
      if (editingStaffId) {
        response = await fetch(`http://localhost:4000/api/staff/${editingStaffId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formData),
        });
      } else {
        // Generate a simple ID for new staff members
        const newId = String(staff.length > 0 ? Math.max(...staff.map(s => parseInt(s.id))) + 1 : 1);
        response = await fetch('http://localhost:4000/api/staff', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...formData, id: newId } as Types.StaffMember), // Cast to Types.StaffMember
        });
      }

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.message || `HTTP error! status: ${response.status}`);
      }

      setFormData({
        name: '',
        roles: [],
        avatar: '',
        preferences: {
          availability: [],
          desiredHours: 0,
          prefersConsecutiveDaysOff: false,
        },
      });
      setEditingStaffId(null);
      fetchStaff(); // Refresh the list
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleEdit = (staffMember: Types.StaffMember) => {
    setFormData(staffMember);
    setEditingStaffId(staffMember.id);
  };

  const handleDelete = async (id: string) => {
    setError(null);
    try {
      const response = await fetch(`http://localhost:4000/api/staff/${id}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.message || `HTTP error! status: ${response.status}`);
      }
      fetchStaff(); // Refresh the list
    } catch (err: any) {
      setError(err.message);
    }
  };

  if (loading) return <div className="loading-message">Loading staff...</div>;
  if (error) return <div className="error-message">Error: {error}</div>;

  return (
    <div className="staff-management">
      <h2>Staff Management</h2>

      <div className="staff-form">
        <h3>{editingStaffId ? 'Edit Staff Member' : 'Add New Staff Member'}</h3>
        {error && <div className="error-message">{error}</div>}
        <form onSubmit={handleSubmit}>
          <div>
            <label htmlFor="name">Name:</label>
            <input
              type="text"
              id="name"
              name="name"
              value={formData.name || ''}
              onChange={handleInputChange}
              required
            />
          </div>
          <div>
            <label>Roles:</label>
            <div className="roles-checkboxes">
              {allRoles.map(role => (
                <label key={role}>
                  <input
                    type="checkbox"
                    value={role}
                    checked={formData.roles?.includes(role) || false}
                    onChange={handleCheckboxChange}
                  />
                  {role}
                </label>
              ))}
            </div>
          </div>
          <div>
            <label htmlFor="avatar">Avatar URL:</label>
            <input
              type="text"
              id="avatar"
              name="avatar"
              value={formData.avatar || ''}
              onChange={handleInputChange}
              required
            />
          </div>
          <div>
            <label htmlFor="desiredHours">Desired Hours (per week):</label>
            <input
              type="number"
              id="desiredHours"
              name="preferences.desiredHours"
              value={formData.preferences?.desiredHours || 0}
              onChange={handleInputChange}
              min="0"
              required
            />
          </div>
          <div>
            <label>
              <input
                type="checkbox"
                name="prefersConsecutiveDaysOff"
                checked={formData.preferences?.prefersConsecutiveDaysOff || false}
                onChange={handlePreferenceCheckboxChange}
              />
              Prefers Consecutive Days Off
            </label>
          </div>
          {/* Availability is more complex, will add later if needed */}
          <button type="submit">{editingStaffId ? 'Update Staff' : 'Add Staff'}</button>
          {editingStaffId && (
            <button type="button" onClick={() => {
              setEditingStaffId(null);
              setFormData({
                name: '',
                roles: [],
                avatar: '',
                preferences: {
                  availability: [],
                  desiredHours: 0,
                  prefersConsecutiveDaysOff: false,
                },
              });
            }}>Cancel Edit</button>
          )}
        </form>
      </div>

      <h3>Current Staff</h3>
      <ul className="staff-list">
        {staff.map(member => (
          <li key={member.id} className="staff-list-item">
            <img src={member.avatar} alt={member.name} />
            <div className="staff-info">
              <h3>{member.name}</h3>
              <p>Roles: {member.roles.join(', ')}</p>
              <p>Desired Hours: {member.preferences.desiredHours}</p>
              <p>Prefers Consecutive Days Off: {member.preferences.prefersConsecutiveDaysOff ? 'Yes' : 'No'}</p>
            </div>
            <div className="staff-actions">
              <button className="edit" onClick={() => handleEdit(member)}>Edit</button>
              <button className="delete" onClick={() => handleDelete(member.id)}>Delete</button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default StaffManagement;
