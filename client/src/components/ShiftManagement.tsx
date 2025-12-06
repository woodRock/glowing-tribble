import React, { useState, useEffect } from 'react';
import * as Types from '../types';
import './ShiftManagement.css';

const allRoles: Types.Role[] = ["bar", "restaurant", "maitre'd", "duty manager"];

const ShiftManagement: React.FC = () => {
  const [shifts, setShifts] = useState<Types.Shift[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState<Partial<Types.Shift>>({
    role: allRoles[0], // Default to first role
    startTime: '',
    endTime: '',
  });
  const [editingShiftId, setEditingShiftId] = useState<string | null>(null);

  const fetchShifts = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:4000/api/shifts');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: Types.Shift[] = await response.json();
      setShifts(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchShifts();
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    // Basic validation
    if (!formData.role || !formData.startTime || !formData.endTime) {
      setError('Please fill in all required fields (Role, Start Time, End Time).');
      return;
    }

    try {
      let response;
      if (editingShiftId) {
        response = await fetch(`http://localhost:4000/api/shifts/${editingShiftId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formData),
        });
      } else {
        const newId = String(shifts.length > 0 ? Math.max(...shifts.map(s => parseInt(s.id))) + 1 : 1);
        response = await fetch('http://localhost:4000/api/shifts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...formData, id: newId } as Types.Shift),
        });
      }

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.message || `HTTP error! status: ${response.status}`);
      }

      setFormData({
        role: allRoles[0],
        startTime: '',
        endTime: '',
      });
      setEditingShiftId(null);
      fetchShifts();
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleEdit = (shift: Types.Shift) => {
    // Format dates for datetime-local input
    const formatDateTimeLocal = (isoString: string) => {
      if (!isoString) return '';
      const date = new Date(isoString);
      date.setMinutes(date.getMinutes() - date.getTimezoneOffset()); // Adjust for timezone
      return date.toISOString().slice(0, 16);
    };

    setFormData({
      ...shift,
      startTime: formatDateTimeLocal(shift.startTime),
      endTime: formatDateTimeLocal(shift.endTime),
    });
    setEditingShiftId(shift.id);
  };

  const handleDelete = async (id: string) => {
    setError(null);
    try {
      const response = await fetch(`http://localhost:4000/api/shifts/${id}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.message || `HTTP error! status: ${response.status}`);
      }
      fetchShifts();
    } catch (err: any) {
      setError(err.message);
    }
  };

  if (loading) return <div className="loading-message">Loading shifts...</div>;
  if (error) return <div className="error-message">Error: {error}</div>;

  return (
    <div className="shift-management">
      <h2>Shift Management</h2>

      <div className="shift-form">
        <h3>{editingShiftId ? 'Edit Shift' : 'Add New Shift'}</h3>
        {error && <div className="error-message">{error}</div>}
        <form onSubmit={handleSubmit}>
          <div>
            <label htmlFor="role">Role:</label>
            <select
              id="role"
              name="role"
              value={formData.role || ''}
              onChange={handleInputChange}
              required
            >
              {allRoles.map(role => (
                <option key={role} value={role}>{role}</option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="startTime">Start Time:</label>
            <input
              type="datetime-local"
              id="startTime"
              name="startTime"
              value={formData.startTime || ''}
              onChange={handleInputChange}
              required
            />
          </div>
          <div>
            <label htmlFor="endTime">End Time:</label>
            <input
              type="datetime-local"
              id="endTime"
              name="endTime"
              value={formData.endTime || ''}
              onChange={handleInputChange}
              required
            />
          </div>
          <button type="submit">{editingShiftId ? 'Update Shift' : 'Add Shift'}</button>
          {editingShiftId && (
            <button type="button" onClick={() => {
              setEditingShiftId(null);
              setFormData({
                role: allRoles[0],
                startTime: '',
                endTime: '',
              });
            }}>Cancel Edit</button>
          )}
        </form>
      </div>

      <h3>Current Shifts</h3>
      <ul className="shift-list">
        {shifts.map(shift => (
          <li key={shift.id} className="shift-list-item">
            <div className="shift-info">
              <h3>{shift.role}</h3>
              <p>Start: {new Date(shift.startTime).toLocaleString()}</p>
              <p>End: {new Date(shift.endTime).toLocaleString()}</p>
              {shift.staffMemberId && <p>Assigned: {shift.staffMemberId}</p>}
            </div>
            <div className="shift-actions">
              <button className="edit" onClick={() => handleEdit(shift)}>Edit</button>
              <button className="delete" onClick={() => handleDelete(shift.id)}>Delete</button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ShiftManagement;
