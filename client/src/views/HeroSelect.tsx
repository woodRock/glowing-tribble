import React, { useEffect, useState } from 'react';
import type { StaffMember, Shift } from '../types.ts';
import DailyRoster from '../components/DailyRoster';
import AvailableStaff from '../components/AvailableStaff';
import './HeroSelect.css';

const weekDays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

const HeroSelect: React.FC = () => {
  const [allStaff, setAllStaff] = useState<StaffMember[]>([]);
  const [allShifts, setAllShifts] = useState<Shift[]>([]);
  const [selectedDay, setSelectedDay] = useState(0); // 0-6 for days, 7 for week view
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch('http://localhost:4000/api/staff').then(res => res.json()),
      fetch('http://localhost:4000/api/shifts').then(res => res.json()),
    ])
    .then(([staffData, shiftsData]) => {
      setAllStaff(staffData);
      setAllShifts(shiftsData);
      setLoading(false);
    })
    .catch(err => {
      setError(err.message);
      setLoading(false);
    });
  }, []);

  const handleAssignStaff = (shift: Shift, staffMember: StaffMember) => {
    // This is where the logic to assign a staff member to a shift would go.
    // For now, it just logs the action.
    console.log(`Assigning ${staffMember.name} to ${shift.role} shift`);
  };

  const handleGenerateRoster = () => {
    fetch('http://localhost:4000/api/roster/generate', { method: 'POST' })
      .then(res => res.json())
      .then((generatedRoster: Shift[]) => {
        setAllShifts(generatedRoster);
      })
      .catch(err => console.error('Error generating roster:', err));
  };

  const getShiftsForDay = (dayIndex: number) => {
    if (dayIndex > 6) return allShifts; // Week view
    const day = new Date(new Date('2025-12-15T00:00:00').getTime() + dayIndex * 24 * 60 * 60 * 1000);
    return allShifts.filter(shift => new Date(shift.startTime).getDate() === day.getDate());
  };

  const shiftsForDay = getShiftsForDay(selectedDay);
  const selectedStaffIdsForDay = shiftsForDay.map(s => s.staffMemberId).filter(Boolean) as string[];

  if (loading) return <div className="hero-select__message">Loading...</div>;
  if (error) return <div className="hero-select__message hero-select__message--error">Error: {error}</div>;

  return (
    <div className="hero-select">
      <div className="hero-select__main-content">
        <div className="hero-select__title-container">
          <h2 className="hero-select__team-title">
            {selectedDay > 6 ? 'Full Week Roster' : `${weekDays[selectedDay]}'s Roster`}
          </h2>
          <div className="hero-select__nav-buttons">
            <button onClick={() => setSelectedDay(prev => Math.max(0, prev - 1))}>Prev Day</button>
            <button onClick={() => setSelectedDay(prev => Math.min(6, prev + 1))}>Next Day</button>
            <button onClick={() => setSelectedDay(7)}>This Week</button>
          </div>
          <button className="hero-select__generate-button" onClick={handleGenerateRoster}>
            Generate Roster
          </button>
        </div>
        <DailyRoster
          shifts={shiftsForDay}
          staff={allStaff}
          onAssign={handleAssignStaff}
        />
      </div>
      <AvailableStaff
        allStaff={allStaff}
        selectedStaffIds={selectedStaffIdsForDay}
        onSelectStaff={() => { /* Needs implementation */ }}
      />
    </div>
  );
};

export default HeroSelect;