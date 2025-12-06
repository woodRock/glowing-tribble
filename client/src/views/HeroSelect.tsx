import React, { useEffect, useState, useCallback } from 'react';
import type { StaffMember, Shift, RosterMetrics, ProgramNode } from '../types.ts';
import DailyRoster from '../components/DailyRoster';
import AvailableStaff from '../components/AvailableStaff';
import WeeklyRoster from '../components/WeeklyRoster';
import StaffRoster from '../components/StaffRoster';
import RosterMetricsDisplay from '../components/RosterMetrics';
import FitnessChart from '../components/FitnessChart';
import ProgramTree from '../components/ProgramTree';
import ShiftDetailsModal from '../components/ShiftDetailsModal'; // Import ShiftDetailsModal
import { DndContext } from '@dnd-kit/core';
import type { DragEndEvent } from '@dnd-kit/core';
import './HeroSelect.css';

const weekDays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
type Algorithm = 'greedy' | 'ga' | 'gp' | 'cp' | 'aco' | 'hybrid_ga_cp' | 'neural_network' | 'pso'; // Add PSO
type RosterView = 'weekly' | 'daily' | 'staff';

const HeroSelect: React.FC = () => {
  const [allStaff, setAllStaff] = useState<StaffMember[]>([]);
  const [allShifts, setAllShifts] = useState<Shift[]>([]);
  const [selectedDate, setSelectedDate] = useState<Date>(new Date('2025-12-15T00:00:00'));
  const [rosterView, setRosterView] = useState<RosterView>('weekly');
  const [selectedStaffMemberId, setSelectedStaffMemberId] = useState<string | null>(null);
  const [selectedShiftForDetails, setSelectedShiftForDetails] = useState<Shift | null>(null); // New state for modal
  const [isShiftModalOpen, setIsShiftModalOpen] = useState(false); // New state for modal
  const [loading, setLoading] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<RosterMetrics | null>(null);
  const [rosterConflicts, setRosterConflicts] = useState<Types.Conflict[]>([]); // New state for conflicts
  const [gaHistory, setGaHistory] = useState<number[]>([]);
  const [gpProgramHistory, setGpProgramHistory] = useState<ProgramNode[]>([]);
  const [selectedGeneration, setSelectedGeneration] = useState(0);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<Algorithm>('greedy');
  const [numGenerations, setNumGenerations] = useState<number>(50); // New state for numGenerations

  const fetchMetrics = useCallback((roster: Shift[]) => {
    fetch('http://localhost:4000/api/roster/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(roster),
    })
    .then(res => res.json())
    .then(data => {
      setMetrics(data);
      setRosterConflicts(data.conflicts || []); // Store conflicts
    })
    .catch(err => console.error('Error fetching metrics:', err));
  }, []);

  useEffect(() => {
    Promise.all([
      fetch('http://localhost:4000/api/staff').then(res => res.json()),
      fetch('http://localhost:4000/api/roster').then(res => res.json()),
    ])
    .then(([staffData, rosterData]) => {
      setAllStaff(staffData);
      setAllShifts(rosterData);
      fetchMetrics(rosterData);
      setLoading(false);
    })
    .catch(err => {
      setError(err.message);
      setLoading(false);
    });
  }, [fetchMetrics]);

  const handleGenerateRoster = () => {
    setIsGenerating(true);
    setMetrics(null);
    setGaHistory([]);
    setGpProgramHistory([]);

    const emptyRoster = allShifts.map(shift => ({ ...shift, staffMemberId: undefined }));
    setAllShifts(emptyRoster);

    fetch(`http://localhost:4000/api/roster/generate/${selectedAlgorithm}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ numGenerations: numGenerations }), // Pass numGenerations
    })
      .then(res => res.json())
      .then(result => {
        if (selectedAlgorithm === 'greedy') {
          const finalRoster = result.roster;
          if (!finalRoster) {
            console.error('Greedy algorithm did not return a roster.');
            setIsGenerating(false);
            return;
          }
          let i = 0;
          const interval = setInterval(() => {
            if (i < finalRoster.length) {
              const shiftToUpdate = finalRoster[i];
              if (shiftToUpdate) {
                setAllShifts(prev => {
                  const newShifts = [...prev];
                  const shiftIndex = newShifts.findIndex(s => s && s.id === shiftToUpdate.id);
                  if (shiftIndex !== -1) {
                    newShifts[shiftIndex] = shiftToUpdate;
                  }
                  return newShifts;
                });
              }
              i++;
            } else {
              clearInterval(interval);
              setAllShifts(finalRoster);
              fetchMetrics(finalRoster);
              setIsGenerating(false);
            }
          }, 50);
        } else {
          if (result.roster) {
            setAllShifts(result.roster);
            fetchMetrics(result.roster);
          }
          if (result.history) {
            setGaHistory(result.history);
          }
          if (result.programHistory) {
            setGpProgramHistory(result.programHistory);
            setSelectedGeneration(result.programHistory.length - 1);
          }
          setIsGenerating(false);
        }
      })
      .catch(err => {
        console.error('Error generating roster:', err);
        setIsGenerating(false);
      });
  };

  const handleSaveRoster = () => {
    fetch('http://localhost:4000/api/roster', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(allShifts),
    })
    .then(res => res.json())
    .then((updatedRoster: Shift[]) => {
      setAllShifts(updatedRoster);
      alert('Roster saved!');
    })
    .catch(err => console.error('Error saving roster:', err));
  };

  const handleClearRoster = () => {
    const clearedRoster = allShifts.map(shift => ({ ...shift, staffMemberId: undefined }));
    setAllShifts(clearedRoster);
    fetchMetrics(clearedRoster);
  };

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (over && active.id !== over.id) {
      const staffId = active.id as string;
      const shiftId = over.id as string;

      const newShifts = [...allShifts];
      const shiftIndex = newShifts.findIndex(s => s.id === shiftId);
      if (shiftIndex !== -1) {
        newShifts[shiftIndex].staffMemberId = staffId;
      }
      setAllShifts(newShifts);
      fetchMetrics(newShifts);
    }
  };

  const getShiftsForSelectedDate = (date: Date) => {
    const startOfDay = new Date(date);
    startOfDay.setHours(0, 0, 0, 0);
    const endOfDay = new Date(date);
    endOfDay.setHours(23, 59, 59, 999);

    return allShifts.filter(shift => {
      const shiftStartTime = new Date(shift.startTime);
      return shiftStartTime >= startOfDay && shiftStartTime <= endOfDay;
    });
  };

  const shiftsForSelectedDate = getShiftsForSelectedDate(selectedDate);
  const selectedStaffIdsForDay = shiftsForSelectedDate.map(s => s.staffMemberId).filter(Boolean) as string[];

  const selectedStaffMember = selectedStaffMemberId ? allStaff.find(s => s.id === selectedStaffMemberId) : null;

  const handleShiftClick = (shift: Shift) => {
    setSelectedShiftForDetails(shift);
    setIsShiftModalOpen(true);
  };

  const handleCloseShiftModal = () => {
    setIsShiftModalOpen(false);
    setSelectedShiftForDetails(null);
  };

  if (loading) return <div className="hero-select__message">Loading...</div>;
  if (error) return <div className="hero-select__message hero-select__message--error">Error: {error}</div>;

  return (
    <DndContext onDragEnd={handleDragEnd}>
      <div className="hero-select">
        <div className="hero-select__main-content">
          <div className="hero-select__title-container">
            <h2 className="hero-select__team-title">
              {rosterView === 'weekly' && 'Full Week Roster'}
              {rosterView === 'daily' && `Roster for ${selectedDate.toDateString()}`}
              {rosterView === 'staff' && selectedStaffMember && `Roster for ${selectedStaffMember.name}`}
              {rosterView === 'staff' && !selectedStaffMember && 'Select a Staff Member'}
            </h2>
            <div className="hero-select__view-controls">
              <button 
                onClick={() => setRosterView('weekly')} 
                className={rosterView === 'weekly' ? 'active' : ''}
                disabled={isGenerating}
              >
                Weekly
              </button>
              <button 
                onClick={() => setRosterView('daily')} 
                className={rosterView === 'daily' ? 'active' : ''}
                disabled={isGenerating}
              >
                Daily
              </button>
              <button 
                onClick={() => setRosterView('staff')} 
                className={rosterView === 'staff' ? 'active' : ''}
                disabled={isGenerating}
              >
                Staff
              </button>
            </div>
            {rosterView === 'daily' && (
              <div className="hero-select__date-nav">
                <button onClick={() => setSelectedDate(prev => new Date(prev.getTime() - 24 * 60 * 60 * 1000))} disabled={isGenerating}>Prev Day</button>
                <span>{selectedDate.toDateString()}</span>
                <button onClick={() => setSelectedDate(prev => new Date(prev.getTime() + 24 * 60 * 60 * 1000))} disabled={isGenerating}>Next Day</button>
              </div>
            )}
            {rosterView === 'staff' && (
              <div className="hero-select__staff-select">
                <select 
                  value={selectedStaffMemberId || ''} 
                  onChange={e => setSelectedStaffMemberId(e.target.value)}
                  disabled={isGenerating}
                >
                  <option value="">Select Staff Member</option>
                  {allStaff.map(s => (
                    <option key={s.id} value={s.id}>{s.name}</option>
                  ))}
                </select>
              </div>
            )}
            <div className="hero-select__generation-controls">
              <select 
                value={selectedAlgorithm} 
                onChange={e => setSelectedAlgorithm(e.target.value as Algorithm)}
                disabled={isGenerating}
              >
                <option value="greedy">Greedy Algorithm</option>
                <option value="ga">Genetic Algorithm</option>
                <option value="gp">Genetic Programming</option>
                <option value="cp">Constraint Programming</option>
                <option value="aco">Ant Colony Optimization</option>
                <option value="hybrid_ga_cp">Hybrid GA+CP</option>
                <option value="neural_network">Neural Network</option>
                <option value="pso">Particle Swarm Optimization</option>
              </select>
              {(selectedAlgorithm === 'ga' || selectedAlgorithm === 'hybrid_ga_cp') && (
                <div className="generation-input">
                  <label htmlFor="numGenerations">Generations:</label>
                  <input
                    type="number"
                    id="numGenerations"
                    value={numGenerations}
                    onChange={e => setNumGenerations(Number(e.target.value))}
                    min="1"
                    disabled={isGenerating}
                  />
                </div>
              )}
              <button className="hero-select__generate-button" onClick={handleGenerateRoster} disabled={isGenerating}>
                Run Generation
              </button>
            </div>
            <button className="hero-select__save-button" onClick={handleSaveRoster} disabled={isGenerating}>
              Save Roster
            </button>
            <button className="hero-select__clear-button" onClick={handleClearRoster} disabled={isGenerating}>
              Clear Roster
            </button>
          </div>
          {isGenerating && <div className="hero-select__message">Generating...</div>}
          <div style={{ display: isGenerating ? 'none' : 'block' }}>
            {rosterView === 'weekly' && <WeeklyRoster shifts={allShifts} staff={allStaff} selectedDate={selectedDate} onShiftClick={handleShiftClick} conflicts={rosterConflicts} />}
            {rosterView === 'daily' && <DailyRoster shifts={shiftsForSelectedDate} staff={allStaff} selectedDate={selectedDate} onShiftClick={handleShiftClick} conflicts={rosterConflicts} />}
            {rosterView === 'staff' && selectedStaffMember && <StaffRoster staffMember={selectedStaffMember} shifts={allShifts} />}
            {rosterView === 'staff' && !selectedStaffMember && <p>Please select a staff member to view their roster.</p>}
          </div>
        </div>
        <div className="hero-select__sidebar">
          {rosterView === 'daily' && !isGenerating && (
            <AvailableStaff
              allStaff={allStaff}
              selectedStaffIds={selectedStaffIdsForDay}
            />
          )}
          <RosterMetricsDisplay metrics={metrics} />
          <FitnessChart history={gaHistory} />
          {gpProgramHistory.length > 0 && (
            <div className="gp-visualization">
              <label>
                Generation: {selectedGeneration}
                <input 
                  type="range" 
                  min="0" 
                  max={gpProgramHistory.length - 1} 
                  value={selectedGeneration}
                  onChange={e => setSelectedGeneration(Number(e.target.value))}
                />
              </label>
              <ProgramTree program={gpProgramHistory[selectedGeneration]} />
            </div>
          )}
        </div>
      </div>
      <ShiftDetailsModal 
        shift={selectedShiftForDetails} 
        staff={allStaff} 
        onClose={handleCloseShiftModal} 
      />
    </DndContext>
  );
};

export default HeroSelect;

