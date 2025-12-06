import React, { useEffect, useState, useCallback } from 'react';
import type { StaffMember, Shift, RosterMetrics, ProgramNode } from '../types.ts';
import DailyRoster from '../components/DailyRoster';
import AvailableStaff from '../components/AvailableStaff';
import WeeklyRoster from '../components/WeeklyRoster';
import RosterMetricsDisplay from '../components/RosterMetrics';
import FitnessChart from '../components/FitnessChart';
import ProgramTree from '../components/ProgramTree';
import { DndContext } from '@dnd-kit/core';
import type { DragEndEvent } from '@dnd-kit/core';
import './HeroSelect.css';

const weekDays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
type Algorithm = 'greedy' | 'ga' | 'gp';

const HeroSelect: React.FC = () => {
  const [allStaff, setAllStaff] = useState<StaffMember[]>([]);
  const [allShifts, setAllShifts] = useState<Shift[]>([]);
  const [selectedDay, setSelectedDay] = useState(0);
  const [loading, setLoading] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<RosterMetrics | null>(null);
  const [gaHistory, setGaHistory] = useState<number[]>([]);
  const [gpProgramHistory, setGpProgramHistory] = useState<ProgramNode[]>([]);
  const [selectedGeneration, setSelectedGeneration] = useState(0);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<Algorithm>('greedy');

  const fetchMetrics = useCallback((roster: Shift[]) => {
    fetch('http://localhost:4000/api/roster/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(roster),
    })
    .then(res => res.json())
    .then(setMetrics)
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

    fetch(`http://localhost:4000/api/roster/generate/${selectedAlgorithm}`, { method: 'POST' })
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

  const getShiftsForDay = (dayIndex: number) => {
    if (dayIndex > 6) return allShifts;
    const day = new Date(new Date('2025-12-15T00:00:00').getTime() + dayIndex * 24 * 60 * 60 * 1000);
    return allShifts.filter(shift => new Date(shift.startTime).getDate() === day.getDate());
  };

  const shiftsForDay = getShiftsForDay(selectedDay);
  const selectedStaffIdsForDay = shiftsForDay.map(s => s.staffMemberId).filter(Boolean) as string[];

  if (loading) return <div className="hero-select__message">Loading...</div>;
  if (error) return <div className="hero-select__message hero-select__message--error">Error: {error}</div>;

  return (
    <DndContext onDragEnd={handleDragEnd}>
      <div className="hero-select">
        <div className="hero-select__main-content">
          <div className="hero-select__title-container">
            <h2 className="hero-select__team-title">
              {selectedDay > 6 ? 'Full Week Roster' : `${weekDays[selectedDay]}'s Roster`}
            </h2>
            <div className="hero-select__nav-buttons">
              <button onClick={() => setSelectedDay(prev => Math.max(0, prev - 1))} disabled={isGenerating}>Prev Day</button>
              <button onClick={() => setSelectedDay(prev => Math.min(6, prev + 1))} disabled={isGenerating}>Next Day</button>
              <button onClick={() => setSelectedDay(7)} disabled={isGenerating}>This Week</button>
            </div>
            <div className="hero-select__generation-controls">
              <select 
                value={selectedAlgorithm} 
                onChange={e => setSelectedAlgorithm(e.target.value as Algorithm)}
                disabled={isGenerating}
              >
                <option value="greedy">Greedy Algorithm</option>
                <option value="ga">Genetic Algorithm</option>
                <option value="gp">Genetic Programming</option>
              </select>
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
            {selectedDay > 6 ? (
              <WeeklyRoster shifts={allShifts} staff={allStaff} />
            ) : (
              <DailyRoster shifts={shiftsForDay} staff={allStaff} />
            )}
          </div>
        </div>
        <div className="hero-select__sidebar">
          {selectedDay <= 6 && !isGenerating && (
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
    </DndContext>
  );
};

export default HeroSelect;

