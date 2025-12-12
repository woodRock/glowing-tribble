import React, { useEffect, useState, useCallback } from 'react';
import type { StaffMember, Shift, RosterMetrics, ProgramNode } from '../types';
import DailyRoster from '../components/DailyRoster';
import AvailableStaff from '../components/AvailableStaff';
import WeeklyRoster from '../components/WeeklyRoster';
import StaffRoster from '../components/StaffRoster';
import RosterMetricsDisplay from '../components/RosterMetrics';
import FitnessChart from '../components/FitnessChart';
import ProgramTree from '../components/ProgramTree';
import ShiftDetailsModal from '../components/ShiftDetailsModal'; 
import BenchmarkAnalysis from '../components/BenchmarkAnalysis'; // Import Analysis
import { getBenchmarks, getBenchmarkData } from '../services/rosterApi';
import type { PythonStaffData, PythonShiftData } from '../services/rosterApi'; // Import benchmark services
import { staff as mockStaff, shifts as mockShifts } from '../mockData';
import { DndContext } from '@dnd-kit/core';
import type { DragEndEvent } from '@dnd-kit/core';
import './HeroSelect.css';

const weekDays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
type Algorithm = 'python_ga' | 'python_gp' | 'python_pso' | 'python_aco' | 'python_de' | 'python_es'; 
type RosterView = 'weekly' | 'daily' | 'staff';

const HeroSelect: React.FC = () => {
  const [allStaff, setAllStaff] = useState<StaffMember[]>([]);
  const [allShifts, setAllShifts] = useState<Shift[]>([]);
  const [selectedDate, setSelectedDate] = useState<Date>(new Date('2025-01-01T00:00:00')); // Default to benchmark date
  const [rosterView, setRosterView] = useState<RosterView>('weekly');
  const [selectedStaffMemberId, setSelectedStaffMemberId] = useState<string | null>(null);
  const [selectedShiftForDetails, setSelectedShiftForDetails] = useState<Shift | null>(null);
  const [isShiftModalOpen, setIsShiftModalOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<RosterMetrics | null>(null);
  const [rosterConflicts, setRosterConflicts] = useState<any[]>([]); 
  const [gaHistory, setGaHistory] = useState<number[]>([]);
  const [gpProgramHistory, setGpProgramHistory] = useState<ProgramNode[]>([]);
  const [selectedGeneration, setSelectedGeneration] = useState(0);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<Algorithm>('python_ga');
  const [numGenerations, setNumGenerations] = useState<number>(50);
  
  // Benchmark State
  const [benchmarks, setBenchmarks] = useState<string[]>([]);
  const [selectedBenchmark, setSelectedBenchmark] = useState<string>('');
  const [benchmarkRequests, setBenchmarkRequests] = useState<any>(null); // New state

  const fetchMetrics = useCallback((roster: Shift[]) => {
    // Mock metrics or fetch from backend if available
  }, []);

  useEffect(() => {
    // Fetch benchmarks on load
    getBenchmarks().then(files => {
      setBenchmarks(files);
      if (files.length > 0) {
          // Optional: Load first benchmark by default? Or stick to mock data until selected?
          // Let's load the first one if available, else mock
          // handleBenchmarkChange(files[0]); 
      }
    }).catch(err => console.error("Failed to load benchmarks", err));

    setAllStaff(mockStaff);
    setAllShifts(mockShifts);
    setLoading(false);
  }, []);

  const handleBenchmarkChange = async (filename: string) => {
      setSelectedBenchmark(filename);
      if (!filename) return;

      setLoading(true);
      try {
          const data = await getBenchmarkData(filename);
          setBenchmarkRequests(data.requests); // Set requests state
          
          // Map Python/JSON data to Frontend Types
          const mappedStaff: StaffMember[] = data.staff_data.map((s: PythonStaffData) => ({
              id: s.id,
              name: s.id, // Benchmark data doesn't have names, use ID
              roles: s.skills,
              avatar: `https://api.dicebear.com/7.x/avataaars/svg?seed=${s.id}`, // Generate avatar
              preferences: {
                  desiredHours: s.preferences?.desired_hours || 40,
                  availability: s.availability_slots.map(slot => ({
                      day: new Date(slot.start_time).toLocaleDateString('en-US', { weekday: 'long' }),
                      startTime: slot.start_time,
                      endTime: slot.end_time
                  }))
              }
          }));

          const mappedShifts: Shift[] = data.shift_data.map((s: PythonShiftData) => ({
              id: s.id,
              role: s.role,
              startTime: s.start_time,
              endTime: s.end_time,
              staffMemberId: undefined
          }));

          setAllStaff(mappedStaff);
          setAllShifts(mappedShifts);
          
          // Update selected date to the start of the benchmark problem
          if (mappedShifts.length > 0) {
              const firstShiftDate = new Date(mappedShifts[0].startTime);
              setSelectedDate(firstShiftDate);
          }

          setMetrics(null);
          setGaHistory([]);
      } catch (err) {
          console.error("Error loading benchmark:", err);
          setError("Failed to load benchmark");
      } finally {
          setLoading(false);
      }
  };

  const handleGenerateRoster = async () => {
    setIsGenerating(true);
    setMetrics(null);
    setGaHistory([]);
    setGpProgramHistory([]);

    const emptyRoster = allShifts.map(shift => ({ ...shift, staffMemberId: undefined }));
    setAllShifts(emptyRoster);

    // Map Staff and Shift data to Python API format
    // This mapping logic was previously in the Node bridge or service, we need it here now.
    const pythonStaffData = allStaff.map(staff => ({
      id: staff.id,
      skills: staff.roles,
      availability_slots: staff.preferences.availability.map(slot => ({
        start_time: slot.startTime,
        end_time: slot.endTime
      })),
      contract_details: [{
        min_rest_time: 60,
        max_workload_minutes: 480
      }],
      preferences: {
        desired_hours: staff.preferences.desiredHours
      }
    }));

    const pythonShiftData = emptyRoster.map(shift => ({
      id: shift.id,
      required_skills: [shift.role],
      start_time: shift.startTime,
      end_time: shift.endTime,
      role: shift.role
    }));

    const algoName = selectedAlgorithm.replace('python_', ''); // 'ga', 'gp', ...

    try {
        const response = await fetch('http://localhost:5001/generate_roster', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                algorithm: algoName,
                generations: numGenerations,
                population_size: 100,
                staff_data: pythonStaffData,
                shift_data: pythonShiftData,
                // requests: ... (If we had request data in the UI state, we'd pass it here)
            }),
        });

        const result = await response.json();

        if (result.error) {
            throw new Error(result.error);
        }

        if (result.roster_details && result.roster_details.roster) {
            const assignmentMap = result.roster_details.roster;
            const updatedRoster = emptyRoster.map(shift => ({
                ...shift,
                staffMemberId: assignmentMap[shift.id] || undefined
            }));
            
            setAllShifts(updatedRoster);
            
            // Map metrics
            const serverMetrics = result.roster_details.metrics || {};
            setMetrics({
                totalPenalty: serverMetrics.total_penalty || result.roster_details.fitness || 0,
                details: {
                    uncoveredShifts: serverMetrics.uncovered_shifts_count || 0,
                    skillMismatches: serverMetrics.skill_mismatches_count || 0,
                    availabilityMismatches: serverMetrics.availability_mismatches_count || 0,
                    overlaps: serverMetrics.overlap_penalties_count || 0,
                    minRestViolations: serverMetrics.min_rest_violations_count || 0,
                    maxWorkloadViolations: serverMetrics.max_workload_violations_count || 0,
                    minWorkloadViolations: serverMetrics.min_workload_violations_count || 0,
                    maxConsecutiveShiftsViolations: serverMetrics.max_consecutive_shift_violations_count || 0,
                    minConsecutiveShiftsViolations: serverMetrics.min_consecutive_shift_violations_count || 0,
                    minConsecutiveDaysOffViolations: serverMetrics.min_consecutive_days_off_violations_count || 0,
                    maxWeekendPatternViolations: serverMetrics.max_weekend_pattern_violations_count || 0,
                    shiftOffRequestPenalties: serverMetrics.shift_off_request_penalties || 0,
                    shiftOnRequestPenalties: serverMetrics.shift_on_request_penalties || 0,
                    coverRequirementPenalties: serverMetrics.cover_requirement_penalties || 0,
                    minStaffForRoleViolations: serverMetrics.min_staff_for_role_violations_count || 0,
                    desiredHoursPenalties: serverMetrics.desired_hours_penalties || 0
                },
                conflicts: [] 
            });
        }
        
    } catch (e: any) {
        console.error('Error generating roster:', e);
        setError(e.message);
    } finally {
        setIsGenerating(false);
    }
  };

  const handleSaveRoster = () => {
    console.log("Save roster not implemented for Python API yet.");
    alert("Save functionality is currently disabled.");
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
              {rosterView === 'weekly' && `Full Week Roster (${allShifts.length} Shifts)`}
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
            
            {/* Benchmark Selector */}
            <div className="hero-select__benchmark-select" style={{ marginLeft: '1rem' }}>
                <select 
                    value={selectedBenchmark}
                    onChange={(e) => handleBenchmarkChange(e.target.value)}
                    disabled={isGenerating}
                >
                    <option value="">-- Load Benchmark --</option>
                    {benchmarks.map(b => (
                        <option key={b} value={b}>{b.replace('.json', '')}</option>
                    ))}
                </select>
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
                <optgroup label="Python Solvers (High Performance)">
                    <option value="python_ga">Genetic Algorithm (Best)</option>
                    <option value="python_aco">Ant Colony (Fastest)</option>
                    <option value="python_gp">Genetic Programming</option>
                    <option value="python_pso">PSO</option>
                    <option value="python_de">Differential Evolution</option>
                    <option value="python_es">Evolution Strategy</option>
                </optgroup>
              </select>
              {(selectedAlgorithm.includes('ga') || selectedAlgorithm.includes('gp') || selectedAlgorithm.includes('python')) && (
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
          {selectedBenchmark && (
            <BenchmarkAnalysis 
              staff={allStaff} 
              shifts={allShifts} 
              requests={benchmarkRequests} 
            />
          )}
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

