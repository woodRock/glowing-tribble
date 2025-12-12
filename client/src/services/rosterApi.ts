
import type { StaffMember, Shift, RosterMetrics } from '../types';

// --- Interfaces for Python API Payload ---

interface PythonStaffData {
  id: string;
  skills: string[];
  availability_slots: { start_time: string; end_time: string }[];
  contract_details?: {
    min_rest_time?: number;
    max_workload_minutes?: number;
    // Add other contract fields if we have UI for them
  }[];
  preferences?: {
    desired_hours?: number;
  };
}

interface PythonShiftData {
  id: string;
  required_skills: string[];
  start_time: string;
  end_time: string;
  role: string;
  min_staff_for_role?: number;
}

interface PythonRequestPayload {
  staff_data: PythonStaffData[];
  shift_data: PythonShiftData[];
  generations?: number;
  population_size?: number;
  cxpb?: number;
  mutpb?: number;
  requests?: any; // Add specific types if needed
}

interface PythonApiResponse {
  status: string;
  roster_details: {
    roster: Record<string, string | null>; // shift_id -> staff_id
    fitness: number;
    metrics: any;
    message: string;
  };
  best_heuristic_tree: Record<string, string>;
  error?: string;
}

// --- API Service Function ---

export const generateRosterWithPythonGA = async (
  staffList: StaffMember[],
  shiftList: Shift[],
  config: {
    generations?: number;
    populationSize?: number;
    cxpb?: number;
    mutpb?: number;
  } = {}
): Promise<{ assignments: Record<string, string | null>; metrics: RosterMetrics }> => {
  
  // 1. Transform Staff Data
  const pythonStaffData: PythonStaffData[] = staffList.map(staff => ({
    id: staff.id,
    skills: staff.roles, // Map Role[] -> string[]
    availability_slots: staff.preferences.availability.map(slot => ({
      start_time: slot.startTime,
      end_time: slot.endTime
    })),
    // Default contract details for now since UI might not expose them fully yet
    contract_details: [{
      min_rest_time: 60, // Default 1 hour rest
      max_workload_minutes: 480 // Default 8 hours max per day (simplified)
    }],
    preferences: {
      desired_hours: staff.preferences.desiredHours
    }
  }));

  // 2. Transform Shift Data
  const pythonShiftData: PythonShiftData[] = shiftList.map(shift => ({
    id: shift.id,
    required_skills: [shift.role], // Assuming single role per shift for now
    start_time: shift.startTime,
    end_time: shift.endTime,
    role: shift.role
  }));

  // 3. Construct Payload
  const payload: PythonRequestPayload = {
    staff_data: pythonStaffData,
    shift_data: pythonShiftData,
    generations: config.generations || 50,
    population_size: config.populationSize || 100,
    cxpb: config.cxpb || 0.6,
    mutpb: config.mutpb || 0.3
  };

  console.log(`Sending roster generation request: ${pythonStaffData.length} staff, ${pythonShiftData.length} shifts.`);

  // 4. Call API
  try {
    // Assuming backend is proxy-ed or CORS allows localhost:5000
    // If running separately, you might need full URL: http://127.0.0.1:5000/generate_roster
    const response = await fetch('/api/generate_roster', { 
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API Error ${response.status}: ${errorText}`);
    }

    const data: PythonApiResponse = await response.json();

    if (data.error) {
      throw new Error(`Algorithm Error: ${data.error}`);
    }

    const serverMetrics = data.roster_details.metrics || {};
    const metrics: RosterMetrics = {
        totalPenalty: serverMetrics.total_penalty || data.roster_details.fitness || 0,
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
    };

    return {
      assignments: data.roster_details.roster,
      metrics: metrics
    };

  } catch (error) {
    console.error("Failed to generate roster:", error);
    throw error;
  }
};
