
import type { StaffMember, Shift, RosterMetrics } from '../types';

// --- Interfaces for Python API Payload ---

export interface PythonStaffData {
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

export interface PythonShiftData {
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

// ... existing interfaces ...

// --- Benchmark Service Functions ---

export const getBenchmarks = async (): Promise<string[]> => {
  const response = await fetch('http://localhost:5001/api/benchmarks');
  if (!response.ok) {
    throw new Error('Failed to fetch benchmarks');
  }
  return response.json();
};

export const getBenchmarkData = async (filename: string): Promise<{ staff_data: PythonStaffData[], shift_data: PythonShiftData[], requests: any }> => {
  const response = await fetch(`http://localhost:5001/api/benchmarks/${filename}`);
  if (!response.ok) {
    throw new Error('Failed to fetch benchmark data');
  }
  return response.json();
};


