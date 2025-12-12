export type Role = "bar" | "restaurant" | "maitre'd" | "duty manager";

export interface StaffPreferences {
  availability: {
    startTime: string; // Changed to string for consistency with Shift
    endTime: string;   // Changed to string for consistency with Shift
  }[];
  desiredHours: number;
  prefersConsecutiveDaysOff: boolean;
}

export interface ContractDetails {
  minRestTime?: number;
  maxWorkloadMinutes?: number;
  maxSeqShifts?: { value: number; label?: string };
  minSeqDaysOff?: { value: number; label?: string };
  maxWeekendPatterns?: number;
}

export interface StaffMember {
  id: string;
  name: string;
  roles: string[];
  preferences: StaffPreferences; // Added
  contractDetails?: ContractDetails[];
  avatar: string;
}

export interface Shift {
  id: string;
  role: string;
  startTime: string;
  endTime: string;
  staffMemberId?: string;
}

export interface Conflict {
  type: 'double_booking' | 'unassigned_shift' | 'role_mismatch' | 'availability_violation' | 'insufficient_rest' | 'exceeds_max_hours' | 'exceeds_consecutive_days';
  shiftId?: string;
  staffMemberId?: string;
  message: string;
}

export interface RosterMetrics {
  totalPenalty: number;
  details: {
    uncoveredShifts: number;
    skillMismatches: number;
    availabilityMismatches: number;
    overlaps: number;
    minRestViolations: number;
    maxWorkloadViolations: number;
    minWorkloadViolations: number;
    maxConsecutiveShiftsViolations: number;
    minConsecutiveShiftsViolations: number;
    minConsecutiveDaysOffViolations: number;
    maxWeekendPatternViolations: number;
    shiftOffRequestPenalties: number;
    shiftOnRequestPenalties: number;
    coverRequirementPenalties: number;
    minStaffForRoleViolations: number;
    desiredHoursPenalties: number;
  };
  conflicts: Conflict[];
}

export interface RosterGenerationResult {
  roster: Shift[];
  history: number[];
  programHistory: any[];
}

export interface ProgramNode {
  type: 'function' | 'terminal';
  name: string;
  children: ProgramNode[];
}