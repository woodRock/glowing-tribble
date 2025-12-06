export type Role = "bar" | "restaurant" | "maitre'd" | "duty manager";

export interface StaffPreferences {
  availability: {
    startTime: string; // Changed to string for consistency with Shift
    endTime: string;   // Changed to string for consistency with Shift
  }[];
  desiredHours: number;
  prefersConsecutiveDaysOff: boolean;
}

export interface StaffMember {
  id: string;
  name: string;
  roles: Role[];
  preferences: StaffPreferences; // Added
  avatar: string;
}

export interface Shift {
  id: string;
  role: Role;
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
  penalties: {
    desiredHours: number;
    consecutiveDays: number;
    clopen: number;
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