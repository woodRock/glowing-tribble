export type Role = "bar" | "restaurant" | "maitre'd" | "duty manager";

export interface StaffMember {
  id: string;
  name: string;
  roles: Role[];
  avatar: string;
}

export interface Shift {
  id: string;
  role: Role;
  startTime: string; // Use string for dates from JSON
  endTime: string;   // Use string for dates from JSON
  staffMemberId?: string;
}

export interface RosterMetrics {
  totalPenalty: number;
  penalties: {
    desiredHours: number;
    consecutiveDays: number;
    clopen: number;
  };
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
