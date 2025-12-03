export type Role = "bar" | "restaurant" | "maitre'd" | "duty manager";

export interface StaffMember {
  id: string;
  name: string;
  roles: Role[];
}

export interface Shift {
  id: string;
  role: Role;
  startTime: string; // Use string for dates from JSON
  endTime: string;   // Use string for dates from JSON
  staffMemberId?: string;
}