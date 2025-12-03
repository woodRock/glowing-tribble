export type Role = "bar" | "restaurant" | "maitre'd" | "duty manager";

export interface StaffMember {
  id: string;
  name: string;
  roles: Role[];
}
