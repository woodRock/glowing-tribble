import type { StaffMember, Shift } from './types';

// Mock avatar generator
const generateAvatarSvg = (id: string, name: string): string => {
  return "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==";
};

export const staff: StaffMember[] = [
  {
    id: "S1",
    name: "Alice (Bar Lead)",
    roles: ["bar", "restaurant"],
    avatar: generateAvatarSvg("S1", "Alice"),
    preferences: {
      desiredHours: 35,
      prefersConsecutiveDaysOff: true,
      availability: [
        { startTime: "2025-12-08T08:00:00", endTime: "2025-12-08T23:00:00" }, // Mon
        { startTime: "2025-12-09T08:00:00", endTime: "2025-12-09T23:00:00" }, // Tue
        { startTime: "2025-12-10T08:00:00", endTime: "2025-12-10T23:00:00" }, // Wed
        { startTime: "2025-12-11T08:00:00", endTime: "2025-12-11T23:00:00" }, // Thu
        { startTime: "2025-12-12T08:00:00", endTime: "2025-12-12T23:00:00" }, // Fri
        { startTime: "2025-12-13T10:00:00", endTime: "2025-12-13T23:00:00" }, // Sat
        { startTime: "2025-12-14T10:00:00", endTime: "2025-12-14T23:00:00" }  // Sun
      ]
    },
    contractDetails: [{
      minRestTime: 600, // 10 hours rest between shifts
      maxWorkloadMinutes: 2400, // 40 hours
      maxSeqShifts: { value: 5 },
      maxWeekendPatterns: 2
    }]
  },
  {
    id: "S2",
    name: "Bob (Manager)",
    roles: ["duty manager", "bar"],
    avatar: generateAvatarSvg("S2", "Bob"),
    preferences: {
      desiredHours: 40,
      prefersConsecutiveDaysOff: false,
      availability: [
        { startTime: "2025-12-08T07:00:00", endTime: "2025-12-08T17:00:00" }, // Mon (Morning pref)
        { startTime: "2025-12-09T07:00:00", endTime: "2025-12-09T17:00:00" }, // Tue
        { startTime: "2025-12-10T07:00:00", endTime: "2025-12-10T17:00:00" }, // Wed
        { startTime: "2025-12-11T07:00:00", endTime: "2025-12-11T17:00:00" }, // Thu
        { startTime: "2025-12-12T07:00:00", endTime: "2025-12-12T23:00:00" }, // Fri (Open)
        { startTime: "2025-12-13T07:00:00", endTime: "2025-12-13T23:00:00" }, // Sat (Open)
        { startTime: "2025-12-14T07:00:00", endTime: "2025-12-14T17:00:00" }  // Sun
      ]
    },
    contractDetails: [{
      minRestTime: 600,
      maxWorkloadMinutes: 2400,
      minSeqDaysOff: { value: 1 }
    }]
  },
  {
    id: "S3",
    name: "Charlie (Maitre'd)",
    roles: ["maitre'd", "restaurant"],
    avatar: generateAvatarSvg("S3", "Charlie"),
    preferences: {
      desiredHours: 20,
      prefersConsecutiveDaysOff: true,
      availability: [
        { startTime: "2025-12-09T16:00:00", endTime: "2025-12-09T23:00:00" }, // Tue Evening
        { startTime: "2025-12-10T16:00:00", endTime: "2025-12-10T23:00:00" }, // Wed Evening
        { startTime: "2025-12-11T16:00:00", endTime: "2025-12-11T23:00:00" }, // Thu Evening
        { startTime: "2025-12-12T16:00:00", endTime: "2025-12-12T23:00:00" }, // Fri Evening
        { startTime: "2025-12-13T12:00:00", endTime: "2025-12-13T23:00:00" }, // Sat All day
        { startTime: "2025-12-14T12:00:00", endTime: "2025-12-14T23:00:00" }  // Sun All day
      ]
    },
    contractDetails: [{
      minRestTime: 600,
      maxWorkloadMinutes: 1500, // 25 hours max
      maxSeqShifts: { value: 4 }
    }]
  },
  {
    id: "S4",
    name: "David (Student)",
    roles: ["restaurant", "bar"],
    avatar: generateAvatarSvg("S4", "David"),
    preferences: {
      desiredHours: 15,
      prefersConsecutiveDaysOff: false,
      availability: [
        { startTime: "2025-12-12T17:00:00", endTime: "2025-12-12T23:00:00" }, // Fri Evening only
        { startTime: "2025-12-13T08:00:00", endTime: "2025-12-13T23:00:00" }, // Sat
        { startTime: "2025-12-14T08:00:00", endTime: "2025-12-14T23:00:00" }  // Sun
      ]
    },
    contractDetails: [{
      minRestTime: 480,
      maxWorkloadMinutes: 1200 // 20 hours max
    }]
  },
  {
    id: "S5",
    name: "Eve (Full Time)",
    roles: ["bar", "restaurant"],
    avatar: generateAvatarSvg("S5", "Eve"),
    preferences: {
      desiredHours: 38,
      prefersConsecutiveDaysOff: true,
      availability: [
        { startTime: "2025-12-08T08:00:00", endTime: "2025-12-08T23:00:00" },
        { startTime: "2025-12-09T08:00:00", endTime: "2025-12-09T23:00:00" },
        { startTime: "2025-12-10T08:00:00", endTime: "2025-12-10T23:00:00" },
        { startTime: "2025-12-11T08:00:00", endTime: "2025-12-11T23:00:00" },
        { startTime: "2025-12-12T08:00:00", endTime: "2025-12-12T23:00:00" },
        { startTime: "2025-12-13T08:00:00", endTime: "2025-12-13T16:00:00" } // Sat morning only
      ]
    },
    contractDetails: [{
      minRestTime: 600,
      maxWorkloadMinutes: 2400,
      maxSeqShifts: { value: 5 }
    }]
  },
  {
    id: "S6",
    name: "Frank (Manager)",
    roles: ["duty manager", "maitre'd"],
    avatar: generateAvatarSvg("S6", "Frank"),
    preferences: {
      desiredHours: 30,
      prefersConsecutiveDaysOff: false,
      availability: [
        { startTime: "2025-12-08T16:00:00", endTime: "2025-12-08T23:00:00" }, // Mon Evening
        { startTime: "2025-12-09T16:00:00", endTime: "2025-12-09T23:00:00" }, // Tue Evening
        { startTime: "2025-12-10T16:00:00", endTime: "2025-12-10T23:00:00" }, // Wed Evening
        { startTime: "2025-12-11T16:00:00", endTime: "2025-12-11T23:00:00" }, // Thu Evening
        { startTime: "2025-12-12T16:00:00", endTime: "2025-12-12T23:00:00" }, // Fri Evening
        { startTime: "2025-12-13T14:00:00", endTime: "2025-12-13T23:00:00" }, // Sat Afternoon/Eve
        { startTime: "2025-12-14T14:00:00", endTime: "2025-12-14T23:00:00" }  // Sun Afternoon/Eve
      ]
    },
    contractDetails: [{
      minRestTime: 600,
      maxWorkloadMinutes: 2000
    }]
  }
];

// Generate shifts for 7 days (Dec 8 - Dec 14)
const days = [
  "2025-12-08", "2025-12-09", "2025-12-10", "2025-12-11", "2025-12-12", "2025-12-13", "2025-12-14"
];

const shiftTemplates = [
  { role: "duty manager", start: "08:00:00", end: "16:00:00" },
  { role: "bar", start: "10:00:00", end: "15:00:00" },
  { role: "restaurant", start: "11:00:00", end: "15:00:00" },
  { role: "duty manager", start: "16:00:00", end: "23:00:00" },
  { role: "maitre'd", start: "17:00:00", end: "22:00:00" },
  { role: "bar", start: "17:00:00", end: "23:00:00" },
  { role: "restaurant", start: "18:00:00", end: "22:00:00" }
];

const generatedShifts: Shift[] = [];
let shiftIdCounter = 1;

days.forEach(date => {
  shiftTemplates.forEach(template => {
    // Fri/Sat have extra bar staff
    const isWeekend = date.endsWith("12") || date.endsWith("13"); 
    const copies = (isWeekend && template.role === "bar") ? 2 : 1;

    for (let i = 0; i < copies; i++) {
      generatedShifts.push({
        id: `SH${shiftIdCounter++}`,
        role: template.role as any,
        startTime: `${date}T${template.start}`,
        endTime: `${date}T${template.end}`
      });
    }
  });
});

export const shifts = generatedShifts;