import json
import random
from datetime import datetime, timedelta
import os
import sys
from tqdm import tqdm

# --- Configuration ---
NUM_STAFF_RANGE = (5, 10) # Min and max staff members per example (reduced for tractability)
NUM_SHIFTS_PER_DAY_RANGE = (2, 5) # Min and max shifts per day per example (reduced for tractability)
NUM_DAYS_RANGE = (3, 7) # Min and max days per example (reduced for tractability)
OUTPUT_DIR = 'data'
NUM_EXAMPLES = 100_000 # Number of examples to generate

# --- Global Data (for consistent role encoding) ---
ALL_POSSIBLE_ROLES = ["bar", "restaurant", "maitre'd", "duty manager", "chef", "host"]

# --- Helper Functions ---
def generate_staff_member(staff_id):
    name = f"Staff{staff_id}"
    
    # Assign 3 to 5 random roles (increased diversity)
    num_roles = random.randint(3, 5)
    roles = random.sample(ALL_POSSIBLE_ROLES, num_roles)
    
    desired_hours = random.randint(20, 40)
    prefers_consecutive_days_off = random.choice([True, False])

    availability = []
    base_date = datetime(2025, 12, 15) # Consistent base date
    num_days_for_availability = random.randint(NUM_DAYS_RANGE[0], NUM_DAYS_RANGE[1])

    for d_offset in range(num_days_for_availability):
        if random.random() < 0.05: # 5% chance of a full day off (reduced for more availability)
            continue
        
        day_start_dt = base_date + timedelta(days=d_offset)
        num_slots = random.randint(2, 4) # 2 to 4 availability slots per day
        for _ in range(num_slots):
            slot_start_hour = random.randint(7, 18) # Start between 7 AM and 6 PM
            slot_duration_hours = random.randint(4, 8) # Slot duration 4-8 hours
            slot_end_hour = slot_start_hour + slot_duration_hours
            
            if slot_end_hour > 23: # Ensure end hour doesn't go past midnight
                slot_end_hour = 23

            availability.append({
                "startTime": (day_start_dt + timedelta(hours=slot_start_hour)).isoformat(),
                "endTime": (day_start_dt + timedelta(hours=slot_end_hour)).isoformat()
            })
    
    return {
        "id": str(staff_id),
        "name": name,
        "roles": roles,
        "preferences": {
            "availability": availability,
            "desiredHours": desired_hours,
            "prefersConsecutiveDaysOff": prefers_consecutive_days_off
        },
        "avatar": f"https://i.pravatar.cc/150?img={staff_id}"
    }

def generate_shift(shift_id, day_offset, available_roles_from_staff):
    base_date = datetime(2025, 12, 15) # Consistent base date
    start_time_dt = base_date + timedelta(days=day_offset) + timedelta(hours=random.randint(8, 20), minutes=random.choice([0, 30]))
    end_time_dt = start_time_dt + timedelta(hours=random.randint(4, 8))
    
    if end_time_dt.day > start_time_dt.day: # Ensure shift doesn't end too late into the next day
        end_time_dt = datetime(start_time_dt.year, start_time_dt.month, start_time_dt.day, 23, 59)

    # Ensure the shift role is one that at least one staff member can perform
    role = random.choice(list(available_roles_from_staff))
    
    min_staff_for_role = 1 # For now, we're generating single assignments
    # if random.random() < 0.3: # 30% chance for a shift to require more staff
    #     min_staff_for_role = random.randint(1, 2) # Can be 1 or 2

    return {
        "id": str(shift_id),
        "role": role,
        "startTime": start_time_dt.isoformat(),
        "endTime": end_time_dt.isoformat(),
        "staffMemberId": None, # Initially unassigned, will be filled by rostering algorithm
        "minStaffForRole": min_staff_for_role
    }

def is_overlapping(shift1_start, shift1_end, shift2_start, shift2_end):
    return shift1_start < shift2_end and shift2_start < shift1_end

def check_hard_constraints(staff_member, shift, staff_current_assignments):
    # 1. Role Match
    if shift['role'] not in staff_member['roles']:
        return False

    # 2. Availability
    shift_start_dt = datetime.fromisoformat(shift['startTime'])
    shift_end_dt = datetime.fromisoformat(shift['endTime'])
    
    available_for_shift = False
    for avail_slot in staff_member['preferences']['availability']:
        avail_start_dt = datetime.fromisoformat(avail_slot['startTime'])
        avail_end_dt = datetime.fromisoformat(avail_slot['endTime'])
        
        if shift_start_dt >= avail_start_dt and shift_end_dt <= avail_end_dt:
            available_for_shift = True
            break
    
    if not available_for_shift:
        return False

    # 3. No Double-Booking
    for assigned_shift in staff_current_assignments:
        assigned_start_dt = datetime.fromisoformat(assigned_shift['startTime'])
        assigned_end_dt = datetime.fromisoformat(assigned_shift['endTime'])
        if is_overlapping(shift_start_dt, shift_end_dt, assigned_start_dt, assigned_end_dt):
            return False
            
    return True

def _assign_shift_recursive(shifts_to_assign, current_shift_idx, staff_members_map, staff_current_assignments, solution_roster_partial):
    if current_shift_idx == len(shifts_to_assign):
        # All shifts assigned, return a deep copy of the solution
        return [dict(s) for s in solution_roster_partial]

    current_shift = shifts_to_assign[current_shift_idx]
    
    # Find eligible staff for the current shift
    eligible_staff = []
    for staff_id, staff_member in staff_members_map.items():
        if check_hard_constraints(staff_member, current_shift, staff_current_assignments[staff_id]):
            eligible_staff.append(staff_member)
    
    random.shuffle(eligible_staff) # Randomize order to explore different paths

    for staff_member in eligible_staff:
        # Make assignment
        assigned_shift = {**current_shift, 'staffMemberId': staff_member['id']}
        solution_roster_partial.append(assigned_shift)
        staff_current_assignments[staff_member['id']].append(assigned_shift)

        # Recurse for the next shift
        result = _assign_shift_recursive(shifts_to_assign, current_shift_idx + 1, staff_members_map, staff_current_assignments, solution_roster_partial)
        if result is not None:
            return result # Found a complete solution

        # Backtrack: undo assignment
        staff_current_assignments[staff_member['id']].pop()
        solution_roster_partial.pop()
    
    return None # No solution found for this path

def generate_feasible_roster(staff_members, shifts):
    staff_members_map = {s['id']: s for s in staff_members}
    staff_current_assignments = {s['id']: [] for s in staff_members}
    solution_roster_partial = []

    # Sort shifts by number of eligible staff (fewer eligible staff first)
    # This helps the backtracking algorithm find solutions faster
    
    # Pre-calculate eligible staff count for each shift (without considering current assignments yet)
    num_eligible_staff_per_shift = {}
    for shift in shifts:
        count = 0
        for staff_member in staff_members:
            # Temporarily pass empty list for staff_current_assignments for initial eligibility check
            if check_hard_constraints(staff_member, shift, []): 
                count += 1
        num_eligible_staff_per_shift[shift['id']] = count

    shifts_to_assign = sorted(shifts, key=lambda s: num_eligible_staff_per_shift.get(s['id'], float('inf')))

    final_solution = _assign_shift_recursive(shifts_to_assign, 0, staff_members_map, staff_current_assignments, solution_roster_partial)

    # If backtracking fails to find a full solution, return None
    if final_solution is None:
        return None
    
    # Ensure all original shifts are present in the solution_roster, even if unassigned
    # The backtracking should ideally return a full solution, but this ensures consistency
    # This block is now redundant if _assign_shift_recursive always returns a complete solution or None
    # However, to be absolutely sure, we can reconstruct the full solution from the assigned shifts
    # and ensure all original shifts are accounted for.
    
    # Create a map of assigned shifts from the final_solution
    assigned_shifts_map = {s['id']: s for s in final_solution}
    
    full_solution_roster = []
    for original_shift in shifts:
        if original_shift['id'] in assigned_shifts_map:
            full_solution_roster.append(assigned_shifts_map[original_shift['id']])
        else:
            # This case should ideally not be reached if final_solution is truly complete
            # but as a safeguard, append unassigned if it somehow happens.
            full_solution_roster.append({**original_shift, 'staffMemberId': None})

    return full_solution_roster

def generate_problem_and_solution():
    MAX_RETRIES = 200 # Increased safety counter to prevent infinite loops
    for _ in range(MAX_RETRIES):
        current_num_staff = random.randint(*NUM_STAFF_RANGE)
        current_num_shifts_per_day = random.randint(*NUM_SHIFTS_PER_DAY_RANGE)
        current_num_days = random.randint(*NUM_DAYS_RANGE)

        staff_members = [generate_staff_member(i + 1) for i in range(current_num_staff)]
        
        # Collect all unique roles that the generated staff members can perform
        available_roles_from_staff = set()
        for staff in staff_members:
            available_roles_from_staff.update(staff['roles'])
        
        if not available_roles_from_staff: # Should not happen with current staff generation, but as a safeguard
            continue

        shifts = []
        shift_id_counter = 1
        for day_offset in range(current_num_days):
            for _ in range(current_num_shifts_per_day):
                # Pass the available roles to generate_shift
                shifts.append(generate_shift(shift_id_counter, day_offset, available_roles_from_staff))
                shift_id_counter += 1
        
        solution_roster = generate_feasible_roster(staff_members, shifts)

        if solution_roster is not None and all(s['staffMemberId'] is not None for s in solution_roster):
            return {
                "staff": staff_members,
                "shifts": shifts, # Original unassigned shifts
                "solution_roster": solution_roster
            }
    
    # If after MAX_RETRIES, no complete solution is found, return None
    print(f"Warning: Could not generate a complete solution after {MAX_RETRIES} retries. Discarding this problem.")
    return None

# --- Main Execution ---
if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__), OUTPUT_DIR), exist_ok=True)

    dataset = []
    generated_count = 0

    print(f"Generating {NUM_EXAMPLES} rostering problem examples...")
    
    pbar = tqdm(total=NUM_EXAMPLES, desc="Generating examples")
    while generated_count < NUM_EXAMPLES:
        problem_data = generate_problem_and_solution()
        if problem_data is not None:
            dataset.append(problem_data)
            generated_count += 1
            pbar.update(1)
    pbar.close()

    with open(os.path.join(os.path.dirname(__file__), OUTPUT_DIR, 'rostering_dataset.json'), 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Dataset saved to {os.path.join(os.path.dirname(__file__), OUTPUT_DIR, 'rostering_dataset.json')}")
    print("Please ensure you have Python installed with 'pip install numpy scikit-learn tqdm'.")
    print(f"You can run this script using: python {sys.argv[0]}")