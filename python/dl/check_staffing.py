import ijson
import sys

def check_roster_staffing(file_path):
    """
    Checks if all solution rosters in a large JSON file are fully staffed.
    Assumes a specific JSON structure and definition of "fully staffed".
    """
    try:
        with open(file_path, 'rb') as f:
            # Use ijson to parse the large JSON file iteratively
            # 'item' corresponds to each object in the top-level array
            for i, roster_data in enumerate(ijson.items(f, 'item')):
                print(f"Checking roster {i+1}...")
                
                staff_by_id = {s['id']: s for s in roster_data.get('staff', [])}
                shifts_required = roster_data.get('shifts_required', [])
                solution_roster = roster_data.get('solution_roster', [])

                # Check 1: All required shifts have enough staff
                shift_assignments = {}
                for assignment in solution_roster:
                    shift_id = assignment.get('shift_id')
                    if shift_id:
                        shift_assignments.setdefault(shift_id, []).append(assignment)

                for required_shift in shifts_required:
                    shift_id = required_shift.get('shift_id')
                    required_count = required_shift.get('required_count', 1) # Default to 1 if not specified

                    assigned_staff_count = len(shift_assignments.get(shift_id, []))

                    if assigned_staff_count < required_count:
                        print(f"  Roster {i+1} is NOT fully staffed: Shift '{shift_id}' requires {required_count} staff but has {assigned_staff_count}.")
                        return False # Found an understaffed roster

                    # Check 2: Assigned staff have the required roles
                    role_needed = required_shift.get('role_needed')
                    for assignment in shift_assignments.get(shift_id, []):
                        staff_id = assignment.get('staff_id')
                        staff_member = staff_by_id.get(staff_id)

                        if not staff_member:
                            print(f"  Roster {i+1} is NOT fully staffed: Staff ID '{staff_id}' not found for shift '{shift_id}'.")
                            return False
                        
                        if role_needed and role_needed not in staff_member.get('roles', []):
                            print(f"  Roster {i+1} is NOT fully staffed: Staff '{staff_member['name']}' (ID: {staff_id}) assigned to shift '{shift_id}' does not have required role '{role_needed}'.")
                            return False
            
            print("All rosters checked appear to be fully staffed according to the defined criteria.")
            return True

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    json_file_path = "/Users/woodj/Desktop/roster-forge/server/dl_poc/data/rostering_dataset.json"
    
    # IMPORTANT: This script assumes a specific JSON structure for 'shifts_required' and 'solution_roster'.
    # If your JSON structure is different, you will need to adjust the parsing logic within the check_roster_staffing function.
    # The current snippet of your file only showed 'staff' information.
    
    print("Attempting to check staffing for rosters in the dataset...")
    is_fully_staffed = check_roster_staffing(json_file_path)
    print(f"\nOverall result: {'All rosters are fully staffed' if is_fully_staffed else 'Some rosters are NOT fully staffed'}")
