import operator
import random
from deap import gp, base
# Assuming RosterProblem and related classes/functions are defined here
from roster_problem import shifts_overlap, is_staff_available_for_shift, RosterProblem 

# --- Helper functions for GP ---

def protectedDiv(left, right):
    """Safely divides, returning 1.0 on division by zero."""
    try:
        return left / right
    except ZeroDivisionError:
        return 1.0

def if_then_else(input_cond, output_true, output_false):
    """If input_cond > 0.5, returns output_true, else output_false."""
    return output_true if input_cond > 0.5 else output_false

def num_lt(left, right):
    """Returns 1.0 (float) if left < right, else 0.0 (comparison primitive)."""
    return 1.0 if left < right else 0.0

def num_gt(left, right):
    """Returns 1.0 (float) if left > right, else 0.0 (comparison primitive)."""
    return 1.0 if left > right else 0.0

def protected_sqrt(x):
    """Safely calculates square root."""
    return abs(x)**0.5

# --- GPDecision Class ---
class GPDecision:
    def __init__(self, name, terminals_map):
        self.name = name
        self.terminals_map = terminals_map
        self.pset = self._create_primitive_set()

    def _create_primitive_set(self):
        # Arity is defined by the number of terminals/input arguments
        arity = len(self.terminals_map)
        
        # FIX: Corrected function name to PrimitiveSetTyped
        pset = gp.PrimitiveSetTyped(self.name, [float]*arity, float) 

        # Add standard arithmetic primitives (input types: [float, float], return type: float)
        pset.addPrimitive(operator.add, [float, float], float)
        pset.addPrimitive(operator.sub, [float, float], float)
        pset.addPrimitive(operator.mul, [float, float], float)
        pset.addPrimitive(protectedDiv, [float, float], float)
        pset.addPrimitive(operator.neg, [float], float)
        pset.addPrimitive(max, [float, float], float)
        pset.addPrimitive(min, [float, float], float)
        pset.addPrimitive(abs, [float], float)
        
        # Add comparison and conditional primitives
        pset.addPrimitive(num_lt, [float, float], float)
        pset.addPrimitive(num_gt, [float, float], float)
        pset.addPrimitive(if_then_else, [float, float, float], float) 
        
        # Add constant terminals
        pset.addTerminal(1.0, float, name="ONE") 
        pset.addTerminal(0.0, float, name="ZERO") 
        
        # Rename arguments
        for arg_idx, arg_name in self.terminals_map.items():
            pset.renameArguments(**{f'ARG{arg_idx}': arg_name})
        
        return pset

    def compile_heuristic(self, individual):
        """Compiles an individual (PrimitiveTree) into a callable function."""
        return gp.compile(individual, self.pset)

# --- Terminal Definitions for each Decision ---

D1_TERMINALS = {0: 'unassigned_shift_ratio', 1: 'staff_utilization_ratio', 2: 'current_roster_quality', 3: 'num_eligible_staff'}
D1_Decision = GPDecision("D1_AssignStaffNow", D1_TERMINALS)

D2_TERMINALS = {0: 'staff_skill_count', 1: 'shift_req_skill_count', 2: 'staff_available_for_shift', 
                3: 'staff_has_all_req_skills', 4: 'staff_min_rest_time', 5: 'staff_max_workload', 
                6: 'staff_desired_hours', 7: 'staff_current_shift_count'}
D2_Decision = GPDecision("D2_ChooseStaffForShift", D2_TERMINALS)

D3_TERMINALS = {0: 'staff_idle_time', 1: 'num_available_shifts', 2: 'staff_preference_to_work', 3: 'current_roster_quality'}
D3_Decision = GPDecision("D3_AssignIdleStaff", D3_TERMINALS)

D4_TERMINALS = {0: 'skill_match_count', 1: 'is_shift_on_request', 2: 'shift_duration', 
                3: 'num_eligible_staff'}
D4_Decision = GPDecision("D4_ChooseShiftForStaff", D4_TERMINALS)

D5_TERMINALS = {0: 'consecutive_working_days', 1: 'total_hours_worked', 2: 'hours_since_last_shift', 
                3: 'shifts_worked_count'}
D5_Decision = GPDecision("D5_FatigueAdjustment", D5_TERMINALS)

# --- Orchestrator / Hyper-Heuristic Simulation ---
def evaluate_hyper_heuristic_roster(meta_individual, roster_problem):
    """
    This function acts as the orchestrator, using the evolved heuristics
    to build a complete roster and then evaluating it.
    Uses all 5 decisions: D1 (Shift Order), D2 (Staff Selection), D3 (Idle Staff Priority),
    D4 (Shift Selection for Staff), D5 (Post-Shift Adjustment).
    """
    func_d1 = D1_Decision.compile_heuristic(meta_individual[0])
    func_d2 = D2_Decision.compile_heuristic(meta_individual[1])
    func_d3 = D3_Decision.compile_heuristic(meta_individual[2])
    func_d4 = D4_Decision.compile_heuristic(meta_individual[3])
    func_d5 = D5_Decision.compile_heuristic(meta_individual[4])

    assignments = []
    staff_current_schedules = {staff_id: [] for staff_id in roster_problem.staff_ids}
    staff_assigned_shift_ids = {staff_id: set() for staff_id in roster_problem.staff_ids}
    
    unassigned_shifts = list(roster_problem.shifts.values())
    total_shifts_count = len(unassigned_shifts)
    
    # --- helper to calculate roster quality ---
    def get_roster_quality():
        if total_shifts_count == 0: return 0.0
        return len(assignments) / total_shifts_count

    # --- helper for D5 Fatigue Features ---
    def get_fatigue_features(staff_id, current_shift_start):
        shifts = sorted(staff_current_schedules[staff_id], key=lambda s: s.end_time)
        
        shifts_worked_count = len(shifts)
        total_hours_worked = sum((s.end_time - s.start_time).total_seconds() / 3600.0 for s in shifts)
        
        hours_since_last_shift = 0.0
        if shifts:
            last_end = shifts[-1].end_time
            hours_since_last_shift = (current_shift_start - last_end).total_seconds() / 3600.0
            # Cap at some reasonable number if it's very large (e.g. first shift of week)
            if hours_since_last_shift < 0: hours_since_last_shift = 0.0
        else:
             hours_since_last_shift = 999.0 # Effectively infinite rest
        
        # Consecutive working days (simplified: count backwards from last shift)
        consecutive_working_days = 0
        if shifts:
            visited_days = set()
            for s in reversed(shifts):
                d = s.start_time.date()
                if d in visited_days: continue
                visited_days.add(d)
                # Check continuity is hard without full calendar iteration, 
                # but we can check if gaps are <= 1 day.
                # Simplified: just count shifts as proxy or simple day count
                consecutive_working_days += 1 

        return consecutive_working_days, total_hours_worked, hours_since_last_shift, shifts_worked_count

    # ==========================================================
    # PHASE 1: Shift-Centric Assignment (Uses D1, D2, D5)
    # ==========================================================
    
    # 1. Score and Sort Shifts using D1
    shift_scores = []
    
    # Pre-calc global stats for D1
    total_max_workload = 0
    for s in roster_problem.staff.values():
        if not s.contract_details:
             total_max_workload += 2400
             continue
        c = s.contract_details[0]
        val = 2400
        # Defensive check: if contract is still a dict for some reason, handle it
        if isinstance(c, dict):
             mw = c.get('max_workload_minutes', float('inf'))
             if mw != float('inf'): val = mw
        else:
             if c.max_workload_minutes != float('inf'):
                 val = c.max_workload_minutes
        total_max_workload += val

    # Pre-calculate static eligibility (performance optimization)
    static_eligibility_counts = {}
    for shift in unassigned_shifts:
        count = 0
        for staff_obj in roster_problem.staff.values():
            if is_staff_available_for_shift(staff_obj, shift) and staff_obj.skills.issuperset(shift.required_skills):
                # Also check contract valid shifts if present
                contract = staff_obj.contract_details[0]
                if hasattr(contract, 'valid_shifts') and contract.valid_shifts and shift.role not in contract.valid_shifts:
                    continue
                count += 1
        static_eligibility_counts[shift.id] = count
    
    for shift in unassigned_shifts:
        # D1 Terminals
        current_assigned_workload = sum(
             sum((s.end_time - s.start_time).total_seconds() / 60 for s in sched)
             for sched in staff_current_schedules.values()
        )
        
        unassigned_shift_ratio = len(unassigned_shifts) / total_shifts_count if total_shifts_count > 0 else 0
        staff_utilization_ratio = current_assigned_workload / total_max_workload if total_max_workload > 0 else 0
        curr_quality = get_roster_quality()
        num_eligible = static_eligibility_counts.get(shift.id, 0)
        
        try:
            score = func_d1(float(unassigned_shift_ratio), float(staff_utilization_ratio), float(curr_quality), float(num_eligible))
        except Exception:
            score = 0.0
        shift_scores.append((score, shift))
    
    # Sort descending
    shift_scores.sort(key=lambda x: x[0], reverse=True)
    sorted_shifts = [s for _, s in shift_scores]

    for shift_to_assign in sorted_shifts:
        # Check if already assigned (in case of dynamic updates, though here we iterate static list)
        if any(a[0] == shift_to_assign.id for a in assignments):
            continue

        eligible_staff_with_scores = []

        for staff_obj in roster_problem.staff.values():
            # --- Eligibility Checks ---
            if not is_staff_available_for_shift(staff_obj, shift_to_assign): continue
            if not staff_obj.skills.issuperset(shift_to_assign.required_skills): continue
            
            is_overlapping = False
            for assigned_shift_obj in staff_current_schedules[staff_obj.id]:
                if shifts_overlap(shift_to_assign, assigned_shift_obj):
                    is_overlapping = True; break
            if is_overlapping: continue
            
            contract = staff_obj.contract_details[0]
            if hasattr(contract, 'valid_shifts') and contract.valid_shifts and shift_to_assign.role not in contract.valid_shifts: continue

            current_workload = sum((s.end_time - s.start_time).total_seconds() / 60 for s in staff_current_schedules[staff_obj.id])
            shift_duration = (shift_to_assign.end_time - shift_to_assign.start_time).total_seconds() / 60
            max_workload = contract.max_workload_minutes
            if max_workload != float('inf') and (current_workload + shift_duration) > max_workload: continue
            # --- End Checks ---

            # D2 Terminals (Staff Selection)
            t_skill_count = len(staff_obj.skills)
            t_shift_req = len(shift_to_assign.required_skills)
            t_avail = 1.0
            t_has_skills = 1.0
            t_min_rest = contract.min_rest_time
            t_max_work = max_workload if max_workload != float('inf') else 9999.0
            t_desired = staff_obj.preferences.get('desired_hours', 0)
            t_count = len(staff_current_schedules[staff_obj.id])

            try:
                base_score = func_d2(float(t_skill_count), float(t_shift_req), t_avail, t_has_skills,
                                     float(t_min_rest), float(t_max_work), float(t_desired), float(t_count))
            except Exception:
                base_score = -1000.0

            # D5 Adjustment (Fatigue)
            # Only if staff has prior shifts
            d5_adjustment = 0.0
            if staff_current_schedules[staff_obj.id]:
                c_days, t_hours, h_since, s_count = get_fatigue_features(staff_obj.id, shift_to_assign.start_time)
                try:
                    d5_adjustment = func_d5(float(c_days), float(t_hours), float(h_since), float(s_count))
                except Exception:
                    d5_adjustment = 0.0
            
            final_score = base_score + d5_adjustment
            eligible_staff_with_scores.append((final_score, staff_obj))
        
        if eligible_staff_with_scores:
            eligible_staff_with_scores.sort(key=lambda x: x[0], reverse=True)
            best_score, best_staff = eligible_staff_with_scores[0]
            assignments.append((shift_to_assign.id, best_staff.id))
            staff_current_schedules[best_staff.id].append(shift_to_assign)
            staff_assigned_shift_ids[best_staff.id].add(shift_to_assign.id)

    # ==========================================================
    # PHASE 2: Gap Filling / Idle Staff (Uses D3, D4)
    # ==========================================================
    
    # Refresh unassigned shifts
    remaining_unassigned = [s for s in roster_problem.shifts.values() if not any(a[0] == s.id for a in assignments)]
    
    if remaining_unassigned:
        # Score Staff using D3 (Idle Staff)
        staff_d3_scores = []
        for staff_obj in roster_problem.staff.values():
            # Calculate simple idle time metric (e.g. total hours not working vs total capacity)
            contract = staff_obj.contract_details[0]
            max_w = contract.max_workload_minutes if contract.max_workload_minutes != float('inf') else 40*60
            curr_w = sum((s.end_time - s.start_time).total_seconds() / 60 for s in staff_current_schedules[staff_obj.id])
            idle_metric = max(0.0, max_w - curr_w)
            
            # D3 Terminals
            t_idle = idle_metric
            t_avail_shifts = len(remaining_unassigned)
            t_pref = staff_obj.preferences.get('desired_hours', 0) * 60 - curr_w # Unmet preference
            t_qual = get_roster_quality()

            try:
                s3_score = func_d3(float(t_idle), float(t_avail_shifts), float(t_pref), float(t_qual))
            except Exception:
                s3_score = -999.0
            staff_d3_scores.append((s3_score, staff_obj))
        
        staff_d3_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Try to assign shifts for top staff from D3
        for _, staff_obj in staff_d3_scores:
            if not remaining_unassigned: break
            
            # Find eligible shifts for this staff
            candidate_shifts = []
            for shift in remaining_unassigned:
                 # Reuse eligibility checks logic (simplified here for brevity, ideally factor out)
                if not is_staff_available_for_shift(staff_obj, shift): continue
                if not staff_obj.skills.issuperset(shift.required_skills): continue
                # ... check overlaps ...
                is_overlap = False
                for assigned in staff_current_schedules[staff_obj.id]:
                    if shifts_overlap(shift, assigned): is_overlap=True; break
                if is_overlap: continue
                # ... check workload ...
                contract = staff_obj.contract_details[0]
                s_dur = (shift.end_time - shift.start_time).total_seconds() / 60
                curr_w = sum((s.end_time - s.start_time).total_seconds() / 60 for s in staff_current_schedules[staff_obj.id])
                if contract.max_workload_minutes != float('inf') and (curr_w + s_dur) > contract.max_workload_minutes: continue

                # D4 Terminals (Shift Selection for Staff)
                t_match = len(staff_obj.skills.intersection(shift.required_skills))
                t_on_req = 1.0 if any(r.shift_id == shift.id or r.shift_id == shift.role for r in roster_problem.shift_on_requests if r.employee_id == staff_obj.id) else 0.0
                t_dur = s_dur
                
                # Check how many OTHER staff are eligible (approximation of difficulty)
                # Quick count
                others_count = 0
                for s_other in roster_problem.staff.values():
                    if s_other.id == staff_obj.id: continue
                    if s_other.skills.issuperset(shift.required_skills) and is_staff_available_for_shift(s_other, shift):
                         others_count += 1
                t_others = others_count

                try:
                    s4 = func_d4(float(t_match), t_on_req, float(t_dur), float(t_others))
                except Exception:
                    s4 = -999.0
                candidate_shifts.append((s4, shift))
            
            if candidate_shifts:
                candidate_shifts.sort(key=lambda x: x[0], reverse=True)
                best_s4, best_shift = candidate_shifts[0]
                
                # Assign
                assignments.append((best_shift.id, staff_obj.id))
                staff_current_schedules[staff_obj.id].append(best_shift)
                staff_assigned_shift_ids[staff_obj.id].add(best_shift.id)
                
                # Remove from local list to prevent double assignment
                remaining_unassigned = [s for s in remaining_unassigned if s.id != best_shift.id]

    # Evaluate the generated roster
    fitness = roster_problem.evaluate_roster(assignments)
    return fitness, assignments