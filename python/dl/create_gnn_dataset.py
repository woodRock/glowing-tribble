import json
import torch
from torch_geometric.data import Data
from datetime import datetime, timedelta
import numpy as np
import os

def time_to_features(dt_str):
    dt_obj = datetime.fromisoformat(dt_str)
    return [
        dt_obj.hour,
        dt_obj.weekday(), # Monday is 0, Sunday is 6
        dt_obj.day,
        dt_obj.month
    ]

def calculate_duration_minutes(start_str, end_str):
    start_obj = datetime.fromisoformat(start_str)
    end_obj = datetime.fromisoformat(end_str)
    return (end_obj - start_obj).total_seconds() / 60

def create_gnn_dataset(input_json_path, output_dir):
    with open(input_json_path, 'r') as f:
        rostering_data = json.load(f)

    all_graphs = []
    all_roles = sorted(list(set(role for instance in rostering_data for staff_member in instance['staff'] for role in staff_member['roles']) |
                             set(instance['shifts'][0]['role'] for instance in rostering_data if instance['shifts'] for _ in instance['shifts'])))
    role_to_idx = {role: i for i, role in enumerate(all_roles)}
    num_roles = len(all_roles)

    for instance_idx, instance in enumerate(rostering_data):
        staff_members = instance['staff']
        shifts = instance['shifts']
        solution_roster = instance['solution_roster']

        if not staff_members or not shifts:
            continue

        # Node features for staff (type 0)
        staff_features = []
        staff_id_to_idx = {sm['id']: i for i, sm in enumerate(staff_members)}
        
        for sm in staff_members:
            roles_one_hot = [0] * num_roles
            for role in sm['roles']:
                if role in role_to_idx:
                    roles_one_hot[role_to_idx[role]] = 1
            
            # Calculate total available hours for the staff member
            total_available_minutes = 0
            for availability in sm['preferences']['availability']:
                total_available_minutes += calculate_duration_minutes(availability['startTime'], availability['endTime'])
            
            staff_features.append(
                roles_one_hot + 
                [sm['preferences']['desiredHours'], 
                 int(sm['preferences']['prefersConsecutiveDaysOff']),
                 total_available_minutes / 60 # Convert to hours
                ]
            )
        
        x_staff = torch.tensor(staff_features, dtype=torch.float)

        # Node features for shifts (type 1)
        shift_features = []
        shift_id_to_idx = {s['id']: i for i, s in enumerate(shifts)}

        for s in shifts:
            role_one_hot = [0] * num_roles
            if s['role'] in role_to_idx:
                role_one_hot[role_to_idx[s['role']]] = 1
            
            duration_minutes = calculate_duration_minutes(s['startTime'], s['endTime'])
            start_features = time_to_features(s['startTime'])
            end_features = time_to_features(s['endTime'])

            shift_features.append(
                role_one_hot + 
                [duration_minutes, s['minStaffForRole']] + 
                start_features + end_features
            )
        
        x_shift = torch.tensor(shift_features, dtype=torch.float)

        # Combine node features
        # We'll use a heterogeneous graph approach or simply concatenate if we treat all nodes as one type
        # For simplicity, let's treat them as one type for now, with an indicator feature
        # Staff nodes will have a 'type' feature of 0, shift nodes will have 'type' feature of 1
        
        # Pad staff features to match shift features length, or vice-versa, or use separate node types
        # For now, let's create a bipartite graph structure with separate node types in PyG Data object
        # This means x_staff and x_shift will be separate attributes.

        # Edge index and labels
        edge_index_list = []
        edge_label_list = [] # 1 if assigned, 0 if not assigned but eligible

        # Create a set of actual assignments from the solution roster for quick lookup
        actual_assignments = set()
        for sol_shift in solution_roster:
            if sol_shift['staffMemberId'] is not None:
                actual_assignments.add((sol_shift['staffMemberId'], sol_shift['id']))
        
        # Debugging prints
        
        
        

        # Iterate through all possible staff-shift pairs to determine eligibility and create edges
        for staff_idx, sm in enumerate(staff_members):
            for shift_idx, s in enumerate(shifts):
                # Check role eligibility
                staff_has_role = s['role'] in sm['roles']

                # Check availability
                staff_available = False
                shift_start = datetime.fromisoformat(s['startTime'])
                shift_end = datetime.fromisoformat(s['endTime'])

                for availability in sm['preferences']['availability']:
                    avail_start = datetime.fromisoformat(availability['startTime'])
                    avail_end = datetime.fromisoformat(availability['endTime'])

                    # Check for overlap
                    if max(avail_start, shift_start) < min(avail_end, shift_end):
                        staff_available = True
                        break
                
                if staff_has_role and staff_available:
                    # Edge from staff to shift
                    edge_index_list.append([staff_idx, shift_idx])
                    
                    # Determine label
                    is_assigned = (sm['id'], s['id']) in actual_assignments
                    edge_label_list.append(1 if is_assigned else 0)
        
        if not edge_index_list:
            # No eligible assignments, skip this instance
            print(f"Instance {instance_idx}: No eligible edges, skipping.")
            continue

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_label = torch.tensor(edge_label_list, dtype=torch.float)

        # Create a PyG Data object
        # For a bipartite graph, we can define node types and use HeteroData,
        # but for simplicity with a single edge type (staff-shift), we can use a standard Data object
        # where staff nodes are 0 to N-1 and shift nodes are N to N+M-1.
        # However, PyG's Data object is more flexible. Let's keep staff and shift features separate
        # and define edge_index as (staff_node_idx, shift_node_idx).
        
        # The `Data` object can hold multiple node feature matrices and edge indices.
        # For a bipartite graph, we typically have two node types and edges between them.
        # PyG's `Data` object can represent this by having `x` for one type and `x_other` for another,
        # or by using `HeteroData`. For this problem, let's stick to a simple `Data` object
        # where `x` contains all node features (staff then shifts) and `edge_index` points accordingly.
        
        # Let's refine the node feature creation for a single `x` tensor
        # Staff nodes will be indexed 0 to num_staff - 1
        # Shift nodes will be indexed num_staff to num_staff + num_shifts - 1

        num_staff = len(staff_members)
        num_shifts = len(shifts)

        # To combine features, they need to have the same dimension.
        # This is a common challenge with heterogeneous graphs.
        # For now, let's pad the smaller feature vectors with zeros to match the larger one.
        # A better approach would be to use separate embedding layers for each node type.

        max_feature_dim = max(x_staff.shape[1], x_shift.shape[1])
        
        padded_x_staff = torch.cat([x_staff, torch.zeros(num_staff, max_feature_dim - x_staff.shape[1])], dim=1)
        padded_x_shift = torch.cat([x_shift, torch.zeros(num_shifts, max_feature_dim - x_shift.shape[1])], dim=1)

        x = torch.cat([padded_x_staff, padded_x_shift], dim=0)

        # Adjust edge_index for combined node features
        # Staff indices remain as is, shift indices are offset by num_staff
        adjusted_edge_index_list = []
        for staff_node_idx, shift_node_idx in edge_index_list:
            adjusted_edge_index_list.append([staff_node_idx, shift_node_idx + num_staff])
        
        adjusted_edge_index = torch.tensor(adjusted_edge_index_list, dtype=torch.long).t().contiguous()

        # Create the Data object
        data = Data(x=x, edge_index=adjusted_edge_index, edge_label=edge_label)
        data.num_staff_nodes = num_staff # Store number of staff nodes for later splitting
        data.num_shift_nodes = num_shifts # Store number of shift nodes
        data.original_staff_data = staff_members # Store original staff data
        data.original_shifts_data = shifts # Store original shifts data
        data.staff_id_to_node_idx = staff_id_to_idx
        data.shift_id_to_node_idx = shift_id_to_idx
        
        
        
        all_graphs.append(data)

    output_file_path = os.path.join(output_dir, 'gnn_rostering_dataset.pt')
    torch.save(all_graphs, output_file_path)
    print(f"Successfully created GNN dataset with {len(all_graphs)} graphs and saved to {output_file_path}")

if __name__ == '__main__':
    input_path = '/Users/woodj/Desktop/roster-forge/server/dl_poc/data/rostering_dataset.json'
    output_directory = '/Users/woodj/Desktop/roster-forge/server/dl_poc/data'
    
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    create_gnn_dataset(input_path, output_directory)
