import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import json
import sys
from datetime import datetime, timedelta
import numpy as np
import os

# --- GNN Model Architecture (must match train_gnn.py) ---
class RosterGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(2 * hidden_channels, 1) 

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

    def predict_edge(self, node_embeddings, edge_index_to_predict):
        row, col = edge_index_to_predict
        combined_features = torch.cat([node_embeddings[row], node_embeddings[col]], dim=-1)
        return self.lin(combined_features)

# --- Data Preprocessing (must match create_gnn_dataset.py) ---
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

def preprocess_single_instance(instance, all_roles, role_to_idx, num_roles):
    staff_members = instance['staff']
    shifts = instance['shifts']

    if not staff_members or not shifts:
        return None, None, None, None # Return None for data, staff_map, shift_map, edge_map

    staff_id_to_idx = {sm['id']: i for i, sm in enumerate(staff_members)}
    shift_id_to_idx = {s['id']: i for i, s in enumerate(shifts)}

    # Node features for staff (type 0)
    staff_features = []
    for sm in staff_members:
        roles_one_hot = [0] * num_roles
        for role in sm['roles']:
            if role in role_to_idx:
                roles_one_hot[role_to_idx[role]] = 1
        
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

    num_staff = len(staff_members)
    num_shifts = len(shifts)

    max_feature_dim = max(x_staff.shape[1], x_shift.shape[1])
    
    padded_x_staff = torch.cat([x_staff, torch.zeros(num_staff, max_feature_dim - x_staff.shape[1])], dim=1)
    padded_x_shift = torch.cat([x_shift, torch.zeros(num_shifts, max_feature_dim - x_shift.shape[1])], dim=1)

    x = torch.cat([padded_x_staff, padded_x_shift], dim=0)

    edge_index_list = []
    edge_map = {} # Map (staff_id, shift_id) to edge_index_list position
    
    for staff_idx, sm in enumerate(staff_members):
        for shift_idx, s in enumerate(shifts):
            staff_has_role = s['role'] in sm['roles']

            staff_available = False
            shift_start = datetime.fromisoformat(s['startTime'])
            shift_end = datetime.fromisoformat(s['endTime'])

            for availability in sm['preferences']['availability']:
                avail_start = datetime.fromisoformat(availability['startTime'])
                avail_end = datetime.fromisoformat(availability['endTime'])

                if max(avail_start, shift_start) < min(avail_end, shift_end):
                    staff_available = True
                    break
            
            if staff_has_role and staff_available:
                edge_index_list.append([staff_idx, shift_idx + num_staff])
                edge_map[(sm['id'], s['id'])] = len(edge_index_list) - 1
    
    if not edge_index_list:
        return None, None, None, None

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    data.num_staff_nodes = num_staff
    data.num_shift_nodes = num_shifts
    
    return data, staff_id_to_idx, shift_id_to_idx, edge_map

# --- Inference Function ---
def run_inference(input_data_json):
    MODEL_SAVE_PATH = '/Users/woodj/Desktop/roster-forge/server/dl_poc/data/rostering_gnn_model.pth'
    
    # Load the trained model
    # We need to know num_node_features from the training data
    # For now, let's assume a fixed value or derive it from the input data
    # A more robust solution would save num_node_features with the model or dataset.
    
    # To get num_node_features, we need to preprocess a sample or load the dataset metadata
    # For simplicity, let's load the first graph from the dataset to get num_node_features
    DATASET_PATH = '/Users/woodj/Desktop/roster-forge/server/dl_poc/data/gnn_rostering_dataset.pt'
    try:
        sample_graphs = torch.load(DATASET_PATH, weights_only=False)
        if not sample_graphs:
            raise ValueError("GNN dataset is empty, cannot determine num_node_features.")
        num_node_features = sample_graphs[0].x.shape[1]
    except Exception as e:
        print(f"Error loading sample GNN dataset to determine num_node_features: {e}", file=sys.stderr)
        sys.exit(1)

    hidden_channels = 64 # Must match training script
    model = RosterGNN(num_node_features, hidden_channels)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    # Preprocess the input instance
    # Need to get all_roles, role_to_idx, num_roles from the training data as well
    # For now, let's re-derive them from the original rostering_dataset.json
    # A robust solution would save these mappings.
    original_dataset_path = '/Users/woodj/Desktop/roster-forge/server/dl_poc/data/rostering_dataset.json'
    with open(original_dataset_path, 'r') as f:
        original_rostering_data = json.load(f)
    
    all_roles = sorted(list(set(role for instance in original_rostering_data for staff_member in instance['staff'] for role in staff_member['roles']) |
                             set(instance['shifts'][0]['role'] for instance in original_rostering_data if instance['shifts'] for _ in instance['shifts'])))
    role_to_idx = {role: i for i, role in enumerate(all_roles)}
    num_roles = len(all_roles)

    data, staff_id_to_idx, shift_id_to_idx, edge_map = preprocess_single_instance(
        input_data_json, all_roles, role_to_idx, num_roles
    )

    if data is None:
        print(json.dumps({"error": "No valid staff or shifts in input data."}))
        sys.exit(1)

    with torch.no_grad():
        node_embeddings = model(data.x, data.edge_index)
        logits = model.predict_edge(node_embeddings, data.edge_index)
        probabilities = torch.sigmoid(logits)
    
    # Post-process predictions
    predicted_assignments = []
    # Iterate through the edge_map to reconstruct assignments
    for (staff_id, shift_id), edge_pos in edge_map.items():
        if probabilities[edge_pos].item() > 0.5: # Threshold for assignment
            predicted_assignments.append({
                "staffId": staff_id,
                "shiftId": shift_id,
                "probability": probabilities[edge_pos].item()
            })
    
    print(json.dumps({"predictions": predicted_assignments}))

if __name__ == '__main__':
    # The script expects a single JSON string as a command-line argument
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input JSON provided."}))
        sys.exit(1)
    
    input_json_str = sys.argv[1]
    try:
        input_data = json.loads(input_json_str)
    except json.JSONDecodeError:
        print(json.dumps({"error": "Invalid JSON input."}))
        sys.exit(1)
    
    run_inference(input_data)
