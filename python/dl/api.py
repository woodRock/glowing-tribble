from flask import Flask, request, jsonify
import json
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder # Needed for loading the encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# --- Configuration ---
MODEL_PATH = 'data/rostering_transformer_model.pth'
SCALER_PATH = 'data/transformer_scaler.joblib'
LABEL_ENCODER_PATH = 'data/transformer_role_encoder.joblib'

# GNN Configuration
GNN_MODEL_PATH = 'data/rostering_gnn_model.pth'
GNN_DATASET_PATH = 'data/gnn_rostering_dataset.pt'
ORIGINAL_DATASET_PATH = 'data/rostering_dataset.json'

app = Flask(__name__)

# --- Define the Transformer Model (must match train_transformer.py) ---
class RosteringTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_encoder_layers):
        super(RosteringTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, src):
        embedded_src = self.embedding(src)
        transformer_output = self.transformer_encoder(embedded_src)
        output = self.output_layer(transformer_output)
        return output.squeeze(-1)

# --- Define the GNN Model (must match train_gnn.py) ---
class RosterGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels) # Added a third GCN layer
        self.lin = torch.nn.Linear(2 * hidden_channels, 1) 
        self.dropout = torch.nn.Dropout(0.5) # Added dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x) # Apply dropout
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x) # Apply dropout
        x = self.conv3(x, edge_index) # Pass through the third GCN layer
        x = F.relu(x)
        x = self.dropout(x) # Apply dropout
        return x

    def predict_edge(self, node_embeddings, edge_index_to_predict):
        row, col = edge_index_to_predict
        combined_features = torch.cat([node_embeddings[row], node_embeddings[col]], dim=-1)
        return self.lin(combined_features)

# --- Global Model Instances ---
transformer_model = None
transformer_scaler = None
transformer_role_encoder = None

gnn_model = None
gnn_num_node_features = None
gnn_all_roles = None
gnn_role_to_idx = None
gnn_num_roles = None
GNN_HIDDEN_CHANNELS = 64 # Must match training script

# Transformer specific parameters (must match train_transformer.py)
EMBED_DIM = 64
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2

def load_resources():
    global transformer_model, transformer_scaler, transformer_role_encoder
    global gnn_model, gnn_num_node_features, gnn_all_roles, gnn_role_to_idx, gnn_num_roles

    # Load Transformer resources
    try:
        transformer_scaler = joblib.load(SCALER_PATH)
        transformer_role_encoder = joblib.load(LABEL_ENCODER_PATH)
        
        input_dim = transformer_scaler.n_features_in_
        
        transformer_model = RosteringTransformer(input_dim, EMBED_DIM, NUM_HEADS, NUM_ENCODER_LAYERS)
        transformer_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        transformer_model.eval()
        print("Transformer model, scaler, and role encoder loaded successfully.")
    except FileNotFoundError:
        print("Error: Transformer model, scaler, or role encoder not found. Please run train_transformer.py first.")
        # Exit if resources are not found, as Transformer is the default
        exit(1) 
    except Exception as e:
        print(f"Error loading Transformer resources: {e}")
        exit(1)

    # Load GNN resources
    try:
        # Determine num_node_features from the GNN dataset
        sample_graphs = torch.load(GNN_DATASET_PATH, weights_only=False)
        if not sample_graphs:
            raise ValueError("GNN dataset is empty, cannot determine num_node_features.")
        gnn_num_node_features = sample_graphs[0].x.shape[1]

        gnn_model = RosterGNN(gnn_num_node_features, GNN_HIDDEN_CHANNELS)
        gnn_model.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=torch.device('cpu')))
        gnn_model.eval()

        # Derive all_roles, role_to_idx, num_roles from the original dataset
        with open(ORIGINAL_DATASET_PATH, 'r') as f:
            original_rostering_data = json.load(f)
        
        all_roles_set = set()
        for instance in original_rostering_data:
            for staff_member in instance['staff']:
                all_roles_set.update(staff_member['roles'])
            for shift in instance['shifts']:
                all_roles_set.add(shift['role'])
        
        gnn_all_roles = sorted(list(all_roles_set))
        gnn_role_to_idx = {role: i for i, role in enumerate(gnn_all_roles)}
        gnn_num_roles = len(gnn_all_roles)

        print("GNN model and resources loaded successfully.")
    except FileNotFoundError:
        print("Error: GNN model or dataset not found. Please run train_gnn.py and create_gnn_dataset.py first.")
        # Do not exit, as Transformer can still function
    except Exception as e:
        print(f"Error loading GNN resources: {e}")
        # Do not exit, as Transformer can still function

# --- Helper functions for GNN preprocessing ---
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

def preprocess_input_gnn(staff_data, shifts_data, all_roles, role_to_idx, num_roles):
    staff_id_to_idx = {sm['id']: i for i, sm in enumerate(staff_data)}
    shift_id_to_idx = {s['id']: i for i, s in enumerate(shifts_data)}

    # Node features for staff (type 0)
    staff_features = []
    for sm in staff_data:
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
    for s in shifts_data:
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

    num_staff = len(staff_data)
    num_shifts = len(shifts_data)

    max_feature_dim = max(x_staff.shape[1], x_shift.shape[1])
    
    padded_x_staff = torch.cat([x_staff, torch.zeros(num_staff, max_feature_dim - x_staff.shape[1])], dim=1)
    padded_x_shift = torch.cat([x_shift, torch.zeros(num_shifts, max_feature_dim - x_shift.shape[1])], dim=1)

    x = torch.cat([padded_x_staff, padded_x_shift], dim=0)

    edge_index_list = []
    edge_map = {} # Map (staff_id, shift_id) to edge_index_list position
    
    for staff_idx, sm in enumerate(staff_data):
        for shift_idx, s in enumerate(shifts_data):
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

# --- Feature Engineering (must match preprocess_data.py) ---
def preprocess_input(staff_data, shifts_data, role_encoder_obj):
    features = []
    shift_staff_pairs = [] # To map predictions back to shifts and staff

    # Ensure all_roles is consistent with training
    all_roles = list(role_encoder_obj.classes_)

    for shift in shifts_data:
        shift_start = datetime.fromisoformat(shift['startTime'])
        shift_end = datetime.fromisoformat(shift['endTime'])
        shift_duration = (shift_end - shift_start).total_seconds() / 3600

        for staff_member in staff_data:
            # Features for this (shift, staff) pair
            
            # 1. Role compatibility
            role_compatible = 1 if shift['role'] in staff_member['roles'] else 0
            
            # 2. Availability overlap
            available_for_shift = 0
            for avail_slot in staff_member['preferences']['availability']:
                avail_start = datetime.fromisoformat(avail_slot['startTime'])
                avail_end = datetime.fromisoformat(avail_slot['endTime'])
                if shift_start >= avail_start and shift_end <= avail_end:
                    available_for_shift = 1
                    break
            
            # 3. Desired hours
            desired_hours = staff_member['preferences']['desiredHours']
            
            # 4. Prefers consecutive days off
            prefers_consecutive = 1 if staff_member['preferences']['prefersConsecutiveDaysOff'] else 0

            # 5. Shift duration
            
            # 6. Encoded shift role
            try:
                encoded_shift_role = role_encoder_obj.transform([shift['role']])[0]
            except ValueError:
                encoded_shift_role = -1 # Indicate unseen role

            # 7. Encoded staff roles (one-hot for simplicity, or just count matches)
            encoded_staff_roles = [1 if r in staff_member['roles'] else 0 for r in all_roles]

            # Combine features
            current_features = [
                role_compatible,
                available_for_shift,
                desired_hours,
                prefers_consecutive,
                shift_duration,
                encoded_shift_role,
                *encoded_staff_roles
            ]
            features.append(current_features)
            shift_staff_pairs.append((shift['id'], staff_member['id']))
            
    return np.array(features), shift_staff_pairs

# --- Prediction Function ---
def predict_roster(staff_data, shifts_data):
    global transformer_model, transformer_scaler, transformer_role_encoder
    if transformer_model is None or transformer_scaler is None or transformer_role_encoder is None:
        load_resources() # Ensure resources are loaded if not already

    X_raw, shift_staff_pairs = preprocess_input(staff_data, shifts_data, transformer_role_encoder)
    
    if len(X_raw) == 0:
        return shifts_data # No shifts or staff, return original shifts

    # Reshape X_raw for Transformer input: (1, seq_len, input_dim)
    # Assuming a single problem instance per prediction request
    seq_len = len(shifts_data) * len(staff_data)
    input_dim = X_raw.shape[1]
    X_reshaped_for_transformer = X_raw.reshape(1, seq_len, input_dim)

    X_scaled = transformer_scaler.transform(X_reshaped_for_transformer.squeeze(0)).reshape(1, seq_len, input_dim)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = transformer_model(X_tensor) # (1, seq_len)
        probabilities = torch.sigmoid(outputs).squeeze(0).cpu().numpy() # Convert logits to probabilities

    # Create a map for assignments based on probabilities
    predicted_assignments = {} # shift_id -> staff_id
    
    # Sort predictions by probability in descending order
    sorted_predictions = sorted(zip(probabilities, shift_staff_pairs), key=lambda x: x[0], reverse=True)

    assigned_staff_to_shifts = {} # staff_id -> list of shift_ids
    
    for prob, (shift_id, staff_id) in sorted_predictions:
        if shift_id not in predicted_assignments: # If shift not yet assigned
            # Check for overlaps with already assigned shifts for this staff
            staff_assigned_shifts = assigned_staff_to_shifts.get(staff_id, [])
            
            current_shift = next(s for s in shifts_data if s['id'] == shift_id)
            
            has_overlap = False
            for assigned_s_id in staff_assigned_shifts:
                assigned_s = next(s for s in shifts_data if s['id'] == assigned_s_id)
                if (datetime.fromisoformat(current_shift['startTime']) < datetime.fromisoformat(assigned_s['endTime']) and
                    datetime.fromisoformat(current_shift['endTime']) > datetime.fromisoformat(assigned_s['startTime'])):
                    has_overlap = True
                    break
            
            if not has_overlap:
                predicted_assignments[shift_id] = staff_id
                assigned_staff_to_shifts.setdefault(staff_id, []).append(shift_id)

    # Convert predictions back to Shift[] format
    generated_roster = []
    for shift in shifts_data:
        assigned_staff_id = predicted_assignments.get(shift['id'], None)
        generated_roster.append({**shift, 'staffMemberId': assigned_staff_id})
        
    return generated_roster

# --- GNN Prediction Function ---
def predict_roster_gnn(staff_data, shifts_data):
    global gnn_model, gnn_num_node_features, gnn_all_roles, gnn_role_to_idx, gnn_num_roles

    if gnn_model is None:
        # Attempt to load resources if not already loaded (e.g., if GNN failed to load initially)
        load_resources() 
        if gnn_model is None: # If still None, GNN resources are truly unavailable
            print("GNN model not loaded. Cannot perform GNN prediction.")
            return shifts_data # Return empty roster or handle error appropriately

    data, staff_id_to_idx, shift_id_to_idx, edge_map = preprocess_input_gnn(
        staff_data, shifts_data, gnn_all_roles, gnn_role_to_idx, gnn_num_roles
    )

    if data is None:
        return shifts_data # No valid staff or shifts, return original shifts

    with torch.no_grad():
        node_embeddings = gnn_model(data.x, data.edge_index)
        logits = gnn_model.predict_edge(node_embeddings, data.edge_index)
        probabilities = torch.sigmoid(logits)
    
    # Create a map for assignments based on probabilities
    predicted_assignments = {} # shift_id -> staff_id
    
    # Sort predictions by probability in descending order
    # Need to map edge_map indices back to (staff_id, shift_id)
    reverse_edge_map = {v: k for k, v in edge_map.items()}
    
    sorted_predictions = []
    for i, prob in enumerate(probabilities):
        staff_id, shift_id = reverse_edge_map[i]
        sorted_predictions.append((prob.item(), (shift_id, staff_id)))
    
    sorted_predictions.sort(key=lambda x: x[0], reverse=True)

    assigned_staff_to_shifts = {} # staff_id -> list of shift_ids
    
    for prob, (shift_id, staff_id) in sorted_predictions:
        if shift_id not in predicted_assignments: # If shift not yet assigned
            # Check for overlaps with already assigned shifts for this staff
            staff_assigned_shifts = assigned_staff_to_shifts.get(staff_id, [])
            
            current_shift = next(s for s in shifts_data if s['id'] == shift_id)
            
            has_overlap = False
            for assigned_s_id in staff_assigned_shifts:
                assigned_s = next(s for s in shifts_data if s['id'] == assigned_s_id)
                if (datetime.fromisoformat(current_shift['startTime']) < datetime.fromisoformat(assigned_s['endTime']) and
                    datetime.fromisoformat(current_shift['endTime']) > datetime.fromisoformat(assigned_s['startTime'])):
                    has_overlap = True
                    break
            
            if not has_overlap:
                predicted_assignments[shift_id] = staff_id
                assigned_staff_to_shifts.setdefault(staff_id, []).append(shift_id)

    # Convert predictions back to Shift[] format
    generated_roster = []
    for shift in shifts_data:
        assigned_staff_id = predicted_assignments.get(shift['id'], None)
        generated_roster.append({**shift, 'staffMemberId': assigned_staff_id})
        
    return generated_roster

# --- Flask API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    staff_data = data.get('staff')
    shifts_data = data.get('shifts')
    algorithm = data.get('algorithm', 'Transformer') # Default to Transformer

    if not staff_data or not shifts_data:
        return jsonify({"error": "Missing 'staff' or 'shifts' data"}), 400

    if algorithm == 'Transformer':
        predicted_roster = predict_roster(staff_data, shifts_data)
    elif algorithm == 'GNN':
        predicted_roster = predict_roster_gnn(staff_data, shifts_data)
    else:
        return jsonify({"error": f"Unknown algorithm: {algorithm}. Available algorithms are 'Transformer', 'GNN'."}), 400
        
    return jsonify(predicted_roster)

@app.route('/algorithms', methods=['GET'])
def get_algorithms():
    available_algorithms = ["Transformer"]
    if gnn_model is not None: # Only advertise GNN if it loaded successfully
        available_algorithms.append("GNN")
    return jsonify(available_algorithms)

# --- Main Execution ---
if __name__ == '__main__':
    load_resources() # Load resources when the app starts
    print("Starting Flask API. Make sure your Python virtual environment is activated.")
    print("Run with: flask run --port 5000")
    app.run(port=5000)