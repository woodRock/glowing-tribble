import json
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# --- Configuration ---
DATASET_PATH = 'data/rostering_dataset.json'
PREPROCESSED_DATA_PATH = 'data/preprocessed_rostering_data.npz'
SCALER_PATH = 'data/transformer_scaler.joblib'
ROLE_ENCODER_PATH = 'data/transformer_role_encoder.joblib'

# --- Feature Engineering ---
def extract_features_for_transformer(example, role_encoder_obj):
    features_list = []
    labels_list = []
    metadata_list = [] # To store (example_idx, shift_id, staff_id)

    staff_map = {s['id']: s for s in example['staff']}
    shifts_map = {s['id']: s for s in example['shifts']} # Added for easier lookup

    # Ensure all_roles is consistent with training
    all_roles = list(role_encoder_obj.classes_)

    # --- Pre-calculate problem-level and shift-level features ---
    problem_num_staff = len(example['staff'])
    problem_num_shifts = len(example['shifts'])
    problem_staff_to_shift_ratio = problem_num_staff / (problem_num_shifts + 1e-6) # Avoid division by zero

    # Calculate initial desired hours deviation for each staff member
    staff_initial_desired_hours_deviation = {}
    for staff_member in example['staff']:
        staff_initial_desired_hours_deviation[staff_member['id']] = staff_member['preferences']['desiredHours']

    # Calculate num_staff_with_required_role and num_available_staff_for_shift for each shift
    shift_stats = {}
    for shift in example['shifts']:
        num_staff_with_required_role = 0
        num_available_staff_for_shift = 0
        
        shift_start_dt = datetime.fromisoformat(shift['startTime'])
        shift_end_dt = datetime.fromisoformat(shift['endTime'])

        for staff_member in example['staff']:
            if shift['role'] in staff_member['roles']:
                num_staff_with_required_role += 1
                
                is_available = False
                for avail_slot in staff_member['preferences']['availability']:
                    avail_start = datetime.fromisoformat(avail_slot['startTime'])
                    avail_end = datetime.fromisoformat(avail_slot['endTime'])
                    if shift_start_dt >= avail_start and shift_end_dt <= avail_end:
                        is_available = True
                        break
                if is_available:
                    num_available_staff_for_shift += 1
        
        shift_stats[shift['id']] = {
            'num_staff_with_required_role': num_staff_with_required_role,
            'num_available_staff_for_shift': num_available_staff_for_shift
        }


    for shift in example['shifts']:
        shift_start = datetime.fromisoformat(shift['startTime'])
        shift_end = datetime.fromisoformat(shift['endTime'])
        shift_duration = (shift_end - shift_start).total_seconds() / 3600

        # Temporal Features
        shift_start_hour = shift_start.hour
        shift_day_of_week = shift_start.weekday() # Monday is 0, Sunday is 6
        shift_is_weekend = 1 if shift_day_of_week >= 5 else 0 # Saturday and Sunday

        for staff_member in example['staff']:
            # Features for this (shift, staff) pair
            
            # 1. Role compatibility (already present)
            role_compatible = 1 if shift['role'] in staff_member['roles'] else 0
            
            # 2. Availability overlap (already present)
            available_for_shift = 0
            for avail_slot in staff_member['preferences']['availability']:
                avail_start = datetime.fromisoformat(avail_slot['startTime'])
                avail_end = datetime.fromisoformat(avail_slot['endTime'])
                if shift_start >= avail_start and shift_end <= avail_end:
                    available_for_shift = 1
                    break
            
            # 3. Desired hours (already present)
            desired_hours = staff_member['preferences']['desiredHours']
            
            # 4. Prefers consecutive days off (already present)
            prefers_consecutive = 1 if staff_member['preferences']['prefersConsecutiveDaysOff'] else 0

            # 5. Shift duration (already present)
            
            # 6. Encoded shift role (already present)
            try:
                encoded_shift_role = role_encoder_obj.transform([shift['role']])[0]
            except ValueError:
                encoded_shift_role = -1 # Indicate unseen role

            # 7. Encoded staff roles (one-hot for simplicity, or just count matches) (already present)
            encoded_staff_roles = [1 if r in staff_member['roles'] else 0 for r in all_roles]

            # New Advanced Features
            staff_desired_hours_deviation_initial = staff_initial_desired_hours_deviation[staff_member['id']]
            num_staff_with_required_role = shift_stats[shift['id']]['num_staff_with_required_role']
            num_available_staff_for_shift = shift_stats[shift['id']]['num_available_staff_for_shift']

            # Combine all features
            current_features = [
                role_compatible,
                available_for_shift,
                desired_hours,
                prefers_consecutive,
                shift_duration,
                encoded_shift_role,
                *encoded_staff_roles, # Unpack list of encoded roles
                shift_start_hour,
                shift_day_of_week,
                shift_is_weekend,
                staff_desired_hours_deviation_initial,
                num_staff_with_required_role,
                num_available_staff_for_shift,
                problem_num_staff,
                problem_num_shifts,
                problem_staff_to_shift_ratio
            ]
            features_list.append(current_features)

            # Label: Is this staff member assigned to this shift in the solution?
            is_assigned = 0
            for sol_shift in example['solution_roster']:
                if sol_shift['id'] == shift['id'] and sol_shift['staffMemberId'] == staff_member['id']:
                    is_assigned = 1
                    break
            labels_list.append(is_assigned)
            metadata_list.append((shift['id'], staff_member['id']))
            
    return np.array(features_list), np.array(labels_list), metadata_list

# --- Main Execution ---
if __name__ == "__main__":
    try:
        with open(DATASET_PATH, 'r') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATASET_PATH}. Please run generate_data.py first.")
        exit()

    all_features_combined = []
    all_labels_combined = []
    all_metadata_combined = []
    all_original_staff_data = [] # New list to store original staff data
    all_original_shifts_data = [] # New list to store original shifts data
    
    # Initialize role encoder
    all_possible_roles_in_dataset = set()
    for example in dataset:
        for s in example['staff']:
            all_possible_roles_in_dataset.update(s['roles'])
        for shift in example['shifts']:
            all_possible_roles_in_dataset.add(shift['role'])
    
    role_encoder = LabelEncoder()
    role_encoder.fit(list(all_possible_roles_in_dataset))
    joblib.dump(role_encoder, ROLE_ENCODER_PATH)
    print(f"Role Encoder saved to {ROLE_ENCODER_PATH}")

    print("Preprocessing data for Transformer...")
    for i, example in enumerate(dataset):
        features, labels, metadata = extract_features_for_transformer(example, role_encoder)
        all_features_combined.append(features)
        all_labels_combined.append(labels)
        all_metadata_combined.append(metadata) # Keep metadata per example
        all_original_staff_data.append(example['staff']) # Store original staff data
        all_original_shifts_data.append(example['shifts']) # Store original shifts data

    # Stack all features and labels
    X_combined = np.vstack(all_features_combined)
    y_combined = np.concatenate(all_labels_combined)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    # Save preprocessed data
    np.savez(PREPROCESSED_DATA_PATH, 
            X=X_scaled, 
            y=y_combined, 
            metadata=np.array(all_metadata_combined, dtype=object),
            original_staff=np.array(all_original_staff_data, dtype=object), # Save original staff data
            original_shifts=np.array(all_original_shifts_data, dtype=object)) # Save original shifts data
    print(f"Preprocessed data saved to {PREPROCESSED_DATA_PATH}")

    print("\nTo run this script, ensure you have Python installed with:")
    print("pip install numpy scikit-learn joblib")
    print(f"You can run this script using: python server/dl_poc/preprocess_data.py")
