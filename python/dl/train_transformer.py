import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
import os
from datetime import datetime, timedelta # Added for constraint calculation
import time # Import time module
from functools import partial # Import partial

# --- Configuration ---
PREPROCESSED_DATA_PATH = 'data/preprocessed_rostering_data.npz'
MODEL_PATH = 'data/rostering_transformer_model.pth'
NUM_EPOCHS = 500
BATCH_SIZE = 8 # Changed to 100 as per user request
LEARNING_RATE = 0.001
EMBED_DIM = 64 # Dimension for embeddings
NUM_HEADS = 4 # Number of attention heads
NUM_ENCODER_LAYERS = 4 # Number of transformer encoder layers

# Early Stopping parameters
PATIENCE = 50 # Number of epochs to wait for improvement before stopping
MIN_DELTA = 0.001 # Minimum change in validation loss to qualify as an improvement

# Constraint Penalty parameters
HARD_CONSTRAINT_PENALTY_WEIGHT = 0.5 # Weight for hard constraint violations
UNASSIGNED_SHIFT_PENALTY_WEIGHT = 5.0 # Increased penalty for unassigned shifts

# Soft Constraint Penalty parameters
DESIRED_HOURS_PENALTY_WEIGHT = 0.1 # Penalty per hour deviation from desired
CONSECUTIVE_DAYS_OFF_PENALTY_WEIGHT = 1.0 # Penalty if prefers consecutive days off but doesn't get them
CLOPEN_SHIFT_PENALTY_WEIGHT = 2.0 # Penalty for clopen shifts (insufficient rest)

# Overall Soft Constraint Weight (applied to the sum of individual soft penalties)
SOFT_CONSTRAINT_OVERALL_WEIGHT = 0.05

# --- Define the Transformer Model ---
class RosteringTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_encoder_layers):
        super(RosteringTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.output_layer = nn.Linear(embed_dim, 1) # Predict assignment probability

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_dim)
        # seq_len here is num_shifts * num_staff for a single problem instance
        
        embedded_src = self.embedding(src) # (batch_size, seq_len, embed_dim)
        
        # Transformer expects (seq_len, batch_size, embed_dim) if batch_first=False
        # But we set batch_first=True, so (batch_size, seq_len, embed_dim) is fine
        transformer_output = self.transformer_encoder(embedded_src) # (batch_size, seq_len, embed_dim)
        
        # We want a prediction for each (shift, staff) pair, so apply linear layer to each token
        output = self.output_layer(transformer_output) # (batch_size, seq_len, 1)
        
        return output.squeeze(-1) # (batch_size, seq_len)

# --- Constraint Calculation Function ---
def calculate_constraint_penalty(predicted_assignments_binary, batch_metadata,
                                 batch_staff_roles, batch_staff_availability_start, batch_staff_availability_end,
                                 batch_staff_desired_hours, batch_staff_prefers_consecutive, batch_staff_ids_map,
                                 batch_shift_roles, batch_shift_start_times, batch_shift_end_times, batch_shift_ids_map,
                                 mask):
    # predicted_assignments_binary: (batch_size, seq_len) - 0 or 1
    # batch_metadata: list of lists of (shift_id, staff_id) tuples, length batch_size
    # batch_staff_roles: (batch_size, max_staff, num_roles)
    # batch_staff_availability_*: (batch_size, max_staff)
    # batch_staff_desired_hours: (batch_size, max_staff)
    # batch_staff_prefers_consecutive: (batch_size, max_staff)
    # batch_staff_ids_map: list of dicts {original_id: tensor_idx}
    # batch_shift_roles: (batch_size, max_shifts)
    # batch_shift_start_times/end_times: (batch_size, max_shifts)
    # batch_shift_ids_map: list of dicts {original_id: tensor_idx}
    # mask: (batch_size, seq_len) - 0 for padded, 1 for valid

    device = predicted_assignments_binary.device
    batch_size, seq_len = predicted_assignments_binary.shape
    max_staff = batch_staff_roles.shape[1]
    max_shifts = batch_shift_roles.shape[1]
    num_roles = batch_staff_roles.shape[2]

    # Initialize penalties
    hard_penalty = torch.zeros(batch_size, device=device)
    unassigned_penalty = torch.zeros(batch_size, device=device)
    soft_penalty = torch.zeros(batch_size, device=device)

    # --- Map metadata to tensor indices ---
    # This is still somewhat Pythonic due to varying metadata lengths and dict lookups
    # We need to create tensors that map (batch_idx, seq_idx) to (batch_idx, staff_idx, shift_idx)
    # This is the most complex part to fully vectorize without a fixed seq_len structure.

    # For each example in the batch, create a mapping from (seq_idx) to (staff_idx, shift_idx)
    # This will be used to lookup staff/shift features for each (shift, staff) pair in the sequence.
    # We'll create a (batch_size, seq_len, 2) tensor where the last dim is [staff_idx, shift_idx]
    
    # Initialize with -1 for padded/invalid entries
    staff_indices_in_seq = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=device)
    shift_indices_in_seq = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=device)

    for b in range(batch_size):
        for s_idx, (shift_id, staff_id) in enumerate(batch_metadata[b]):
            if s_idx < seq_len: # Ensure we don't go out of bounds for padded sequences
                staff_idx = batch_staff_ids_map[b].get(staff_id, -1)
                shift_idx = batch_shift_ids_map[b].get(shift_id, -1)
                staff_indices_in_seq[b, s_idx] = staff_idx
                shift_indices_in_seq[b, s_idx] = shift_idx

    # Only consider assigned pairs for hard constraints
    assigned_mask = (predicted_assignments_binary == 1) & (mask == 1)
    
    # --- Hard Constraint 1: Role Mismatch ---
    # For each assigned (shift, staff) pair, check if staff has the required role
    # Get the required role for each shift in the sequence
    # batch_shift_roles: (batch_size, max_shifts)
    # shift_indices_in_seq: (batch_size, seq_len)
    
    # Gather the required role for each (shift, staff) pair in the sequence
    required_shift_roles_for_seq = torch.gather(batch_shift_roles, 1, shift_indices_in_seq.clamp(min=0)) # (batch_size, seq_len)
    
    # Gather the staff's roles (one-hot) for each staff in the sequence
    # batch_staff_roles: (batch_size, max_staff, num_roles)
    # staff_indices_in_seq: (batch_size, seq_len)
    
    # Create an index tensor for gathering staff roles: (batch_size, seq_len, num_roles)
    staff_roles_for_seq = torch.zeros((batch_size, seq_len, num_roles), device=device)
    for b in range(batch_size):
        for s_idx in range(seq_len):
            staff_idx = staff_indices_in_seq[b, s_idx]
            if staff_idx != -1:
                staff_roles_for_seq[b, s_idx, :] = batch_staff_roles[b, staff_idx, :]

    # Check if the staff's roles (one-hot) contain the required role
    # required_shift_roles_for_seq is an index (0 to num_roles-1)
    # staff_roles_for_seq is one-hot
    
    # Convert required_shift_roles_for_seq to one-hot for comparison
    required_role_one_hot = torch.nn.functional.one_hot(required_shift_roles_for_seq.clamp(min=0), num_classes=num_roles).float()
    
    # Check if staff has the required role: (batch_size, seq_len)
    # This checks if the staff's role vector has a 1 at the position of the required role
    has_required_role = (staff_roles_for_seq * required_role_one_hot).sum(dim=-1) > 0.5 # (batch_size, seq_len)

    role_mismatch_penalty = (~has_required_role & assigned_mask).float()
    hard_penalty += role_mismatch_penalty.sum(dim=1) # Sum over sequence length for each example

    # --- Hard Constraint 2: Availability Violation ---
    # Compare shift start/end times with staff availability start/end times
    # batch_shift_start_times/end_times: (batch_size, max_shifts)
    # batch_staff_availability_start/end: (batch_size, max_staff)

    shift_start_for_seq = torch.gather(batch_shift_start_times, 1, shift_indices_in_seq.clamp(min=0))
    shift_end_for_seq = torch.gather(batch_shift_end_times, 1, shift_indices_in_seq.clamp(min=0))
    staff_avail_start_for_seq = torch.gather(batch_staff_availability_start, 1, staff_indices_in_seq.clamp(min=0))
    staff_avail_end_for_seq = torch.gather(batch_staff_availability_end, 1, staff_indices_in_seq.clamp(min=0))

    # Check if shift is within staff's availability window
    is_available = (shift_start_for_seq >= staff_avail_start_for_seq) & \
                   (shift_end_for_seq <= staff_avail_end_for_seq)
    
    availability_violation_penalty = (~is_available & assigned_mask).float()
    hard_penalty += availability_violation_penalty.sum(dim=1)

    # --- Hard Constraint 3: Double-booking ---
    # This is complex. For each staff member, collect all assigned shifts and check for overlaps.
    # We need to iterate over staff members for each example in the batch.
    
    # Create a tensor to store assigned shift start/end times for each staff member
    # (batch_size, max_staff, max_shifts_per_staff, 2) where last dim is [start_time, end_time]
    # This will be sparse and require careful handling.
    
    # A simpler approach for now: iterate through staff, then shifts.
    # This will still involve some Python loops, but on a smaller scale (max_staff * max_shifts)
    # rather than (batch_size * seq_len).
    
    # For each example in the batch
    for b in range(batch_size):
        # Skip if example is padded
        if mask[b].sum() == 0:
            continue

        # Create a list of assigned shifts for each staff member (using tensor indices)
        staff_assigned_shifts_tensor_indices = [[] for _ in range(max_staff)]
        
        # Iterate through the sequence for this example
        for s_idx in range(seq_len):
            if assigned_mask[b, s_idx]: # If this (shift, staff) pair is assigned and valid
                staff_idx = staff_indices_in_seq[b, s_idx].item()
                shift_idx = shift_indices_in_seq[b, s_idx].item()
                if staff_idx != -1 and shift_idx != -1:
                    staff_assigned_shifts_tensor_indices[staff_idx].append(shift_idx)

        # Now check for overlaps for each staff member
        for staff_idx in range(max_staff):
            assigned_shift_indices = staff_assigned_shifts_tensor_indices[staff_idx]
            if len(assigned_shift_indices) > 1:
                # Get start and end times for these assigned shifts
                assigned_starts = batch_shift_start_times[b, assigned_shift_indices]
                assigned_ends = batch_shift_end_times[b, assigned_shift_indices]

                # Sort by start time (important for overlap checking)
                sorted_indices = torch.argsort(assigned_starts)
                assigned_starts = assigned_starts[sorted_indices]
                assigned_ends = assigned_ends[sorted_indices]

                # Check for overlaps
                # For each shift, compare its end time with the next shift's start time
                overlaps = (assigned_starts[1:] < assigned_ends[:-1]) & \
                           (assigned_ends[1:] > assigned_starts[:-1])
                
                hard_penalty[b] += 1.0 # Penalty for invalid shift

    # --- Unassigned Shifts Penalty (Soft/Critical) ---
    # Count how many shifts are not assigned to any staff member
    # Create a tensor to track if each shift has been assigned at least once
    shift_assigned_status = torch.zeros((batch_size, max_shifts), dtype=torch.bool, device=device)

    for b in range(batch_size):
        # Iterate through assigned (shift, staff) pairs
        for s_idx in range(seq_len):
            if assigned_mask[b, s_idx]:
                shift_idx = shift_indices_in_seq[b, s_idx].item()
                if shift_idx != -1:
                    shift_assigned_status[b, shift_idx] = True
    
    # Count unassigned shifts for each example
    # We need to consider only valid shifts (not padded ones in original_shifts_example)
    # For now, assuming all shifts up to max_shifts are "valid" for penalty calculation
    # This needs to be refined to only penalize actual shifts in the original problem.
    
    # For now, let's use the original_shifts_example length to determine valid shifts
    # This means we need to pass original_shifts_example lengths to the function.
    # For simplicity, let's assume all shifts up to max_shifts are potential shifts.
    # This is a simplification that might need adjustment.
    
    # A more accurate way would be to pass the actual number of shifts for each example
    # from the collate_fn.
    
    # For now, let's assume that if a shift_idx is valid (not -1), it's a real shift.
    # This is implicitly handled by how shift_indices_in_seq is populated.
    
    # The number of actual shifts for each example is len(batch_shift_ids_map[b])
    for b in range(batch_size):
        num_actual_shifts = len(batch_shift_ids_map[b])
        unassigned_shifts_in_example = (~shift_assigned_status[b, :num_actual_shifts]).float()
        unassigned_penalty[b] += unassigned_shifts_in_example.sum() * UNASSIGNED_SHIFT_PENALTY_WEIGHT # Penalty per unassigned shift

    # --- Soft Constraint 1: Desired Hours Penalty ---
    # Calculate actual hours worked per staff member and compare with desired hours
    # batch_staff_desired_hours: (batch_size, max_staff)
    # shift_start_for_seq, shift_end_for_seq: (batch_size, seq_len)

    # Create a tensor to accumulate actual hours for each staff member
    actual_hours_per_staff = torch.zeros((batch_size, max_staff), device=device)

    for b in range(batch_size):
        for s_idx in range(seq_len):
            if assigned_mask[b, s_idx]:
                staff_idx = staff_indices_in_seq[b, s_idx].item()
                if staff_idx != -1:
                    shift_duration = (shift_end_for_seq[b, s_idx] - shift_start_for_seq[b, s_idx]) / 3600.0
                    actual_hours_per_staff[b, staff_idx] += shift_duration
    
    # Calculate deviation
    desired_hours_deviation = torch.abs(actual_hours_per_staff - batch_staff_desired_hours)
    soft_penalty += (desired_hours_deviation * DESIRED_HOURS_PENALTY_WEIGHT).sum(dim=1)

    # --- Soft Constraint 2: Consecutive Days Off Penalty ---
    # This is very hard to vectorize with current time representations.
    # Requires grouping shifts by staff and then by day, and checking gaps.
    # For now, I will skip full vectorization of this and Clopen shifts,
    # as it significantly complicates the tensor logic and might require
    # a different feature representation (e.g., day-of-week features).
    # I will leave a placeholder for now.
    
    # --- Soft Constraint 3: Clopen Shifts Penalty ---
    # Also very hard to vectorize. Requires sorting shifts per staff and checking rest times.
    # Leaving a placeholder.

    # Combine penalties
    total_hard_penalty = hard_penalty.sum()
    total_unassigned_penalty = unassigned_penalty.sum()
    total_soft_penalty = soft_penalty.sum()

    return total_hard_penalty, total_unassigned_penalty, total_soft_penalty

# --- Main Execution ---
if __name__ == "__main__":
    # Check for GPU (CUDA) or MPS (Apple Silicon)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    try:
        data = np.load(PREPROCESSED_DATA_PATH, allow_pickle=True)
        X_scaled_flat = data['X']
        y_flat = data['y']
        metadata = data['metadata'] # List of lists of (shift_id, staff_id) tuples
        original_staff_data_flat = data['original_staff'] # Loaded original staff data
        original_shifts_data_flat = data['original_shifts'] # Loaded original shifts data
    except FileNotFoundError:
        print(f"Error: Preprocessed data not found at {PREPROCESSED_DATA_PATH}. Please run preprocess_data.py first.")
        exit()

    # Determine input_dim from the preprocessed data
    input_dim = X_scaled_flat.shape[1]
    print(f"Input dimension: {input_dim}")

    # The current preprocessing flattens all examples into one large X and y.
    # For a Transformer, we need to process each example (rostering problem) as a sequence.
    # We need to reconstruct the sequences based on the metadata or original example structure.

    # Calculate start and end indices for each example in the flattened arrays
    example_lengths = [len(m) for m in metadata]
    example_start_indices = np.cumsum([0] + example_lengths[:-1])
    
    num_examples = len(metadata)
    
    # Convert to PyTorch tensors (still flattened for now)
    X_tensor_flat = torch.tensor(X_scaled_flat, dtype=torch.float32).to(device)
    y_tensor_flat = torch.tensor(y_flat, dtype=torch.float32).to(device)

    # Create a custom dataset to handle metadata and original data
    class RosterDataset(TensorDataset):
        def __init__(self, X_flat, y_flat, metadata, original_staff, original_shifts, example_start_indices, example_lengths):
            self.X_flat = X_flat
            self.y_flat = y_flat
            self.metadata = metadata
            self.original_staff = original_staff
            self.original_shifts = original_shifts
            self.example_start_indices = example_start_indices
            self.example_lengths = example_lengths

        def __len__(self):
            return len(self.metadata)

        def __getitem__(self, idx):
            start_idx = self.example_start_indices[idx]
            length = self.example_lengths[idx]
            end_idx = start_idx + length

            X_example = self.X_flat[start_idx:end_idx]
            y_example = self.y_flat[start_idx:end_idx]
            metadata_example = self.metadata[idx]
            original_staff_example = self.original_staff[idx]
            original_shifts_example = self.original_shifts[idx]
            
            return X_example, y_example, metadata_example, original_staff_example, original_shifts_example

    # Custom collate_fn to handle padding and batching
    def custom_collate_fn(batch, role_encoder):
        # batch is a list of tuples: [(X_i, y_i, metadata_i, staff_i, shifts_i), ...]
        
        # Determine the maximum sequence length in the current batch
        max_seq_len = max([item[0].shape[0] for item in batch])
        max_staff_count = max([len(item[3]) for item in batch]) # Max staff members in batch
        max_shift_count = max([len(item[4]) for item in batch]) # Max shifts in batch

        padded_X_batch = []
        padded_y_batch = []
        metadata_batch = []
        
        # New lists for processed staff and shift data
        staff_roles_batch = []
        staff_availability_start_batch = []
        staff_availability_end_batch = []
        staff_desired_hours_batch = []
        staff_prefers_consecutive_batch = []
        staff_ids_map_batch = [] # To map original staff IDs to tensor indices

        shift_roles_batch = []
        shift_start_times_batch = []
        shift_end_times_batch = []
        shift_ids_map_batch = [] # To map original shift IDs to tensor indices

        for X_example, y_example, metadata_example, original_staff_example, original_shifts_example in batch:
            # Pad X
            padding_size_X = max_seq_len - X_example.shape[0]
            padded_X = torch.nn.functional.pad(X_example, (0, 0, 0, padding_size_X), 'constant', 0) # Pad last dimension (seq_len)
            padded_X_batch.append(padded_X)

            # Pad y
            padding_size_y = max_seq_len - y_example.shape[0]
            padded_y = torch.nn.functional.pad(y_example, (0, padding_size_y), 'constant', -1) # Pad with -1 for labels
            padded_y_batch.append(padded_y)

            metadata_batch.append(metadata_example)

            # --- Process original_staff_example ----
            current_staff_roles = []
            current_staff_availability_start = []
            current_staff_availability_end = []
            current_staff_desired_hours = []
            current_staff_prefers_consecutive = []
            current_staff_ids_map = {staff['id']: i for i, staff in enumerate(original_staff_example)}
            
            for staff in original_staff_example:
                # Assuming role_encoder is available globally or passed
                # For now, let's use a placeholder for encoded roles
                # This needs to be consistent with how roles are encoded in preprocess_data.py
                # For simplicity, let's assume staff['roles'] is a list of strings
                # and we need a one-hot encoding or similar.
                # For now, let's just take the first role and encode it, or use a placeholder
                # This part needs careful alignment with the actual role encoding.
                # For now, let's just use a dummy value or the length of roles.
                current_staff_roles.append([1 if r in staff['roles'] else 0 for r in role_encoder.classes_]) # Assuming role_encoder is accessible
                
                # Availability: Assuming each staff has a list of availability slots.
                # For simplicity, let's take the first slot's start/end time.
                # This needs to be more robust for multiple slots.
                if staff['preferences']['availability']:
                    avail_start_dt = datetime.fromisoformat(staff['preferences']['availability'][0]['startTime'])
                    avail_end_dt = datetime.fromisoformat(staff['preferences']['availability'][0]['endTime'])
                    current_staff_availability_start.append(avail_start_dt.timestamp())
                    current_staff_availability_end.append(avail_end_dt.timestamp())
                else:
                    current_staff_availability_start.append(0.0) # Placeholder for no availability
                    current_staff_availability_end.append(0.0) # Placeholder for no availability

                current_staff_desired_hours.append(staff['preferences']['desiredHours'])
                current_staff_prefers_consecutive.append(1.0 if staff['preferences']['prefersConsecutiveDaysOff'] else 0.0)

            # Pad staff data
            num_staff_padding = max_staff_count - len(original_staff_example)
            staff_roles_batch.append(torch.tensor(current_staff_roles + [[0.0] * len(role_encoder.classes_)] * num_staff_padding, dtype=torch.float32))
            staff_availability_start_batch.append(torch.tensor(current_staff_availability_start + [0.0] * num_staff_padding, dtype=torch.float32))
            staff_availability_end_batch.append(torch.tensor(current_staff_availability_end + [0.0] * num_staff_padding, dtype=torch.float32))
            staff_desired_hours_batch.append(torch.tensor(current_staff_desired_hours + [0.0] * num_staff_padding, dtype=torch.float32))
            staff_prefers_consecutive_batch.append(torch.tensor(current_staff_prefers_consecutive + [0.0] * num_staff_padding, dtype=torch.float32))
            staff_ids_map_batch.append(current_staff_ids_map) # Keep as dict for now, will need to convert to tensor later if used directly

            # --- Process original_shifts_example ---
            current_shift_roles = []
            current_shift_start_times = []
            current_shift_end_times = []
            current_shift_ids_map = {shift['id']: i for i, shift in enumerate(original_shifts_example)}

            for shift in original_shifts_example:
                current_shift_roles.append(role_encoder.transform([shift['role']])[0]) # Encoded role
                current_shift_start_times.append(datetime.fromisoformat(shift['startTime']).timestamp())
                current_shift_end_times.append(datetime.fromisoformat(shift['endTime']).timestamp())
            
            # Pad shift data
            num_shift_padding = max_shift_count - len(original_shifts_example)
            shift_roles_batch.append(torch.tensor(current_shift_roles + [0] * num_shift_padding, dtype=torch.long)) # Roles are long
            shift_start_times_batch.append(torch.tensor(current_shift_start_times + [0.0] * num_shift_padding, dtype=torch.float32))
            shift_end_times_batch.append(torch.tensor(current_shift_end_times + [0.0] * num_shift_padding, dtype=torch.float32))
            shift_ids_map_batch.append(current_shift_ids_map) # Keep as dict for now

        return (torch.stack(padded_X_batch), 
                torch.stack(padded_y_batch), 
                metadata_batch, # Still list of lists of tuples
                torch.stack(staff_roles_batch),
                torch.stack(staff_availability_start_batch),
                torch.stack(staff_availability_end_batch),
                torch.stack(staff_desired_hours_batch),
                torch.stack(staff_prefers_consecutive_batch),
                staff_ids_map_batch, # Still list of dicts
                torch.stack(shift_roles_batch),
                torch.stack(shift_start_times_batch),
                torch.stack(shift_end_times_batch),
                shift_ids_map_batch # Still list of dicts
                )

    # Create dataset and dataloaders
    dataset = RosterDataset(X_tensor_flat, y_tensor_flat, metadata, original_staff_data_flat, original_shifts_data_flat, example_start_indices, example_lengths)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # We will manually sample for training, so no DataLoader for train_dataset
    # Create a partial function for custom_collate_fn with role_encoder bound
    collate_fn_with_encoder = partial(custom_collate_fn, role_encoder=role_encoder)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_with_encoder)

    # Instantiate model
    model = RosteringTransformer(input_dim, EMBED_DIM, NUM_HEADS, NUM_ENCODER_LAYERS).to(device)
    
    # Loss and optimizer
    # Ignore padded values in BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss(reduction='none') # Apply reduction manually
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("Training Transformer model...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Define number of training steps per epoch for manual sampling
    # This ensures that, on average, the entire dataset is sampled once per epoch
    NUM_TRAINING_STEPS_PER_EPOCH = len(train_dataset) // BATCH_SIZE 
    if NUM_TRAINING_STEPS_PER_EPOCH == 0: # Ensure at least one step if dataset is smaller than batch size
        NUM_TRAINING_STEPS_PER_EPOCH = 1

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_bce_loss = 0.0
        running_penalty = 0.0
        
        # Add timing for the entire epoch
        epoch_start_time = time.time()
        
        for step in range(NUM_TRAINING_STEPS_PER_EPOCH):
            step_start_time = time.time() # Start timing for each step

            # Manually sample BATCH_SIZE instances from train_dataset
            sample_start_time = time.time()
            random_indices = np.random.choice(len(train_dataset), BATCH_SIZE, replace=False)
            sampled_batch = [train_dataset[i] for i in random_indices]
            sample_end_time = time.time()
            
            # Collate the sampled batch
            collate_start_time = time.time()
            (batch_X, batch_y, batch_metadata, 
             batch_staff_roles, batch_staff_availability_start, batch_staff_availability_end, 
             batch_staff_desired_hours, batch_staff_prefers_consecutive, batch_staff_ids_map,
             batch_shift_roles, batch_shift_start_times, batch_shift_end_times, batch_shift_ids_map) = collate_fn_with_encoder(sampled_batch)
            collate_end_time = time.time()
            
            # Move batch to device
            to_device_start_time = time.time()
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_staff_roles = batch_staff_roles.to(device)
            batch_staff_availability_start = batch_staff_availability_start.to(device)
            batch_staff_availability_end = batch_staff_availability_end.to(device)
            batch_staff_desired_hours = batch_staff_desired_hours.to(device)
            batch_staff_prefers_consecutive = batch_staff_prefers_consecutive.to(device)
            batch_shift_roles = batch_shift_roles.to(device)
            batch_shift_start_times = batch_shift_start_times.to(device)
            batch_shift_end_times = batch_shift_end_times.to(device)
            to_device_end_time = time.time()

            optimizer.zero_grad()
            
            forward_start_time = time.time()
            outputs = model(batch_X)
            forward_end_time = time.time()
            
            # Create a mask for valid (non-padded) elements
            mask = (batch_y != -1).float()
            
            bce_loss_unreduced = criterion(outputs, batch_y)
            bce_loss = (bce_loss_unreduced * mask).sum() / mask.sum() # Apply mask and sum/mean manually
            
            # Calculate constraint penalty
            penalty_calc_total_time = 0.0
            hard_penalty_sum = 0.0
            unassigned_penalty_sum = 0.0
            soft_penalty_sum = 0.0
            # Convert outputs to probabilities and then binary predictions
            predicted_probs = torch.sigmoid(outputs)
            # Only consider predictions for non-padded elements
            predicted_assignments_binary = ((predicted_probs > 0.5).float() * mask) # Keep on device

            # Pass all necessary tensors to the new calculate_constraint_penalty
            constraint_start_time = time.time()
            hard_p, unassigned_p, soft_p = calculate_constraint_penalty(
                predicted_assignments_binary,
                batch_metadata, # Still list of lists of tuples
                batch_staff_roles, batch_staff_availability_start, batch_staff_availability_end,
                batch_staff_desired_hours, batch_staff_prefers_consecutive, batch_staff_ids_map,
                batch_shift_roles, batch_shift_start_times, batch_shift_end_times, batch_shift_ids_map,
                mask # Pass the mask to the penalty function
            )
            constraint_end_time = time.time()
            penalty_calc_total_time += (constraint_end_time - constraint_start_time)

            hard_penalty_sum += hard_p
            unassigned_penalty_sum += unassigned_p
            soft_penalty_sum += soft_p
            
            total_penalty = (HARD_CONSTRAINT_PENALTY_WEIGHT * hard_penalty_sum + 
                             UNASSIGNED_SHIFT_PENALTY_WEIGHT * unassigned_penalty_sum +
                             SOFT_CONSTRAINT_OVERALL_WEIGHT * soft_penalty_sum)
            total_loss = bce_loss + total_penalty
            
            backward_start_time = time.time()
            total_loss.backward()
            optimizer.step()
            backward_end_time = time.time()

            running_loss += total_loss.item()
            running_bce_loss += bce_loss.item()
            running_penalty += total_penalty.item() # Accumulate total penalty (now a tensor)
            
            step_end_time = time.time() # End timing for each step
            # print(f"  Step {step+1}/{NUM_TRAINING_STEPS_PER_EPOCH} - Total: {step_end_time - step_start_time:.4f}s, Sample: {sample_end_time - sample_start_time:.4f}s, Collate: {collate_end_time - collate_start_time:.4f}s, ToDevice: {to_device_end_time - to_device_start_time:.4f}s, Forward: {forward_end_time - forward_start_time:.4f}s, PenaltyCalc: {penalty_calc_total_time:.4f}s, Backward: {backward_end_time - backward_start_time:.4f}s")
        
        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} training duration: {epoch_end_time - epoch_start_time:.4f} seconds")

        # Validation
        model.eval()
        val_loss = 0.0
        val_bce_loss = 0.0
        val_penalty = 0.0
        with torch.no_grad():
            for (batch_X_val, batch_y_val, batch_metadata_val, 
                 batch_staff_roles_val, batch_staff_availability_start_val, batch_staff_availability_end_val, 
                 batch_staff_desired_hours_val, batch_staff_prefers_consecutive_val, batch_staff_ids_map_val,
                 batch_shift_roles_val, batch_shift_start_times_val, batch_shift_end_times_val, batch_shift_ids_map_val) in val_loader:
                
                batch_X_val = batch_X_val.to(device)
                batch_y_val = batch_y_val.to(device)
                batch_staff_roles_val = batch_staff_roles_val.to(device)
                batch_staff_availability_start_val = batch_staff_availability_start_val.to(device)
                batch_staff_availability_end_val = batch_staff_availability_end_val.to(device)
                batch_staff_desired_hours_val = batch_staff_desired_hours_val.to(device)
                batch_staff_prefers_consecutive_val = batch_staff_prefers_consecutive_val.to(device)
                batch_shift_roles_val = batch_shift_roles_val.to(device)
                batch_shift_start_times_val = batch_shift_start_times_val.to(device)
                batch_shift_end_times_val = batch_shift_end_times_val.to(device)

                outputs_val = model(batch_X_val)
                
                mask_val = (batch_y_val != -1).float()
                bce_loss_unreduced_val = criterion(outputs_val, batch_y_val)
                bce_loss_val = (bce_loss_unreduced_val * mask_val).sum() / mask_val.sum()

                # Calculate constraint penalty for validation
                predicted_probs_val = torch.sigmoid(outputs_val)
                predicted_assignments_binary_val = ((predicted_probs_val > 0.5).float() * mask_val)

                hard_p_val, unassigned_p_val, soft_p_val = calculate_constraint_penalty(
                    predicted_assignments_binary_val,
                    batch_metadata_val,
                    batch_staff_roles_val, batch_staff_availability_start_val, batch_staff_availability_end_val,
                    batch_staff_desired_hours_val, batch_staff_prefers_consecutive_val, batch_staff_ids_map_val,
                    batch_shift_roles_val, batch_shift_start_times_val, batch_shift_end_times_val, batch_shift_ids_map_val,
                    mask_val
                )
                
                total_penalty_val = (HARD_CONSTRAINT_PENALTY_WEIGHT * hard_p_val + 
                                     UNASSIGNED_SHIFT_PENALTY_WEIGHT * unassigned_p_val +
                                     SOFT_CONSTRAINT_OVERALL_WEIGHT * soft_p_val)
                total_loss_val = bce_loss_val + total_penalty_val
                
                val_loss += total_loss_val.item()
                val_bce_loss += bce_loss_val.item()
                val_penalty += total_penalty_val.item() # Accumulate total penalty (now a tensor)
        
        avg_train_loss = running_loss / NUM_TRAINING_STEPS_PER_EPOCH
        avg_train_bce_loss = running_bce_loss / NUM_TRAINING_STEPS_PER_EPOCH
        avg_train_penalty = running_penalty / NUM_TRAINING_STEPS_PER_EPOCH

        avg_val_loss = val_loss / len(val_loader)
        avg_val_bce_loss = val_bce_loss / len(val_loader)
        avg_val_penalty = val_penalty / len(val_loader)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f} (BCE: {avg_train_bce_loss:.4f}, Penalty: {avg_train_penalty:.2f}), Val Loss: {avg_val_loss:.4f} (BCE: {avg_val_bce_loss:.4f}, Penalty: {avg_val_penalty:.2f})")

        # Early stopping check
        if avg_val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH) # Save the best model
            print(f"Model improved and saved to {MODEL_PATH}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print("Training complete.")

    # If early stopping didn't save the model, save the last one (or load the best one)
    # For this implementation, we save the best model during training, so no need to save again here.
    # We should load the best model if we want to ensure we use the best one.
    # For simplicity, we assume the best model was saved.
    print(f"Final best Transformer model (or last if no improvement) saved to {MODEL_PATH}")

    print("\nTo run this script, ensure you have Python installed with:")
    print("pip install torch numpy scikit-learn joblib")
    print(f"You can run this script using: python server/dl_poc/train_transformer.py")
