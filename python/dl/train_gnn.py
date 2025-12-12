import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import os
import random # Import random for shuffling
from datetime import datetime

# --- Configuration ---
DATASET_PATH = 'server/dl_poc/data/gnn_rostering_dataset.pt'
MODEL_SAVE_PATH = 'server/dl_poc/data/rostering_gnn_model.pth'
TRAIN_SPLIT_RATIO = 0.8 # 80% for training, 20% for validation

# Penalty weights for hard constraints
DOUBLE_BOOKING_PENALTY_WEIGHT = 10.0
MIN_STAFF_PENALTY_WEIGHT = 5.0

# --- 1. GNN Model Architecture ---
class RosterGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels) # Added a third GCN layer
        # Output layer for edge prediction (binary classification)
        self.lin = torch.nn.Linear(2 * hidden_channels, 1)
        self.dropout = torch.nn.Dropout(0.5) # Added dropout

    def forward(self, x, edge_index):
        # Node embeddings
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
        # Given node embeddings and a list of potential edges, predict if they exist
        row, col = edge_index_to_predict
        
        # Concatenate features of source and target nodes
        combined_features = torch.cat([node_embeddings[row], node_embeddings[col]], dim=-1)
        
        # Pass through a linear layer for logits
        return self.lin(combined_features)

# --- Helper function for time overlap ---
def check_overlap(start1, end1, start2, end2):
    return max(start1, start2) < min(end1, end2)

# --- Constraint Penalty Calculation ---
def calculate_constraint_penalty(logits, data, double_booking_weight, min_staff_weight):
    device = logits.device
    probabilities = torch.sigmoid(logits).squeeze(-1) # Shape: (num_edges,)
    
    total_penalty = torch.tensor(0.0, device=device)

    # --- Double-booking Penalty ---
    # Iterate through each staff member
    for staff_node_idx in range(data.num_staff_nodes):
        # Find all edges connected to this staff member
        # edge_index[0] contains staff node indices, edge_index[1] contains shift node indices
        staff_edges_mask = (data.edge_index[0] == staff_node_idx)
        
        # Get the indices of these edges in the original logits/probabilities tensor
        edge_indices_for_staff = torch.nonzero(staff_edges_mask).squeeze(-1)
        
        if len(edge_indices_for_staff) > 1:
            # Get the probabilities and corresponding shift node indices for these edges
            staff_probs = probabilities[edge_indices_for_staff]
            shift_node_indices = data.edge_index[1, edge_indices_for_staff]
            
            # Map shift node indices back to original shift IDs and then to shift details
            # Shift node indices are offset by data.num_staff_nodes
            original_shift_indices = shift_node_indices - data.num_staff_nodes
            
            # Get shift start and end times for these shifts
            shift_times = []
            for original_shift_idx in original_shift_indices:
                shift_data = data.original_shifts_data[original_shift_idx.item()]
                shift_times.append({
                    'start': datetime.fromisoformat(shift_data['startTime']),
                    'end': datetime.fromisoformat(shift_data['endTime'])
                })
            
            # Check for overlaps among predicted shifts for this staff member
            for i in range(len(staff_probs)):
                for j in range(i + 1, len(staff_probs)):
                    prob_i = staff_probs[i]
                    prob_j = staff_probs[j]
                    
                    shift_i_start = shift_times[i]['start']
                    shift_i_end = shift_times[i]['end']
                    shift_j_start = shift_times[j]['start']
                    shift_j_end = shift_times[j]['end']

                    if check_overlap(shift_i_start, shift_i_end, shift_j_start, shift_j_end):
                        # Apply penalty proportional to the product of probabilities
                        # This encourages the model to push down at least one of the overlapping assignments
                        total_penalty += double_booking_weight * prob_i * prob_j

    # --- Minimum Staff for Role Penalty ---
    # Iterate through each shift
    for shift_node_idx_offset in range(data.num_shift_nodes):
        # Convert offset shift node index to actual node index in the graph
        shift_node_idx = data.num_staff_nodes + shift_node_idx_offset
        
        # Find all edges connected to this shift
        shift_edges_mask = (data.edge_index[1] == shift_node_idx)
        edge_indices_for_shift = torch.nonzero(shift_edges_mask).squeeze(-1)
        
        if len(edge_indices_for_shift) > 0:
            # Sum the probabilities of staff being assigned to this shift
            sum_assigned_probs = probabilities[edge_indices_for_shift].sum()
            
            # Get the required min staff for this shift
            original_shift_data = data.original_shifts_data[shift_node_idx_offset]
            min_staff_required = original_shift_data['minStaffForRole']
            
            # If sum of probabilities is less than required, apply penalty
            if sum_assigned_probs < min_staff_required:
                total_penalty += min_staff_weight * (min_staff_required - sum_assigned_probs)
    
    return total_penalty

# --- 2. Training Loop ---
def train_gnn():
    # Load dataset
    graphs = torch.load(DATASET_PATH, weights_only=False)

    if not graphs:
        print("No valid graphs loaded. Exiting.")
        return

    # Shuffle and split dataset into training and validation sets
    random.shuffle(graphs)
    split_idx = int(len(graphs) * TRAIN_SPLIT_RATIO)
    train_graphs = graphs[:split_idx]
    val_graphs = graphs[split_idx:]

    if not train_graphs:
        print("No training graphs available. Exiting.")
        return
    if not val_graphs:
        print("No validation graphs available. Proceeding with training only.")

    # Determine number of node features from the first graph
    num_node_features = train_graphs[0].x.shape[1]
    hidden_channels = 128 # Increased hidden channels
    num_epochs = 200 # Increased number of epochs
    learning_rate = 0.005 # Adjusted learning rate

    model = RosterGNN(num_node_features, hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Calculate pos_weight for imbalanced datasets
    total_pos_labels = 0
    total_neg_labels = 0
    for data in train_graphs:
        total_pos_labels += data.edge_label.sum().item()
        total_neg_labels += (data.edge_label.numel() - data.edge_label.sum()).item()
    
    if total_pos_labels == 0:
        print("Warning: No positive labels found in the training set. Model will likely predict all unassigned.")
        pos_weight = torch.tensor(1.0) # Default to 1 if no positive samples
    else:
        pos_weight = torch.tensor(total_neg_labels / total_pos_labels)
    
    print(f"Calculated pos_weight: {pos_weight.item():.2f}")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    for epoch in range(100): # Train for 100 epochs
        # Training phase
        model.train()
        train_loss = 0
        for data in train_graphs:
            optimizer.zero_grad()
            
            node_embeddings = model(data.x, data.edge_index)
            logits = model.predict_edge(node_embeddings, data.edge_index) # Get logits before sigmoid
            
            target = data.edge_label.view(-1, 1)
            bce_loss = criterion(logits, target) # Use BCEWithLogitsLoss

            # Calculate constraint penalty
            constraint_penalty = calculate_constraint_penalty(
                logits, data, DOUBLE_BOOKING_PENALTY_WEIGHT, MIN_STAFF_PENALTY_WEIGHT
            )
            
            loss = bce_loss + constraint_penalty
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_graphs)

        # Validation phase
        val_loss = 0
        if val_graphs:
            model.eval()
            with torch.no_grad():
                for data in val_graphs:
                    node_embeddings = model(data.x, data.edge_index)
                    logits = model.predict_edge(node_embeddings, data.edge_index) # Get logits before sigmoid
                    target = data.edge_label.view(-1, 1)
                    bce_loss = criterion(logits, target) # Use BCEWithLogitsLoss

                    # Calculate constraint penalty for validation
                    constraint_penalty = calculate_constraint_penalty(
                        logits, data, DOUBLE_BOOKING_PENALTY_WEIGHT, MIN_STAFF_PENALTY_WEIGHT
                    )
                    val_loss += (bce_loss + constraint_penalty).item()
            avg_val_loss = val_loss / len(val_graphs)
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

    print("Training complete. Saving model...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    # Ensure necessary directories exist
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        print("Please run 'python server/dl_poc/create_gnn_dataset.py' first to generate the dataset.")
    else:
        train_gnn()