import torch
import os
import sys

DATASET_PATH = '/Users/woodj/Desktop/roster-forge/server/dl_poc/data/gnn_rostering_dataset.pt'

def inspect_dataset():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}", file=sys.stderr)
        sys.exit(1)

    try:
        graphs = torch.load(DATASET_PATH, weights_only=False)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    if not graphs:
        print("Dataset is empty.", file=sys.stderr)
        sys.exit(1)

    total_edges = 0
    total_assigned_edges = 0
    total_unassigned_edges = 0
    
    for i, data in enumerate(graphs):
        if hasattr(data, 'edge_label') and data.edge_label is not None:
            num_edges = data.edge_label.numel()
            num_assigned = torch.sum(data.edge_label).item()
            num_unassigned = num_edges - num_assigned

            total_edges += num_edges
            total_assigned_edges += num_assigned
            total_unassigned_edges += num_unassigned
        else:
            print(f"Graph {i} has no 'edge_label' attribute.", file=sys.stderr)

    print(f"Dataset Inspection Results:")
    print(f"Total graphs: {len(graphs)}")
    print(f"Total edges across all graphs: {total_edges}")
    print(f"Total assigned edges (label 1): {total_assigned_edges}")
    print(f"Total unassigned edges (label 0): {total_unassigned_edges}")

    if total_edges > 0:
        print(f"Percentage assigned: {(total_assigned_edges / total_edges * 100):.2f}%")
        print(f"Percentage unassigned: {(total_unassigned_edges / total_edges * 100):.2f}%")
    else:
        print("No edges found in the dataset.")

if __name__ == '__main__':
    inspect_dataset()
