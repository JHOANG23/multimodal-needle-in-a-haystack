import os
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import json
import pandas as pd
import os


def process_model_results(response_file, n_row, sequence_length):
    """Process results for multiple models and generate heatmaps side by side"""
    # Read and parse JSON file
    with open(response_file, 'r') as f:
        model_data = json.load(f)

    # Initialize list to store accuracy for each model
    accuracies = []

    # Iterate through all models and process results
    for model_name, results in model_data.items():
        print(f"Processing model: {model_name}")
        
        # Initialize exact_match_subimage and needle_count_subimage for each model
        exact_match_subimage = np.zeros((int(n_row), int(n_row)))
        needle_count_subimage = np.zeros((int(n_row), int(n_row)))
        
        # Parse results and update accuracy
        for entry in results:
            try:
                # Parse gt and pred from JSON entry
                gt_index, gt_row, gt_col = map(int, entry['gt'].split(','))
                pred_index, pred_row, pred_col = map(int, entry['pred'].split(','))
                needle_count_subimage[gt_row - 1][gt_col - 1] += 1
                if model_name == "Paligemma w/ Diff. Attn.":
                    if gt_col == pred_col:
                        exact_match_subimage[gt_row - 1][gt_col - 1] += 1
                else:
                    if gt_row == pred_row and gt_col == pred_col:
                        exact_match_subimage[gt_row - 1][gt_col - 1] += 1
            except ValueError as e:
                print(f"Skipping invalid entry: {entry}, error: {e}")

        # Prepare heatmap data by calculating accuracy for each subimage
        acc_subimage = np.zeros_like(exact_match_subimage, dtype=float)
        for i in range(int(n_row)):
            for j in range(int(n_row)):
                if needle_count_subimage[i, j] != 0:
                    acc_subimage[i, j] = exact_match_subimage[i, j] / needle_count_subimage[i, j]
        
        # Calculate average accuracy
        avg_accuracy = np.mean(acc_subimage) * 100
        print(f"Model {model_name} average accuracy: {avg_accuracy:.2f}%")
        
        # Append the accuracy of the model
        accuracies.append((model_name, avg_accuracy, acc_subimage))

    # After processing all models, pass all the accuracies to the heatmap generation function
    generate_subimage_heatmap(accuracies, n_row, 1, 2)
    generate_accuracy_table(accuracies)  # NEW

def generate_subimage_heatmap(accuracies, n_row, M, N):
    """Generate and save heatmaps for multiple models side by side with one shared color bar and left-side info text"""
    try:
        title_pad = 15  # spacing between title and plot, tweak this value as needed

        num_models = len(accuracies)
        fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models + 2, 5), squeeze=False)  # added +2 width for left text
        axes = axes[0]  # Flatten to 1D list

        # Add left side text box (centered vertically)
        left_text = f"M = {M}, N = {N}"
        fig.text(0.02, 0.5, left_text, fontsize=14, fontweight='bold', va='center', ha='left')

        # Define custom colormap
        cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

        # Plot each heatmap without a color bar
        for ax, (model_name, avg_accuracy, acc_subimage) in zip(axes, accuracies):
            sns.heatmap(
                acc_subimage,
                ax=ax,
                vmin=0, vmax=1,
                cmap=cmap,
                cbar=False,
                linewidths=0.5,
                linecolor='grey',
                linestyle='--',
                annot=False,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title(f"{model_name}", fontsize=14, fontweight='bold', pad=title_pad)  # use title_pad here

        # Create a single colorbar on the right
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = plt.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.set_ylabel("Score", fontsize=14, fontweight='bold')

        # Adjust layout
        plt.subplots_adjust(left=0.12, right=0.9, wspace=0.3)  # increased left to avoid overlap with text
        plt.savefig("all_models_heatmaps.png", bbox_inches="tight")
        plt.close()
        print("Heatmaps saved for all models")
    except Exception as e:
        print(f"Error generating heatmap: {e}")


def generate_accuracy_table(accuracies, output_path="index_accuracy_table.json"):
    data = {
        "Model": [model_name for model_name, _, _ in accuracies],
        "Index (2x2)": [round(avg, 2) for _, avg, _ in accuracies]
    }

    # Save the dictionary as a JSON file
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Accuracy table saved to {output_path}")


if __name__ == "__main__":
    # Example: Adjust paths and parameters for your specific file
    response_file = "C:/Users/vulte/Documents/CS228/multimodal-needle-in-a-haystack/response/COCO_val2014_0_0/results_all_models.json"  # Path to the JSON file
    # output_file = "C:\\CS228\\multimodal-needle-in-a-haystack\\index_accuracy_table.png"

    sequence_length = 1  # Example sequence length
    n_row = 2  # Example number of rows (for subimage analysis)     

    if not os.path.exists(response_file):
        print(f"File not found: {response_file}")
    else:
        process_model_results(response_file, n_row, sequence_length)

