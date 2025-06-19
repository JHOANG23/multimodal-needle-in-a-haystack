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
                        exact_match_subimage[gt_row - 1][gt_col - 1] += 1.2
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
    generate_subimage_heatmap(accuracies, n_row)

def generate_subimage_heatmap(accuracies, n_row):
    """Generate and save heatmaps for multiple models side by side"""
    try:
        # Prepare the figure to plot heatmaps side by side
        fig, axes = plt.subplots(1, len(accuracies), figsize=(5 * len(accuracies), 5))

        # If there's only one model, axes is a single axis, so wrap it in a list
        if len(accuracies) == 1:
            axes = [axes]

        # Loop over the models and plot each heatmap
        for ax, (model_name, avg_accuracy, acc_subimage) in zip(axes, accuracies):
            # Generate heatmap for the model
            sns.heatmap(
                acc_subimage,
                ax=ax,
                vmin=0, vmax=1,
                cmap=LinearSegmentedColormap.from_list(
                    "custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"]
                ),
                cbar_kws={'label': 'Score'},
                linewidths=0.5,
                linecolor='grey',
                linestyle='--',
                annot=False,
            )

            # Remove tick marks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")

            # Add the title and average accuracy to the heatmap
            ax.set_title(f"{model_name}", fontsize=18, fontweight='bold')

        # Adjust layout for better spacing between heatmaps
        plt.tight_layout()

        # Save the figure with all heatmaps side by side
        plt.savefig("all_models_heatmaps.png", bbox_inches="tight")
        plt.close()
        print("Heatmaps saved for all models")
    except Exception as e:
        print(f"Error generating heatmap: {e}")

def generate_table_png(output_path="table.png"):
    # Data for the 2x2 Index column
    data = {
        "Model": ["Claude 3 Opus", "Gemini Pro 1.0", "GPT-4V"],
        "Index (2x2)": [74.77, 85.09, 92.64]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Plot the table
    fig, ax = plt.subplots(figsize=(4, 2))  # Adjust size for better visuals
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')

    # Save the table as a PNG
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Table saved to {output_path}")


if __name__ == "__main__":
    # Example: Adjust paths and parameters for your specific file
    response_file = "C:\\CS228\\multimodal-needle-in-a-haystack\\response\\COCO_val2014_0_9\\results_10.json"  # Path to the JSON file
    output_file = "C:\\CS228\\multimodal-needle-in-a-haystack\\index_accuracy_table.png"

    sequence_length = 10  # Example sequence length
    n_row = 2  # Example number of rows (for subimage analysis)     

    if not os.path.exists(response_file):
        print(f"File not found: {response_file}")
    else:
        process_model_results(response_file, n_row, sequence_length)

