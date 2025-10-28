# ruff: noqa
"""Visualization utils.

Functions to plot and see results.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from src.config import Configuration
from src.data import SampleImage, apply_pipeline
from src.models import RoadSegmentationDataset


def visualize_originals_by_seed(
    CONFIG: Configuration,
    dataset: RoadSegmentationDataset,
    seed: int,
    cols_per_row: int = 5,
    show: bool = False,
):
    """Plot original images and their ground-truth masks deterministically.

    Displays up to `cols_per_row` samples selected deterministically using
    `seed`. For each sample the original image is shown on the first row and
    its mask on the second row. No augmentation is applied.
    """
    rng = np.random.RandomState(seed)
    N = len(dataset)
    n_samples = min(cols_per_row, N)

    if N <= n_samples:
        indices = list(range(N))
    else:
        indices = list(rng.choice(N, size=n_samples, replace=False))

    # determine a target size from the first selected sample
    first_sample = dataset.sample_points[indices[0]]
    t_img, t_gt = first_sample.get_images(keep_in_memory=False)
    target_size = t_img.size

    n_cols = n_samples
    n_rows = 2
    fig_w = max(6, 3.2 * n_cols)
    fig_h = max(6, 3.5 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    def norm_mask(a: np.ndarray) -> np.ndarray:
        a = a.astype(np.float32)
        if a.max() > 1.5:
            a = (a / 255.0).clip(0, 1)
        return a

    for col_idx, idx in enumerate(indices):
        sample: SampleImage = dataset.sample_points[idx]
        orig_img, orig_gt = sample.get_images(keep_in_memory=False)

        orig_img = orig_img.convert("RGB").resize(target_size, resample=Image.BILINEAR)
        orig_gt = orig_gt.convert("L").resize(target_size, resample=Image.NEAREST)

        orig_img_arr = np.array(orig_img)
        orig_gt_arr = norm_mask(np.array(orig_gt))

        ax_img = axes[0, col_idx]
        ax_img.imshow(orig_img_arr)
        ax_img.axis("off")
        ax_img.set_title(f"Idx {idx} - Orig")

        ax_mask = axes[1, col_idx]
        ax_mask.imshow(orig_gt_arr, cmap="gray", vmin=0, vmax=1)
        ax_mask.axis("off")
        ax_mask.set_title("GT")

    plt.tight_layout()
    plt.savefig(f"{CONFIG.log_folder}/originals_seed_{seed}.png")
    if show:
        plt.show()


def visualize_augmentations_by_seed(
    CONFIG: Configuration,
    dataset: RoadSegmentationDataset,
    seed: int,
    cols_per_row: int = 5,
    show: bool = False,
):
    """Visualize augmentations deterministically using a seed.

    For each of `cols_per_row` samples (selected deterministically from the
    dataset using `seed`) this function will display 3 rows:
      1) original image
      2) augmented image (one pipeline chosen deterministically per-sample)
      3) ground truths of both (original | augmented) concatenated horizontally

    Args:
        CONFIG (Configuration): project configuration (used for logging path)
        dataset (RoadSegmentationDataset): dataset instance (must have pipelines)
        seed (int): Seed to deterministically pick samples and augmentations
        cols_per_row (int): Number of columns (samples) to plot
        show (bool): If True, call plt.show()
    """
    if dataset.pipelines is None or len(dataset.pipelines) == 0:
        raise ValueError("Dataset has no augmentation pipelines configured")

    rng = np.random.RandomState(seed)
    N = len(dataset)
    n_samples = min(cols_per_row, N)

    # pick distinct indices deterministically
    if N <= n_samples:
        indices = list(range(N))
    else:
        indices = list(rng.choice(N, size=n_samples, replace=False))

    # For layout: for each sample we show two columns (orig, aug)
    n_cols = n_samples * 2
    n_rows = 2  # row 0: images, row 1: masks

    # Determine a common target size (use first sample original image)
    first_sample = dataset.sample_points[indices[0]]
    t_img, t_gt = first_sample.get_images(keep_in_memory=False)
    target_size = t_img.size  # (width, height)

    # Create figure sized so images are visible and not squished
    fig_w = max(6, 3.2 * n_cols)
    fig_h = max(6, 3.5 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    # normalize axes shape
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    def norm_mask(a: np.ndarray) -> np.ndarray:
        a = a.astype(np.float32)
        if a.max() > 1.5:
            a = (a / 255.0).clip(0, 1)
        return a

    for s_idx, idx in enumerate(indices):
        sample: SampleImage = dataset.sample_points[idx]
        orig_img, orig_gt = sample.get_images(keep_in_memory=False)

        # choose a pipeline deterministically for this sample
        pipe_idx = rng.randint(0, len(dataset.pipelines))
        sample_seed = int((seed + idx) & 0xFFFFFFFF)
        pipeline = dataset.pipelines[pipe_idx]
        aug_img, aug_gt = apply_pipeline(sample, pipeline, sample_seed)

        # Convert and resize everything to target_size
        orig_img = orig_img.convert("RGB").resize(target_size, resample=Image.BILINEAR)
        aug_img = aug_img.convert("RGB").resize(target_size, resample=Image.BILINEAR)

        orig_gt = orig_gt.convert("L").resize(target_size, resample=Image.NEAREST)
        aug_gt = aug_gt.convert("L").resize(target_size, resample=Image.NEAREST)

        orig_img_arr = np.array(orig_img)
        aug_img_arr = np.array(aug_img)
        orig_gt_arr = norm_mask(np.array(orig_gt))
        aug_gt_arr = norm_mask(np.array(aug_gt))

        # Column indices for this sample (orig then aug)
        col_base = s_idx * 2

        # Plot images on first row
        ax_img_orig = axes[0, col_base]
        ax_img_orig.imshow(orig_img_arr)
        ax_img_orig.axis("off")
        ax_img_orig.set_title(f"Idx {idx}\nOrig")

        ax_img_aug = axes[0, col_base + 1]
        ax_img_aug.imshow(aug_img_arr)
        ax_img_aug.axis("off")
        ax_img_aug.set_title(f"Aug (pipe {pipe_idx})")

        # Plot masks on second row
        ax_mask_orig = axes[1, col_base]
        ax_mask_orig.imshow(orig_gt_arr, cmap="gray", vmin=0, vmax=1)
        ax_mask_orig.axis("off")
        ax_mask_orig.set_title("GT Orig")

        ax_mask_aug = axes[1, col_base + 1]
        ax_mask_aug.imshow(aug_gt_arr, cmap="gray", vmin=0, vmax=1)
        ax_mask_aug.axis("off")
        ax_mask_aug.set_title("GT Aug")

    plt.tight_layout()
    plt.savefig(f"{CONFIG.log_folder}/augmentations_seed_{seed}.png")
    if show:
        plt.show()


def plot_model_scores(all_scores: dict):
    """Plot model scores from a dictionary of scores.

    Args:
        all_scores (dict):
            Dictionary with model names as keys and a tuple of (model_path, scores_dict) as values.
            scores_dict should contain metrics like IoU_mean, Dice_mean, etc.
    """
    model_data = []
    for model_name, (model_path, scores) in all_scores.items():
        # Clean model name by removing encoder information
        clean_name = model_name
        # Remove ENC=encoder_name part
        import re

        clean_name = re.sub(r"-ENC=[^-]+", "", clean_name)

        row = {"Model": clean_name}
        row.update(scores)
        model_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(model_data)

    # Extract components from model names for better grouping
    df["Architecture"] = df["Model"].str.extract(r"ARC=([^,]+)")
    df["Encoder"] = df["Model"].str.extract(r"ENC=([^,]+)")
    df["Augmentation"] = df["Model"].str.extract(r"AUG=([^,]+)")
    df["Loss"] = df["Model"].str.extract(r"LSS=([^,]+)")
    df["Model"] = df["Architecture"] + "_" + df["Augmentation"] + "_" + df["Loss"]

    # Define metrics to plot
    metrics = [
        "IoU_mean",
        "Dice_mean",
        "Precision_mean",
        "Recall_mean",
        "PixelAcc_mean",
        "PR_AUC_mean",
    ]

    # Create colors for architectures
    colors = px.colors.qualitative.Set3
    arch_colors = {
        arch: colors[i % len(colors)] for i, arch in enumerate(df["Architecture"].unique())
    }

    # Create interactive figure with buttons for metric selection
    fig = go.Figure()

    # Add traces for each metric (initially hidden except the first one)
    for metric_idx, metric in enumerate(metrics):
        for idx, model_row in df.iterrows():
            fig.add_trace(
                go.Bar(
                    x=[model_row["Model"]],
                    y=[model_row[metric]],
                    name=model_row["Architecture"],
                    marker_color=arch_colors[model_row["Architecture"]],
                    text=f"{model_row[metric]:.3f}",
                    textposition="outside",
                    visible=(metric_idx == 0),  # Only show first metric initially
                    legendgroup=model_row["Architecture"],
                    showlegend=(
                        metric_idx == 0
                        and model_row["Architecture"]
                        not in [
                            trace.legendgroup
                            for trace in fig.data
                            if hasattr(trace, "showlegend") and trace.showlegend
                        ]
                    ),
                )
            )

    # Create buttons for metric selection
    buttons = []
    for metric_idx, metric in enumerate(metrics):
        # Create visibility array for this metric
        visibility = []
        for m_idx in range(len(metrics)):
            for _ in range(len(df)):
                visibility.append(m_idx == metric_idx)

        buttons.append(
            dict(
                label=metric.replace("_mean", ""),
                method="update",
                args=[
                    {"visible": visibility},
                    {
                        "title": f"Model Performance: {metric.replace('_mean', '')}",
                        "yaxis.title": metric.replace("_mean", ""),
                    },
                ],
            )
        )

    # Update layout with buttons
    fig.update_layout(
        title=f"Model Performance: {metrics[0].replace('_mean', '')}",
        xaxis_title="Models",
        yaxis_title=metrics[0].replace("_mean", ""),
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        updatemenus=[
            dict(
                type="buttons",
                direction="down",  # Changed from "left" to "down" for vertical layout
                buttons=buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.02,  # Moved to right side
                xanchor="left",
                y=1.0,  # Positioned at top right
                yanchor="top",
            ),
        ],
        margin=dict(r=150),  # Add right margin to make space for buttons
    )

    # Update x-axis labels to be more readable
    fig.update_xaxes(tickangle=45)

    fig.show()

    # Create a detailed comparison table
    print("\n" + "=" * 100)
    print("DETAILED MODEL PERFORMANCE COMPARISON")
    print("=" * 100)

    # Sort by IoU_mean (descending)
    df_sorted = df.sort_values("IoU_mean", ascending=False)

    for idx, row in df_sorted.iterrows():
        print(f"\nModel: {row['Model']}")
        print(
            f"Architecture: {row['Architecture']} | Encoder: {row['Encoder']} | Aug: {row['Augmentation']} | Loss: {row['Loss']}"
        )
        print(
            f"IoU: {row['IoU_mean']:.4f} | Dice: {row['Dice_mean']:.4f} | Precision: {row['Precision_mean']:.4f}"
        )
        print(
            f"Recall: {row['Recall_mean']:.4f} | PixelAcc: {row['PixelAcc_mean']:.4f} | PR_AUC: {row['PR_AUC_mean']:.4f}"
        )
        print("-" * 80)

    # # Create a heatmap of metrics by model configuration
    # metrics_matrix = df[["Architecture", "Augmentation", "Loss"] + metrics].copy()

    # # Create a more focused visualization by architecture
    # fig2 = go.Figure()

    # for arch in df["Architecture"].unique():
    #     arch_data = df[df["Architecture"] == arch]

    #     fig2.add_trace(
    #         go.Scatter(
    #             x=arch_data["IoU_mean"],
    #             y=arch_data["Dice_mean"],
    #             mode="markers+text",
    #             name=arch,
    #             text=arch_data["Augmentation"] + "_" + arch_data["Loss"],
    #             textposition="top center",
    #             marker=dict(
    #                 size=arch_data["PR_AUC_mean"] * 20,  # Size based on PR_AUC
    #                 color=arch_colors[arch],
    #                 opacity=0.7,
    #                 line=dict(width=2, color="white"),
    #             ),
    #         )
    #     )

    # fig2.update_layout(
    #     title="Model Performance: IoU vs Dice (bubble size = PR_AUC)",
    #     xaxis_title="IoU Mean",
    #     yaxis_title="Dice Mean",
    #     showlegend=True,
    #     width=800,
    #     height=600,
    # )

    # fig2.show()


def plot_model_scores_by_architecture(all_scores: dict):
    """Plot model scores grouped by architecture with data augmentation coloring.

    Args:
        all_scores (dict):
            Dictionary with model names as keys and a tuple of (model_path, scores_dict) as values.
            scores_dict should contain metrics like IoU_mean, Dice_mean, etc.
    """
    model_data = []
    for model_name, (model_path, scores) in all_scores.items():
        # Clean model name by removing encoder information
        clean_name = model_name
        # Remove ENC=encoder_name part
        import re

        clean_name = re.sub(r"-ENC=[^-]+", "", clean_name)

        row = {"Model": clean_name, "Original_Model": model_name}
        row.update(scores)
        model_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(model_data)

    # Extract components from model names for better grouping
    df["Architecture"] = df["Original_Model"].str.extract(r"ARC=([^,-]+)")
    df["Encoder"] = df["Original_Model"].str.extract(r"ENC=([^,-]+)")
    df["Augmentation"] = df["Original_Model"].str.extract(r"AUG=([^,-]+)")
    df["Loss"] = df["Original_Model"].str.extract(r"LSS=([^,-]+)")

    # Fill NaN values for ViT models or missing components
    df["Architecture"] = df["Architecture"].fillna("ViT")
    df["Encoder"] = df["Encoder"].fillna("ViT-Base")
    df["Augmentation"] = df["Augmentation"].fillna("Unknown")
    df["Loss"] = df["Loss"].fillna("Unknown")

    # Create display names: Loss function only (since we're grouping by architecture)
    df["Display_Model"] = df["Loss"]

    # Define colors for data augmentation strategies with consistent ordering
    sorted_augmentations = ["none", "single", "double", "all", "Unknown"]
    # augmentation_types = df['Augmentation'].unique()
    # # Sort augmentation types according to the desired order
    # sorted_augmentations = [aug for aug in augmentation_order if aug in augmentation_types]
    # # Add any additional augmentations that weren't in our predefined order
    # sorted_augmentations.extend([aug for aug in augmentation_types if aug not in augmentation_order])

    colors = px.colors.qualitative.Set3
    aug_colors = {aug: colors[i % len(colors)] for i, aug in enumerate(sorted_augmentations)}

    # Get unique architectures for buttons
    architectures = df["Architecture"].unique()

    # Create interactive figure
    fig = go.Figure()

    # Add traces for each architecture and augmentation combination
    for arch_idx, architecture in enumerate(architectures):
        arch_data = df[df["Architecture"] == architecture]

        # Keep track of which augmentations we've seen for this architecture's legend
        arch_augmentations_shown = set()

        for idx, model_row in arch_data.iterrows():
            # Show legend for each augmentation type for each architecture
            show_legend_for_this_trace = model_row["Augmentation"] not in arch_augmentations_shown
            if show_legend_for_this_trace:
                arch_augmentations_shown.add(model_row["Augmentation"])

            fig.add_trace(
                go.Bar(
                    x=[model_row["Display_Model"]],
                    y=[model_row["IoU_mean"]],
                    name=f"{model_row['Augmentation']}",
                    marker_color=aug_colors[model_row["Augmentation"]],
                    text=f"{model_row['IoU_mean']:.3f}",
                    textposition="outside",
                    visible=(arch_idx == 0),  # Only show first architecture initially
                    legendgroup=model_row["Augmentation"],
                    showlegend=show_legend_for_this_trace,
                )
            )

    # Create buttons for architecture selection
    buttons = []
    for arch_idx, architecture in enumerate(architectures):
        # Create visibility array for this architecture
        visibility = []
        showlegend_updates = {}
        trace_idx = 0

        for a_idx, arch in enumerate(architectures):
            arch_data = df[df["Architecture"] == arch]
            arch_augmentations_shown = set()

            for idx, model_row in arch_data.iterrows():
                is_visible = a_idx == arch_idx
                visibility.append(is_visible)

                # Update showlegend for this trace
                show_legend_for_this_trace = (
                    model_row["Augmentation"] not in arch_augmentations_shown
                )
                if show_legend_for_this_trace:
                    arch_augmentations_shown.add(model_row["Augmentation"])

                showlegend_updates[trace_idx] = is_visible and show_legend_for_this_trace
                trace_idx += 1

        buttons.append(
            dict(
                label=architecture,
                method="update",
                args=[
                    {"visible": visibility, "showlegend": list(showlegend_updates.values())},
                    {
                        "title": f"IoU Performance by Data Augmentation - {architecture} Architecture"
                    },
                ],
            )
        )

    # Update layout with buttons on the right side
    fig.update_layout(
        title=f"IoU Performance by Data Augmentation - {architectures[0]} Architecture",
        xaxis_title="Loss Function",
        yaxis_title="IoU Mean",
        height=600,
        width=1200,
        showlegend=True,
        legend=dict(
            title="Data Augmentation", orientation="v", yanchor="top", y=1, xanchor="left", x=1.02
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="down",
                buttons=buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.15,  # Further right to avoid legend overlap
                xanchor="left",
                y=1.0,
                yanchor="top",
            ),
        ],
        margin=dict(r=200),  # Larger right margin for buttons and legend
    )

    # Update x-axis labels to be more readable
    fig.update_xaxes(tickangle=45)

    fig.show()

    # Print summary by architecture
    print("\n" + "=" * 100)
    print("IoU PERFORMANCE BY ARCHITECTURE AND DATA AUGMENTATION")
    print("=" * 100)

    for architecture in architectures:
        arch_data = df[df["Architecture"] == architecture]
        print(f"\n{architecture} Architecture:")
        print(f"Number of models: {len(arch_data)}")

        # Group by augmentation and show statistics (in proper order)
        for augmentation in sorted_augmentations:
            if augmentation in arch_data["Augmentation"].values:
                aug_data = arch_data[arch_data["Augmentation"] == augmentation]
                mean_iou = aug_data["IoU_mean"].mean()
                max_iou = aug_data["IoU_mean"].max()
                print(f"  {augmentation:12}: Mean IoU={mean_iou:.4f}, Best IoU={max_iou:.4f}")

        # Best model for this architecture
        best_idx = arch_data["IoU_mean"].idxmax()
        best_model = arch_data.loc[best_idx]
        print(
            f"  Best model: {best_model['Loss']} with {best_model['Augmentation']} augmentation (IoU={best_model['IoU_mean']:.4f})"
        )
        print("-" * 80)


def plot_iou_boxplots_by_parameter(all_scores: dict, group_by="AUG"):
    """Plot IoU distribution as box plots grouped by a specific parameter.

    Args:
        all_scores (dict):
            Dictionary with model names as keys and a tuple of (model_path, scores_dict) as values.
            scores_dict should contain metrics like IoU_mean, Dice_mean, etc.
        group_by (str): Parameter to group by. Options: 'ARC' (Architecture), 'LSS' (Loss), 'AUG' (Augmentation)
    """
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    model_data = []
    for model_name, (model_path, scores) in all_scores.items():
        # Clean model name by removing encoder information
        clean_name = model_name
        # Remove ENC=encoder_name part
        import re

        clean_name = re.sub(r"-ENC=[^-]+", "", clean_name)

        row = {"Model": clean_name, "Original_Model": model_name}
        row.update(scores)
        model_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(model_data)

    # Extract components from model names for better grouping
    df["Architecture"] = df["Original_Model"].str.extract(r"ARC=([^,-]+)")
    df["Encoder"] = df["Original_Model"].str.extract(r"ENC=([^,-]+)")
    df["Augmentation"] = df["Original_Model"].str.extract(r"AUG=([^,-]+)")
    df["Loss"] = df["Original_Model"].str.extract(r"LSS=([^,-]+)")

    # Fill NaN values for ViT models or missing components
    df["Architecture"] = df["Architecture"].fillna("ViT")
    df["Encoder"] = df["Encoder"].fillna("ViT-Base")
    df["Augmentation"] = df["Augmentation"].fillna("Unknown")
    df["Loss"] = df["Loss"].fillna("Unknown")

    # Define parameter mapping and ordering
    param_mapping = {
        "ARC": ("Architecture", ["FPN", "PSPNet", "Unet", "ViT"]),
        "LSS": ("Loss", ["DiceLoss", "BCEWithLogitsLoss", "BCEDice"]),
        "AUG": ("Augmentation", ["none", "single", "double", "all", "Unknown"]),
    }

    if group_by not in param_mapping:
        raise ValueError(f"group_by must be one of {list(param_mapping.keys())}")

    column_name, preferred_order = param_mapping[group_by]

    # Get unique values in the preferred order
    unique_values = df[column_name].unique()
    ordered_values = [val for val in preferred_order if val in unique_values]
    ordered_values.extend([val for val in unique_values if val not in preferred_order])

    # Create colors
    colors = px.colors.qualitative.Set3
    group_colors = {val: colors[i % len(colors)] for i, val in enumerate(ordered_values)}

    # Create interactive figure
    fig = go.Figure()

    # Add box plot for each group
    for group_value in ordered_values:
        group_data = df[df[column_name] == group_value]
        iou_values = group_data["IoU_mean"].values

        fig.add_trace(
            go.Box(
                y=iou_values,
                name=group_value,
                marker_color=group_colors[group_value],
                boxpoints="all",  # Show all points
                jitter=0.3,  # Add some jitter to see overlapping points
                pointpos=-1.8,  # Position points to the left of the box
                showlegend=False,
            )
        )

    # Update layout
    param_names = {"ARC": "Architecture", "LSS": "Loss Function", "AUG": "Data Augmentation"}

    fig.update_layout(
        title=f"IoU Distribution by {param_names[group_by]}",
        xaxis_title=param_names[group_by],
        yaxis_title="IoU Mean",
        height=600,
        width=800,
        showlegend=False,
    )

    fig.show()

    # Print summary statistics
    print("\n" + "=" * 80)
    print(f"IoU STATISTICS BY {param_names[group_by].upper()}")
    print("=" * 80)

    for group_value in ordered_values:
        group_data = df[df[column_name] == group_value]
        iou_values = group_data["IoU_mean"]

        print(f"\n{group_value}:")
        print(f"  Count: {len(iou_values)}")
        print(f"  Mean:  {iou_values.mean():.4f}")
        print(f"  Std:   {iou_values.std():.4f}")
        print(f"  Min:   {iou_values.min():.4f}")
        print(f"  Max:   {iou_values.max():.4f}")
        print(f"  Q1:    {iou_values.quantile(0.25):.4f}")
        print(f"  Q3:    {iou_values.quantile(0.75):.4f}")

        # Best model in this group
        best_idx = iou_values.idxmax()
        best_model = group_data.loc[best_idx]
        print(f"  Best:  {best_model['Original_Model']} (IoU={best_model['IoU_mean']:.4f})")
        print("-" * 60)
