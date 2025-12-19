import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.patheffects as patheffects
from matplotlib.lines import Line2D

def plot_correlation_matrices(dataframes, dataframe_names, figsize=(16, 16), colormap="coolwarm"):
    """
    Plot correlation matrices for multiple DataFrames.

    Args:
        dataframes (list[pd.DataFrame]): List of DataFrames to plot.
        dataframe_names (list[str]): List of names corresponding to each DataFrame.
        figsize (tuple, optional): Figure size for the grid. Defaults to (16, 16).
        colormap (str, optional): Color map for heatmaps. Defaults to "coolwarm".

    Returns:
        matplotlib.figure.Figure: The created figure containing the correlation matrices.
    """
    def _plot_correlation_matrix(axis, dataframe, title):
        correlation_matrix = dataframe.corr(numeric_only=True)
        if correlation_matrix.empty:
            axis.axis("off")
            axis.text(0.5, 0.5, "No numeric columns", ha="center", va="center")
            return
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap=colormap, cbar=True, ax=axis)
        axis.set_title(title)

    num_dataframes = len(dataframes)
    num_columns = 2
    num_rows = (num_dataframes + num_columns - 1) // num_columns

    figure, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=figsize)
    axes = axes.flatten() if num_dataframes > 1 else [axes]

    for axis, dataframe, name in zip(axes, dataframes, dataframe_names):
        numeric_columns = dataframe.select_dtypes(include=["float64", "int64"])
        _plot_correlation_matrix(axis, numeric_columns, f"Correlation Matrix: {name}")

    for axis in axes[len(dataframes):]:
        axis.axis("off")

    plt.tight_layout()
    return figure

def plot_log_histogram(
    dataframe, column_name, percentile_markers=None, figsize=(8, 5),
    bins=np.logspace(3, 8, 100), bar_color="gray", transparency=0.6,
    plot_title=None, x_axis_label=None, y_axis_label="Count"
):
    """
    Plot a histogram with a log-scaled x-axis and optional percentile markers.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the data.
        column_name (str): Name of the numeric column to plot.
        percentile_markers (dict[str, float], optional): Dict of {label: value} for percentile markers. Defaults to None.
        figsize (tuple, optional): Size of the figure. Defaults to (8, 5).
        bins (array-like, optional): Bin edges for histogram. Defaults to np.logspace(3, 8, 100).
        bar_color (str, optional): Color of histogram bars. Defaults to "gray".
        transparency (float, optional): Transparency of histogram bars. Defaults to 0.6.
        plot_title (str, optional): Plot title. Defaults to None.
        x_axis_label (str, optional): X-axis label. Defaults to None.
        y_axis_label (str, optional): Y-axis label. Defaults to "Count".

    Returns:
        None
    """
    data = dataframe[column_name].dropna()

    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins, color=bar_color, alpha=transparency, edgecolor="white")

    # Add percentile lines if provided
    if percentile_markers:
        for label, value in percentile_markers.items():
            plt.axvline(value, linestyle="--", linewidth=1.5, label=f"{label} = {value:,.0f}")

    plt.xscale("log")
    plt.xlabel(x_axis_label or column_name)
    plt.ylabel(y_axis_label)
    plt.title(plot_title or f"Distribution of {column_name}")
    plt.grid(True, which="both", ls="--", lw=0.4, alpha=0.6)
    plt.legend()
    plt.tight_layout()

def plot_residual_distribution(
    residual_values,
    threshold_gamma=None,
    num_bins=60,
    bar_color="gray",
    plot_title="Distribution of Efficiency Residuals",
    figsize=(8, 5),
):
    """
    Plot the residual distribution with visual thresholds.

    Args:
        residual_values (pd.Series or np.ndarray): Residual values from the log-log efficiency model.
        threshold_gamma (float, optional): Optional threshold value (e.g., precomputed percentile). Defaults to None.
        num_bins (int, optional): Number of histogram bins. Defaults to 60.
        bar_color (str, optional): Histogram color. Defaults to "gray".
        plot_title (str, optional): Plot title. Defaults to "Distribution of Efficiency Residuals".
        figsize (tuple, optional): Figure size. Defaults to (8, 5).

    Returns:
        None
    """
    residual_values = np.asarray(residual_values.dropna() if hasattr(residual_values, "dropna") else residual_values)
    mean_value, std_dev = residual_values.mean(), residual_values.std()

    plt.figure(figsize=figsize)
    sns.histplot(residual_values, bins=num_bins, kde=True, color=bar_color, alpha=0.6)

    # Add reference lines for mean and standard deviation
    plt.axvline(mean_value, color="dodgerblue", linestyle="--", label="Mean")
    plt.axvline(mean_value + 2 * std_dev, color="salmon", linestyle="--", label="+2σ (z=2)")
    if threshold_gamma is not None:
        plt.axvline(threshold_gamma, color="limegreen", linestyle="--", label="Selected γ threshold")

    plt.title(plot_title, fontsize=13, weight="bold")
    plt.xlabel("Residual (log-scale deviation)")
    plt.ylabel("Channel count")
    plt.legend(frameon=True, fontsize=10)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def plot_category_bubble(
    dataframe,
    size_column,
    value_column,
    category_column,
    color_column=None,
    size_scaling_factor=25,
    use_log_scale_x=True,
    plot_title="Category-Level Bubble Chart",
    x_axis_label="Average Size (log scale)",
    y_axis_label="Average Value",
    color_map="coolwarm"
):
    """
    Plot a bubble chart for category-level data.
    """
    # Compute group averages
    grouped_data = (
        dataframe.groupby(category_column)
        .agg(
            avg_value=(value_column, "mean"),
            avg_size=(size_column, "mean"),
            count=(category_column, "count"),
        )
        .reset_index()
    )

    # Handle color metric
    if color_column:
        # If numeric or boolean → take mean; if categorical → take mode
        if np.issubdtype(dataframe[color_column].dtype, np.number) or dataframe[color_column].dtype == bool:
            grouped_data["color_metric"] = dataframe.groupby(category_column)[color_column].mean().values
        else:
            grouped_data["color_metric"] = (
                dataframe.groupby(category_column)[color_column]
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
                .values
            )
    else:
        grouped_data["color_metric"] = grouped_data["avg_size"]

    # Bubble sizes scaled by sqrt(count)
    bubble_sizes = np.sqrt(grouped_data["count"]) * size_scaling_factor

    plt.figure(figsize=(10, 6))
    scatter_plot = plt.scatter(
        grouped_data["avg_size"],
        grouped_data["avg_value"],
        s=bubble_sizes,
        c=grouped_data["color_metric"],
        cmap=color_map if np.issubdtype(grouped_data["color_metric"].dtype, np.number) else None,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.7,
    )

    if use_log_scale_x:
        plt.xscale("log")

    # Annotate category names
    for _, row in grouped_data.iterrows():
        plt.text(
            row["avg_size"], row["avg_value"], row[category_column],
            fontsize=8.5, ha="center", va="center", weight="bold",
            color="white",
            path_effects=[patheffects.withStroke(linewidth=0.8, foreground="black", alpha=0.8)]
        )

    # Add colorbar only if numeric
    if np.issubdtype(grouped_data["color_metric"].dtype, np.number):
        colorbar = plt.colorbar(scatter_plot)
        colorbar.set_label(color_column or size_column, rotation=270, labelpad=15)

    plt.title(plot_title, fontsize=14, weight="bold")
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_timeseries_trend(
    df,
    date_col,
    value_col,
    title="Time Series Trend",
    xlabel=None,
    ylabel=None,
    figsize=(12, 6),
    smooth=True,
    window=6,
    color="seagreen",
    alpha=0.25,
    band_width=0.5,
):
    """
    Plot a time series trend with mean and a narrower standard deviation band.

    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        date_col (str): Column name for dates.
        value_col (str): Column name for values.
        title (str, optional): Plot title. Defaults to "Time Series Trend".
        xlabel (str, optional): X-axis label. Defaults to None.
        ylabel (str, optional): Y-axis label. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (12, 6).
        smooth (bool, optional): Whether to smooth the trend. Defaults to True.
        window (int, optional): Rolling window size for smoothing. Defaults to 6.
        color (str, optional): Line color. Defaults to "seagreen".
        alpha (float, optional): Transparency of the shaded band. Defaults to 0.25.
        band_width (float, optional): Fraction of standard deviation for the band. Defaults to 0.5.

    Returns:
        None
    """
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])

    trend = (
        data.groupby(date_col)[value_col]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "avg", "std": "std"})
    )

    if smooth:
        trend["avg"] = trend["avg"].rolling(window, min_periods=1).mean()
        trend["std"] = trend["std"].rolling(window, min_periods=1).mean()

    plt.figure(figsize=figsize)

    plt.plot(trend[date_col], trend["avg"], color=color, linewidth=2.5, label="Mean")

    plt.fill_between(
        trend[date_col],
        trend["avg"] - band_width * trend["std"],
        trend["avg"] + band_width * trend["std"],
        color=color,
        alpha=alpha,
        label=f"±{band_width:.1f} SD"
    )

    plt.title(title, weight="bold")
    plt.xlabel(xlabel if xlabel else date_col)
    plt.ylabel(ylabel if ylabel else value_col)
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.show()

def plot_channel_comparison(
    df,
    target_channel,
    avg_channel_id,
    efficiency_col="efficiency_4week_filled",
    week_col="week_start",
    window=6,
    figsize=(12, 6),
):
    """
    Plot engagement efficiency of an outperformer channel versus an average baseline channel over time.

    Args:
        df (pd.DataFrame): DataFrame containing timeseries data with columns [channel, efficiency_col, week_col].
        target_channel (str): Channel ID of the outperformer.
        avg_channel_id (str): Channel ID of the average baseline channel.
        efficiency_col (str, optional): Column name for efficiency metric. Defaults to "efficiency_4week_filled".
        week_col (str, optional): Column name for week start date. Defaults to "week_start".
        window (int, optional): Rolling window size for smoothing. Defaults to 6.
        figsize (tuple, optional): Figure size for the plot. Defaults to (12, 6).

    Returns:
        None
    """
    target_ts = df.loc[df["channel"] == target_channel].copy()
    avg_ts = df.loc[df["channel"] == avg_channel_id].copy()

    target_ts.loc[:, "eff_smooth"] = (
        target_ts[efficiency_col].rolling(window, min_periods=1).mean()
    )
    avg_ts.loc[:, "eff_smooth"] = (
        avg_ts[efficiency_col].rolling(window, min_periods=1).mean()
    )

    plt.figure(figsize=figsize)

    plt.plot(
        target_ts[week_col],
        target_ts["eff_smooth"],
        color="seagreen",
        linewidth=2.5,
        label="Outperformer Channel",
    )

    plt.plot(
        avg_ts[week_col],
        avg_ts["eff_smooth"],
        color="red",
        linewidth=2,
        label="Average Gaming Channel (gaps = no uploads)",
    )

    plt.yscale("log")
    plt.title("Outperformer vs. Average Gaming Channel Over Time", weight="bold")
    plt.xlabel("Week")
    plt.ylabel("4-Week Engagement Efficiency (log scale)")
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.show()

def plot_comparison_boxplot(
    data, 
    x_col, 
    y_col, 
    hue_col=None, 
    title="Group Comparison", 
    xlabel="Group", 
    ylabel="Value", 
    palette=['#332288', '#CC6677'],
    showfliers=False,
    figsize=(10, 6),
    xticklabels=None
):
    """
    Plots a boxplot to compare distributions between groups.
    """
    plt.figure(figsize=figsize)
    ax = sns.boxplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=palette,
        showfliers=showfliers,
        legend=False
    )
    
    plt.title(title, weight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis='y', alpha=0.3)
    
    if xticklabels:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
        
    plt.show()

def plot_dual_metric_boxplot(
    data, 
    x_col, 
    metric1, 
    metric2, 
    titles, 
    ylabels, 
    xticklabels=None, 
    palette=['#332288', '#CC6677'],
    figsize=(16, 6)
):
    """
    Plots two boxplots side-by-side for dual-metric analysis.
    
    Args:
        data (pd.DataFrame): The dataset.
        x_col (str): Grouping column.
        metric1 (str): Column name for left plot.
        metric2 (str): Column name for right plot.
        titles (list): [Title 1, Title 2].
        ylabels (list): [Y-label 1, Y-label 2].
        xticklabels (list, optional): Custom labels for x-axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1
    sns.boxplot(
        data=data, x=x_col, y=metric1, hue=x_col, 
        palette=palette, showfliers=False, ax=axes[0], legend=False
    )
    axes[0].set_title(titles[0], weight='bold')
    axes[0].set_ylabel(ylabels[0])
    axes[0].grid(True, axis='y', alpha=0.3)
    if xticklabels:
        axes[0].set_xticks(range(len(xticklabels)))
        axes[0].set_xticklabels(xticklabels)

    # Plot 2
    sns.boxplot(
        data=data, x=x_col, y=metric2, hue=x_col, 
        palette=palette, showfliers=False, ax=axes[1], legend=False
    )
    axes[1].set_title(titles[1], weight='bold')
    axes[1].set_ylabel(ylabels[1])
    axes[1].grid(True, axis='y', alpha=0.3)
    if xticklabels:
        axes[1].set_xticks(range(len(xticklabels)))
        axes[1].set_xticklabels(xticklabels)

    plt.tight_layout()
    plt.show()


def plot_growth_trajectory(
    trajectory_df,
    x_col='week_num',
    y_col='cum_subs',
    hue_col='is_outperformer',
    title="Cumulative Growth Trajectory",
    xlabel="Weeks Relative to Start",
    ylabel="Cumulative Subscriber Gain",
    palette=['#332288', '#CC6677'], # Default: [Control, Outperformer]
    figsize=(12, 6)
):
    """
    Plots the median cumulative growth trajectory for two groups using Seaborn.
    
    Args:
        trajectory_df (pd.DataFrame): Dataframe containing the aggregated trajectory data.
        x_col (str): Column for relative time (x-axis).
        y_col (str): Column for the metric (y-axis).
        hue_col (str): Column defining the groups (Control vs Outperformer).
        palette (list): Colors for [Group 0, Group 1].
    """
    plt.figure(figsize=figsize)
    
    # Plot Lines
    sns.lineplot(
        data=trajectory_df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=palette,
        linewidth=3,
        legend=False
    )
    
    # Add custom legend manually to ensure correct labels
    custom_lines = [
        Line2D([0], [0], color=palette[0], lw=3),
        Line2D([0], [0], color=palette[1], lw=3)
    ]
    plt.legend(custom_lines, ['Control (Median)', 'Outperformer (Median)'], loc='upper left', frameon=False)

    plt.title(title, weight='bold', fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    # Small annotation to identify the break-even point
    plt.axvline(x=160, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()