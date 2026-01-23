def plot_attempt_distribution(
    df,
    puzzle_ids,
    p_type,
    runs=(1, 2),
    figsize=(18, 6),
    show_points=False,
    savepath=None, 
):
    """
    Plot attempt distributions as paired boxplots (Run 1 vs Run 2) for selected puzzles.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: ['participant_id', 'run', 'puzzle_id', 'attempt']
    puzzle_ids : list
        List of puzzle IDs to plot (x-axis order preserved)
    p_type : string
        Type of puzzle (for title purposes)
    runs : tuple
        Runs to compare (default: (1, 2))
    figsize : tuple
        Figure size
    show_points : bool
        Overlay individual data points
    savepath : str or None
        If provided, saves the figure to this path
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # ---- collect data: attempts per participant per puzzle per run ----
    data = {run: [] for run in runs}

    for pid in puzzle_ids:
        for run in runs:
            counts = (
                df[(df["puzzle_id"] == pid) & (df["run"] == run)]
                .groupby("participant_id")
                .size()
                .values
            )
            data[run].append(counts)

    # ---- plotting ----
    fig, ax = plt.subplots(figsize=figsize)

    positions = np.arange(len(puzzle_ids))
    width = 0.35

    colors = {
        runs[0]: "#F8766D",  # ggplot-like red
        runs[1]: "#00BFC4",  # ggplot-like teal
    }

    for i, run in enumerate(runs):
        offset = (-width / 2) if i == 0 else (width / 2)
        pos = positions + offset

        bp = ax.boxplot(
            data[run],
            positions=pos,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=True,
            medianprops=dict(color="black", linewidth=2),
        )

        for box in bp["boxes"]:
            box.set(facecolor=colors[run], alpha=0.85)

        # ---- optional scatter overlay ----
        if show_points:
            for x, y in zip(pos, data[run]):
                ax.scatter(
                    np.full_like(y, x, dtype=float),
                    y,
                    color="black",
                    s=30,
                    alpha=0.8,
                    zorder=3,
                )

    # ---- formatting ----
    label_fontsize = 20
    tick_fontsize = 20
    legend_fontsize = 20

    ax.set_xticks(positions)
    ax.set_xticklabels(puzzle_ids, rotation=90, fontsize=tick_fontsize)
    ax.set_ylabel("Number of Attempts", fontsize=label_fontsize)
    ax.set_xlabel(f"{p_type} Puzzles", fontsize=label_fontsize)

    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=colors[runs[0]], lw=8),
            plt.Line2D([0], [0], color=colors[runs[1]], lw=8),
        ],
        labels=[f"Run {runs[0]}", f"Run {runs[1]}"],
        frameon=False,
        fontsize=legend_fontsize,
    )

    ax.grid(axis="y", linestyle=":", alpha=0.4)

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()

import pandas as pd

df = pd.read_csv("./Data/df.csv")

selected_puzzles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

plot_attempt_distribution(
    df,
    p_type="",
    puzzle_ids=selected_puzzles,
    savepath="./Data/attempt_boxplots.png"
)
