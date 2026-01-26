def plot_solved_time_distribution(
    matrix1_path,
    matrix2_path,
    puzzle_ids=None,
    p_type="",
    runs=(1, 2),
    figsize=(18, 6),
    show_points=False,
    savepath=None,
):
    """
    Plot solved time distributions as paired boxplots (Run 1 vs Run 2) from saved matrices.

    Parameters
    ----------
    matrix1_path : str
        Path to sol_matrix1_all.csv
    matrix2_path : str
        Path to sol_matrix2_all.csv
    puzzle_ids : list or None
        List of puzzle IDs to plot. If None, plots all puzzles.
    p_type : str
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

    # ---- load matrices ----
    sol_matrix1_all = np.genfromtxt(matrix1_path, delimiter=',')
    sol_matrix2_all = np.genfromtxt(matrix2_path, delimiter=',')

    # ---- determine puzzle range ----
    if puzzle_ids is None:
        puzzle_ids = list(range(sol_matrix1_all.shape[1]))
    
    puzzle_indices = puzzle_ids

    # ---- collect data: solving times per participant per puzzle per run ----
    data = {runs[0]: [], runs[1]: []}

    for pid in puzzle_indices:
        # Extract times for all participants for this puzzle
        times_run1 = sol_matrix1_all[:, pid]
        times_run2 = sol_matrix2_all[:, pid]
        
        # Filter out zeros (unsolved or missing data) and inf values
        times_run1 = times_run1[(times_run1 > 0) & (times_run1 != np.inf) & ~np.isnan(times_run1)]
        times_run2 = times_run2[(times_run2 > 0) & (times_run2 != np.inf) & ~np.isnan(times_run2)]
        
        data[runs[0]].append(times_run1)
        data[runs[1]].append(times_run2)

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
                if len(y) > 0:
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
    ax.set_ylabel("Time until first solved attempt [s]", fontsize=14)
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


# Example usage:
plot_solved_time_distribution(
    matrix1_path="./Data/sol_matrix1_all.csv",
    matrix2_path="./Data/sol_matrix2_all.csv",
    puzzle_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    p_type="",
    savepath="./Data/solved_time_boxplots.png"
)