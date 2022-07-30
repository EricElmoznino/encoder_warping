import seaborn as sns

sns.set(
    style="ticks",
    context="paper",
    rc={
        "font.size": 6,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "axes.labelpad": 2,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.title_fontsize": 5.5,
        "legend.fontsize": 5.5,
        "legend.markerscale": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.minor.width": 0.2,
        "ytick.minor.width": 0.2,
        "figure.constrained_layout.use": True,
        "figure.dpi": 200,
    },
)