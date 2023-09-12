import numpy as np

from typing import cast
import matplotlib
matplotlib.use("svg")

import matplotlib.pyplot as plt

ratings = np.arange(8)*100+1500
names = ("Random Pairs", "Single Elimination\nTournament", "GlickBest")
win_probs = [0.36, 0.59, 0.77]

fig, ax = cast(tuple[plt.Figure, plt.Axes], plt.subplots())
ax.set_title("GlickBest vs. Others\n(35 comparisons, 8 options, 100 point rating spread)")
ax.set_ylabel("Accuracy")
ax.bar(names, win_probs)
plt.savefig("mainplot.svg")

