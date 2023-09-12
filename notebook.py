from typing import Any, cast

from IPython.core.getipython import get_ipython
from IPython.core.interactiveshell import InteractiveShell

ipython = cast(InteractiveShell, get_ipython())

import numpy as np
import numpy.typing as npt

import matplotlib
#matplotlib.use("GTK4Cairo")
matplotlib.use("pdf")

import matplotlib.pyplot as plt

#from matplotlib_inline.backend_inline import set_matplotlib_formats
#set_matplotlib_formats("png")
#from jupyterthemes import jtplot
#jtplot.style()

ipython.run_line_magic("reload_ext", "autoreload")
ipython.run_line_magic("autoreload", "1")

import glickbest
ipython.run_line_magic("aimport", "glickbest")

fig, ax = cast(tuple[plt.Figure, plt.Axes], plt.subplots())

x = np.linspace(0, 32, 200)
y = np.sin(x)

ax.plot(x, y)

plt.savefig("/tmp/out.pdf")
