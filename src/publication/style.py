"""
=========================================================
IEEE Publication Plot Style

Shared plotting style for all publication figures.

=========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl


class IEEEStyle:

    """
    Shared plotting style for all figures.
    """

    def __init__(self):

        self.figure_width = 6.5
        self.figure_height = 4.5

        self.dpi = 600

        self.font_family = "Times New Roman"

        self.font_size = 9

        self.title_size = 10

        self.label_size = 9

        self.tick_size = 8

        self.legend_size = 8

        self.line_width = 2

        self.marker_size = 6

        self.grid_alpha = 0.25

    # ------------------------------------------------------

    def apply(self):

        mpl.rcParams.update({

            "font.family": self.font_family,

            "font.size": self.font_size,

            "axes.titlesize": self.title_size,

            "axes.labelsize": self.label_size,

            "xtick.labelsize": self.tick_size,

            "ytick.labelsize": self.tick_size,

            "legend.fontsize": self.legend_size,

            "figure.dpi": self.dpi,

            "savefig.dpi": self.dpi,

            "axes.linewidth": 1,

            "grid.alpha": self.grid_alpha,

            "lines.linewidth": self.line_width,

            "lines.markersize": self.marker_size,

            "figure.facecolor": "white",

            "axes.facecolor": "white",

            "savefig.facecolor": "white",

            "savefig.bbox": "tight"

        })

    # ------------------------------------------------------

    def figure(self):

        self.apply()

        fig = plt.figure(

            figsize=(

                self.figure_width,

                self.figure_height

            )

        )

        return fig

    # ------------------------------------------------------

    @staticmethod
    def finalize(

        xlabel=None,

        ylabel=None,

        title=None,

        legend=True,

        grid=True

    ):

        if xlabel:

            plt.xlabel(xlabel)

        if ylabel:

            plt.ylabel(ylabel)

        if title:

            plt.title(title)

        if grid:

            plt.grid(True)

        if legend:

            handles, labels = plt.gca().get_legend_handles_labels()

            if len(handles):

                plt.legend(frameon=False)

        plt.tight_layout()