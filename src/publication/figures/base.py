"""
=========================================================
Base Publication Figure

Shared utilities for every publication figure.

=========================================================
"""

from src.publication import IEEEStyle
from src.publication import FigureExporter


class BaseFigure:

    def __init__(self):

        self.style = IEEEStyle()

        self.exporter = FigureExporter()

    # -------------------------------------------------

    def create(self):

        self.style.figure()

    # -------------------------------------------------

    def save(self, filename):

        self.exporter.save(filename)

    # -------------------------------------------------

    def finish(

        self,

        xlabel=None,

        ylabel=None,

        title=None,

        legend=True,

        grid=True

    ):

        IEEEStyle.finalize(

            xlabel=xlabel,

            ylabel=ylabel,

            title=title,

            legend=legend,

            grid=grid

        )