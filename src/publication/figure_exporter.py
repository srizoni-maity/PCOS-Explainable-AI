"""
=========================================================
Figure Export Utility

Exports figures to PNG and PDF.

=========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt


class FigureExporter:

    def __init__(

        self,

        output_folder="outputs/publication/figures"

    ):

        self.output = Path(output_folder)

        self.output.mkdir(

            parents=True,

            exist_ok=True

        )

    # ------------------------------------------------------

    def save(

        self,

        filename

    ):

        png = self.output / f"{filename}.png"

        pdf = self.output / f"{filename}.pdf"

        plt.savefig(

            png,

            dpi=600,

            bbox_inches="tight"

        )

        plt.savefig(

            pdf,

            bbox_inches="tight"

        )

        plt.close()