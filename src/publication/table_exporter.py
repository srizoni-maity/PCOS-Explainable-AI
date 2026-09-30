"""
=========================================================
Publication Table Exporter

=========================================================
"""

from pathlib import Path


class TableExporter:

    def __init__(

        self,

        output_folder="outputs/publication/tables"

    ):

        self.output = Path(output_folder)

        self.output.mkdir(

            parents=True,

            exist_ok=True

        )

    # ------------------------------------------------------

    def save(

        self,

        dataframe,

        filename

    ):

        dataframe.to_csv(

            self.output / f"{filename}.csv",

            index=False

        )