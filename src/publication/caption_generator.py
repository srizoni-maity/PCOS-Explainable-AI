"""
=========================================================
Automatic Caption Generator

=========================================================
"""

from pathlib import Path


class CaptionGenerator:

    def __init__(

        self,

        output_folder="outputs/publication/captions"

    ):

        self.output = Path(output_folder)

        self.output.mkdir(

            parents=True,

            exist_ok=True

        )

    # ------------------------------------------------------

    def save(

        self,

        filename,

        title,

        caption

    ):

        with open(

            self.output / f"{filename}.txt",

            "w",

            encoding="utf-8"

        ) as f:

            f.write(title)

            f.write("\n\n")

            f.write(caption)