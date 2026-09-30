"""
=========================================================
Publication Package Builder

IEEE Publication Version

=========================================================
"""

from pathlib import Path

import shutil
import json

from .figure_exporter import FigureExporter
from .table_exporter import TableExporter
from .caption_generator import CaptionGenerator


class PublicationBuilder:

    """
    Builds the complete publication package.

    outputs/publication/

        figures/
        tables/
        captions/
        paper_assets.json
    """

    def __init__(

        self,

        output_root="outputs/publication"

    ):

        self.root = Path(output_root)

        self.root.mkdir(

            parents=True,

            exist_ok=True

        )

        self.figures = FigureExporter(

            self.root / "figures"

        )

        self.tables = TableExporter(

            self.root / "tables"

        )

        self.captions = CaptionGenerator(

            self.root / "captions"

        )

        self.assets = {

            "figures": [],

            "tables": [],

            "captions": []

        }

    # --------------------------------------------------

    def register_figure(

        self,

        filename,

        title,

        caption

    ):

        self.assets["figures"].append({

            "file": filename,

            "title": title

        })

        self.captions.save(

            filename,

            title,

            caption

        )

    # --------------------------------------------------

    def register_table(

        self,

        dataframe,

        filename

    ):

        self.tables.save(

            dataframe,

            filename

        )

        self.assets["tables"].append(

            filename

        )

    # --------------------------------------------------

    def copy_file(

        self,

        source,

        destination_name=None

    ):

        source = Path(source)

        if not source.exists():

            return

        destination = self.root

        if destination_name is None:

            destination_name = source.name

        shutil.copy2(

            source,

            destination /

            destination_name

        )

    # --------------------------------------------------

    def save_manifest(self):

        manifest = self.root / "paper_assets.json"

        with open(

            manifest,

            "w",

            encoding="utf-8"

        ) as f:

            json.dump(

                self.assets,

                f,

                indent=4

            )

    # --------------------------------------------------

    def finish(self):

        self.save_manifest()

        print()

        print("=" * 60)

        print("Publication Package Generated")

        print("=" * 60)

        print(self.root)