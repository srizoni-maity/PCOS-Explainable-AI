"""
=========================================================
STEP 19

Publication Package

=========================================================
"""

from src.publication.figures import roc
from src.publication.figures import pr
from src.publication.figures import confusion
from src.publication.figures.shap_plot import generate as generate_shap
from src.publication.figures.permutation import (
    generate as generate_permutation
)
from src.publication.figures.ablation import (
    generate as generate_ablation
)
from src.publication.figures.lofo import (
    generate as generate_lofo
)
from src.publication.figures.calibration import (
    generate as generate_calibration
)
from src.publication.figures.decision_curve import (
    generate as generate_decision_curve
)





def main():

    print("="*70)

    print("STEP 19 : PUBLICATION PACKAGE")

    print("="*70)

    print()

    print("Generating ROC...")

    roc()

    print("Generating PR...")

    pr()

    print("Generating Confusion Matrix...")

    confusion()

    print()

    print("="*70)

    print("Done")

    print("="*70)

    print()

    print("Saved to")

    print("outputs/publication/figures")

    print("Generating SHAP...")

    generate_shap()

    print("Generating Permutation Importance...")

    generate_permutation()

    print("Generating Ablation...")

    generate_ablation()

    print("Generating LOFO...")

    generate_lofo()

    print("Generating Calibration...")

    generate_calibration()

    print("Generating Decision Curve...")

    generate_decision_curve()


if __name__=="__main__":

    main()