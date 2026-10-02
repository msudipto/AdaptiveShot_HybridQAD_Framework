"""Camera-ready composite-figure entry point.

This module intentionally reuses the publication plotting implementation in
``src.make_figures``. It exists so the full reproducibility workflow can be
invoked with the camera-ready command:

    python -m src.make_combined_figures

No QNN retraining or finite-shot evaluation is performed here; only saved
experimental artifacts are read.
"""

from .make_figures import main

if __name__ == "__main__":
    main()
