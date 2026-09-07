"""Backward-compatible entry point for the multi-format VersaMammo evaluator.

The implementation now lives in ``evaluate_versamammo.py`` and supports NPZ,
JPG/JPEG, and PNG images and masks.
"""

from evaluate_versamammo import main


if __name__ == "__main__":
    main()
