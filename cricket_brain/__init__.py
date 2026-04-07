"""
Cricket-Brain — Biomorphic AI inference engine.

This package contains the Rust source code and Python bindings for
cricket-brain's delay-line coincidence detection engine.

To build the Python bindings:
    cd cricket_brain/crates/python
    pip install maturin
    maturin build --release
    pip install target/wheels/cricket_brain-*.whl

The Rust source is included for reference and for building
native bindings on your platform.
"""

# Re-export compiled Rust bindings if available (built via maturin)
try:
    import importlib
    import sys
    # The .so lives alongside this __init__.py when pip-installed,
    # or in dist-packages when the local dir shadows the install.
    _so_name = "cricket_brain.cpython-" + "".join(map(str, sys.version_info[:2]))
    _mod = importlib.import_module(".cricket_brain", __name__)
    Brain = _mod.Brain
    BrainConfig = _mod.BrainConfig
except (ImportError, AttributeError, ModuleNotFoundError):
    pass
