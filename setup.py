from __future__ import annotations

from pathlib import Path

from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy as np


ROOT = Path(__file__).parent.resolve()
GRF_CORE = ROOT / "grf" / "core" / "src"


def collect_core_sources() -> list[str]:
    """Return all C++ sources required to build the GRF core."""
    core_sources: list[str] = []
    for path in GRF_CORE.rglob("*.cpp"):
        core_sources.append(str(path.relative_to(ROOT)))
    return core_sources


extension_sources = [str(Path("boosting_grf") / "_grf_bindings.cpp")]
extension_sources.extend(collect_core_sources())

ext_modules = [
    Pybind11Extension(
        "boosting_grf._grf",
        sources=extension_sources,
        include_dirs=[
            str(GRF_CORE),
            str(ROOT / "grf" / "core" / "third_party"),
            str(ROOT / "grf" / "core" / "third_party" / "Eigen"),
            str(ROOT / "grf" / "core" / "third_party" / "optional"),
            str(ROOT / "grf" / "core" / "third_party" / "random"),
            np.get_include(),
        ],
        cxx_std=17,
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
