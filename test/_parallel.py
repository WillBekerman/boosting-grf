"""Utilities for parallel execution with graceful fallback."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, Optional, Sequence, Tuple, TypeVar

import numpy as np
from tqdm.auto import tqdm

_T = TypeVar("_T")
_R = TypeVar("_R")


def run_parallel(
    func: Callable[[Tuple[int, np.random.Generator]], _R],
    tasks: Sequence[Tuple[int, np.random.Generator]],
    desc: str,
    workers: Optional[int] = None,
) -> Iterable[_R]:
    if len(tasks) == 0:
        return []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(func, task) for task in tasks]
        results = []
        for f in tqdm(futures, desc=desc, leave=False):
            results.append(f.result())
        return results
