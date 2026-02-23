"""
ga.py — Base classes for genetic algorithm optimisation.

Provides Population (a sorted list of individuals) and GenericIndividual
(the interface each individual must implement).  The specialised
SSopt / SOPopulation classes live in prediction.py.
"""

from __future__ import annotations

import operator
from random import normalvariate, randint


class GenericIndividual:
    """Abstract interface for a GA individual."""

    def optimize(self): pass
    def calculate_fitness(self): pass
    def initialize(self): pass
    def mutate(self): pass
    def crossover(self, other): pass


class Population(list):
    """Sorted list of GenericIndividual objects with GA operators."""

    def __init__(self):
        super().__init__()
        self.splitlib: list = []

    # ------------------------------------------------------------------
    # Core list operations
    # ------------------------------------------------------------------

    def fill_from_random(self, num: int, template) -> None:
        """Append *num* randomly initialised individuals cloned from *template*."""
        for _ in range(num):
            obj = template.initialize_random()
            self.append(obj)

    def sort(self, attr: str) -> None:  # type: ignore[override]
        list.sort(self, key=operator.attrgetter(attr))

    def getbest(self):
        self.sort("energy")
        return self[0]

    def cull(self, num: int) -> None:
        """Keep only the first *num* individuals (assumes list is sorted)."""
        if num <= 0:
            return
        del self[num:]

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def selectNormal(self, selrat: float, size: int) -> int:
        """Return a population index sampled from a half-normal distribution.

        selrat > 1 → uniform random; selrat ≤ 1 → biased toward the front.
        """
        if selrat > 1.0:
            return randint(0, size - 1)
        i = size
        while i > size - 1:
            i = int(abs(normalvariate(0, selrat)) * size)
        return i

    # ------------------------------------------------------------------
    # Population-level operations
    # ------------------------------------------------------------------

    def mergewith(self, other: "Population") -> None:
        """Merge *other* into self, skipping duplicates (by getid())."""
        seen = {obj.getid() for obj in self}
        for obj in other:
            if obj.getid() not in seen:
                self.append(obj)
                seen.add(obj.getid())

    def derive_stats(self, cnt: int) -> None:
        """Print a one-line progress summary (override in subclasses)."""
        import numpy as np
        energies = np.array([obj.energy for obj in self])
        print(
            f"  gen {cnt:4d} | pop {len(self):3d} | "
            f"E_min={energies.min():.3f}  E_avg={energies.mean():.3f}  "
            f"E_std={energies.std():.4f}  best: {self[0].getid()[:40]}"
        )
