# -*- coding: utf-8 -*-
"""BaseQuantifier abstract class."""

from abc import ABC, abstractmethod
from typing import Any, List


class BaseQuantifier(ABC):
    """Interfață de bază pentru toate cuantificatorii."""

    @abstractmethod
    def initial_score(self) -> float:
        """Scorul inițial pentru setul vid"""
        pass

    @abstractmethod
    def update_score(self, current_score: float, new_element: Any,
                    current_subset: List[Any]) -> float:
        """Actualizează scorul cu un nou element"""
        pass

    @abstractmethod
    def meets_threshold(self, score: float, threshold: float) -> bool:
        """Verifică dacă scorul atinge pragul"""
        pass
