"""Base interface implemented by every ArbiGraph adapter."""

from abc import ABC, abstractmethod


# =============================================================================
# Base adapter class
# =============================================================================


class Adapter(ABC):
    @abstractmethod
    def prompt(self, *args, **kwargs) -> str:
        """Describe the adapter transformation using model-facing variable names."""

    @abstractmethod
    def compute(self, value):
        """Apply the adapter transformation to a concrete value for evaluation."""
