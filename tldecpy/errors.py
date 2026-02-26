"""Public exception hierarchy for ``tldecpy``."""

from __future__ import annotations

from .schemas import ErrorDetail


class TLDecPyError(Exception):
    """Base class for all public package errors."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def to_detail(self) -> ErrorDetail:
        """Convert the exception to a serializable error payload."""
        return ErrorDetail(
            error_code=self.__class__.__name__,
            message=self.message,
            hint=self.hint,
        )


class PygcdError(TLDecPyError):
    """Backward-compatible alias for legacy public API."""


class ModelKeyError(TLDecPyError, KeyError):
    """Raised when a model key is not found in the model registry."""


class DomainError(TLDecPyError, ValueError):
    """Raised when a parameter is outside its physical domain."""


class ConvergenceError(TLDecPyError):
    """Raised when an optimizer cannot converge."""


class TypingError(TLDecPyError, TypeError):
    """Raised for invalid input data types."""


class DatasetError(TLDecPyError, FileNotFoundError):
    """Raised when a dataset cannot be found or loaded."""
