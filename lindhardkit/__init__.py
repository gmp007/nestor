# lindhardkit/__init__.py
__version__ = "0.1.0"

from .state import STATE, RuntimeState  # lightweight and safe; remove if it causes heavy imports

__all__ = ["STATE", "RuntimeState", "__version__"]

