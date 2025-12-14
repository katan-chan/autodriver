"""
Simple function registry for routing algorithms.

Usage:
    @register("algorithm_name")
    def solve(graph_data, requests, config):
        return {...}
"""

# Global registry
_ALGORITHMS = {}


def register(name: str):
    """Register an algorithm function."""
    def decorator(func):
        _ALGORITHMS[name] = func
        print(f"[REGISTRY] Registered: {name}")
        return func
    return decorator


def get(name: str):
    """Get algorithm function by name."""
    if name not in _ALGORITHMS:
        available = ", ".join(_ALGORITHMS.keys())
        raise ValueError(f"Algorithm '{name}' not found. Available: {available}")
    return _ALGORITHMS[name]


def list_all():
    """List all registered algorithm names."""
    return list(_ALGORITHMS.keys())
