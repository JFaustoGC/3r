"""Trajectory caching utilities for precomputed maximum speed profiles."""

import os
import pickle
import hashlib
import numpy as np
from pathlib import Path


def _compute_cache_key(v_final, T, q0, xT, config_hash):
    """
    Compute a unique cache key for trajectory parameters.
    
    Args:
        v_final: Final end-effector velocity
        T: Time duration
        q0: Initial joint configuration
        xT: Final end-effector pose
        config_hash: Hash of configuration dict
        
    Returns:
        str: Unique cache key
    """
    # Create a string representation of all parameters
    key_data = f"{v_final.tobytes()}{T}{q0.tobytes()}{xT.tobytes()}{config_hash}"
    return hashlib.md5(key_data.encode()).hexdigest()


def _get_cache_dir():
    """Get or create the cache directory."""
    cache_dir = Path.home() / ".cache" / "3r_trajectories"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _hash_config(config):
    """Create a hash of configuration dict."""
    # Convert config to a stable string representation
    config_str = str(sorted(config.items()))
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def save_trajectory_cache(v_final, T, q0, xT, config, solution):
    """
    Save a trajectory solution to cache.
    
    Args:
        v_final: Final end-effector velocity
        T: Time duration
        q0: Initial joint configuration
        xT: Final end-effector pose
        config: Configuration dict
        solution: Trajectory solution dict
    """
    cache_dir = _get_cache_dir()
    config_hash = _hash_config(config)
    cache_key = _compute_cache_key(v_final, T, q0, xT, config_hash)
    cache_file = cache_dir / f"{cache_key}.pkl"
    
    cache_data = {
        "v_final": v_final,
        "T": T,
        "q0": q0,
        "xT": xT,
        "config": config,
        "solution": solution
    }
    
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)
    
    return cache_file


def load_trajectory_cache(v_final, T, q0, xT, config):
    """
    Load a trajectory solution from cache if it exists.
    
    Args:
        v_final: Final end-effector velocity
        T: Time duration
        q0: Initial joint configuration
        xT: Final end-effector pose
        config: Configuration dict
        
    Returns:
        dict or None: Cached solution if found, None otherwise
    """
    cache_dir = _get_cache_dir()
    config_hash = _hash_config(config)
    cache_key = _compute_cache_key(v_final, T, q0, xT, config_hash)
    cache_file = cache_dir / f"{cache_key}.pkl"
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)
        return cache_data["solution"]
    except Exception as e:
        print(f"Warning: Failed to load cache: {e}")
        return None


def clear_trajectory_cache():
    """Clear all cached trajectories."""
    cache_dir = _get_cache_dir()
    if cache_dir.exists():
        for cache_file in cache_dir.glob("*.pkl"):
            cache_file.unlink()
        return True
    return False


def list_cached_trajectories():
    """List all cached trajectories with their metadata."""
    cache_dir = _get_cache_dir()
    if not cache_dir.exists():
        return []
    
    cached_items = []
    for cache_file in cache_dir.glob("*.pkl"):
        try:
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
            cached_items.append({
                "file": cache_file.name,
                "v_final": cache_data["v_final"],
                "T": cache_data["T"],
                "q0": cache_data["q0"],
                "xT": cache_data["xT"]
            })
        except Exception:
            pass
    
    return cached_items
