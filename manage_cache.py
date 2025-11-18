"""Utility script to manage trajectory cache."""

import argparse
from trajectory.cache import clear_trajectory_cache, list_cached_trajectories


def main():
    parser = argparse.ArgumentParser(description="Manage trajectory cache")
    parser.add_argument(
        "action",
        choices=["list", "clear"],
        help="Action to perform: list cached trajectories or clear cache"
    )
    
    args = parser.parse_args()
    
    if args.action == "list":
        cached = list_cached_trajectories()
        if not cached:
            print("No cached trajectories found.")
        else:
            print(f"Found {len(cached)} cached trajectory(ies):\n")
            for i, item in enumerate(cached, 1):
                print(f"{i}. {item['file']}")
                print(f"   T: {item['T']:.2f}s")
                print(f"   q0: [{item['q0'][0]:.3f}, {item['q0'][1]:.3f}, {item['q0'][2]:.3f}]")
                print(f"   xT: [{item['xT'][0]:.3f}, {item['xT'][1]:.3f}, {item['xT'][2]:.3f}]")
                print(f"   v_final: [{item['v_final'][0]:.3f}, {item['v_final'][1]:.3f}, {item['v_final'][2]:.3f}]")
                print()
    
    elif args.action == "clear":
        if clear_trajectory_cache():
            print("Cache cleared successfully.")
        else:
            print("No cache to clear.")


if __name__ == "__main__":
    main()
