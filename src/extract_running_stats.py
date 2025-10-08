import re
import sys

import numpy as np


def extract_running_values(filename):
    """Extract digits before 'RUNNING' from a text file and calculate stats."""
    running_values = []

    # Pattern to match the specific format: digits | digits RUNNING
    pattern = r"\|\s*(\d+)\s+RUNNING"

    try:
        with open(filename, "r") as file:
            for line_num, line in enumerate(file, 1):
                matches = re.findall(pattern, line)
                for match in matches:
                    running_values.append(int(match))
                    print(f"Line {line_num}: Found {match} before RUNNING")
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None, None, None, None

    if not running_values:
        print("No values found before 'RUNNING'")
        return None, None, None, None

    # Calculate statistics
    mean_val = np.mean(running_values)
    std_val = np.std(running_values, ddof=1)  # sample std deviation
    min_val = np.min(running_values)
    max_val = np.max(running_values)

    print(f"\nFound {len(running_values)} values: {running_values}")
    print(f"Average: {mean_val:.2f}")
    print(f"Standard Deviation: {std_val:.2f}")
    print(f"Minimum: {min_val}")
    print(f"Maximum: {max_val}")

    return mean_val, std_val, min_val, max_val


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_running_stats.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    extract_running_values(filename)
