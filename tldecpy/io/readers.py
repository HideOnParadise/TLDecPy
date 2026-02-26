"""
Simple data ingestion.
"""

import csv
import numpy as np


def read_csv_xy(filepath: str, header: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads a 2-column CSV (Temperature, Intensity).
    Returns (x, y) as numpy arrays.
    """
    x_list = []
    y_list = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        if header:
            next(reader, None)
        for row in reader:
            if row and len(row) >= 2:
                try:
                    x_list.append(float(row[0]))
                    y_list.append(float(row[1]))
                except ValueError:
                    continue
    return np.array(x_list), np.array(y_list)
