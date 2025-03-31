import pandas as pd
import numpy as np
from collections import Counter

def cycle_extraction(values):
    # Save a copy of the original data for later checks (unaffected by subsequent modifications to values)
    original_values = values.copy()
    diff_df = values.diff().fillna(0)
    n = len(values)

    # Compute the list of indices of extreme points (including the start and end of the series)
    extrema_indices = [0]
    i = 1
    while i < n - 1:
        if values.iloc[i] == values.iloc[i - 1]:
            start = i - 1
            while i < n - 1 and values.iloc[i] == values.iloc[i + 1]:
                i += 1
            if i < n - 1:
                if (values.iloc[start] > values.iloc[start - 1] and values.iloc[i] > values.iloc[i + 1]) or \
                        (values.iloc[start] < values.iloc[start - 1] and values.iloc[i] < values.iloc[i + 1]):
                    extrema_indices.append(i)
        else:
            if (values.iloc[i] > values.iloc[i - 1] and values.iloc[i] > values.iloc[i + 1]) or \
                    (values.iloc[i] < values.iloc[i - 1] and values.iloc[i] < values.iloc[i + 1]):
                extrema_indices.append(i)
        i += 1
    extrema_indices.append(n - 1)

    # Calculate the absolute changes between adjacent extreme points and construct LAB intervals (A as start, B as end)
    L = [abs(values.iloc[extrema_indices[j + 1]] - values.iloc[extrema_indices[j]])
         for j in range(len(extrema_indices) - 1)]
    A = extrema_indices[:-1]
    B = extrema_indices[1:]

    cycles_intervals = {}  # Store the list of time points (indices) for each cycle
    cycles_ratios = {}  # Store the list of time ratios corresponding to each cycle
    cycle_number = 1
    # Globally record the remaining available ratio for each time point, initially set to 1 for all
    available = {i: 1 for i in range(n)}

    # Start detecting candidate cycles from the third LAB interval
    i = 2
    while i < len(L):
        l_m1 = L[i - 2]
        l_m2 = L[i - 1]
        l_m3 = L[i]
        if l_m2 <= l_m1 and l_m2 <= l_m3:
            start_index = A[i - 1]
            # Calculate the directional difference d_n for the current candidate cycle (using the original data)
            d_n = values.iloc[A[i]] - values.iloc[A[i - 1]]
            # Only extract cycles where abs(d_n) > 0.1
            if abs(d_n) > 0.1:
                found = False
                # Search for the end boundary j starting from start_index + 1
                for j in range(start_index + 1, len(values)):
                    if ((d_n > 0 and values.iloc[j] <= values.iloc[start_index]) or
                            (d_n < 0 and values.iloc[j] >= values.iloc[start_index])):
                        if diff_df.iloc[j] != 0:
                            recorder = (values.iloc[start_index] - values.iloc[j - 1]) / diff_df.iloc[j]
                        else:
                            recorder = available.get(j, 1)
                        values = values.copy()
                        values.loc[j - 1] = values.loc[start_index]
                        j_first = j
                        while j + 1 < len(values) and values.iloc[j + 1] == values.iloc[j]:
                            j += 1
                        found = True
                        break
                if not found:
                    j = start_index
                    recorder = available.get(j, 1)

                # Construct the current cycle: from start_index to j (for time points other than j, use the available ratio directly)
                current_cycle_points = []
                current_cycle_ratios = []
                for t in range(start_index + 1, j + 1):
                    if t == j_first:
                        avail_ratio = available.get(t, 1)
                        current_cycle_points.append(t)
                        current_cycle_ratios.append(recorder)
                        available[t] = avail_ratio - recorder
                    else:
                        if available.get(t, 0) > 0:
                            current_cycle_points.append(t)
                            current_cycle_ratios.append(available[t])
                            available[t] = 0  # Mark as used
                cycles_intervals[cycle_number] = current_cycle_points
                cycles_ratios[cycle_number] = current_cycle_ratios
                cycle_number += 1

                # Delete the LAB interval corresponding to the current candidate cycle and merge the remaining parts for later extraction of larger cycles
                L[i] = abs(values.iloc[B[i]] - values.iloc[A[i - 2]])
                A[i] = A[i - 2]
                del L[i - 2:i]
                del A[i - 2:i]
                del B[i - 2:i]
                i = 2  # Reset the detection starting point
            else:
                # If abs(d_n) <= 0.1, directly delete the LAB interval for this small cycle and merge the new interval
                L[i] = abs(values.iloc[B[i]] - values.iloc[A[i - 2]])
                A[i] = A[i - 2]
                del L[i - 2:i]
                del A[i - 2:i]
                del B[i - 2:i]
                i = 2
        else:
            i += 1

    # Process the remaining time points that have not been extracted by any cycle
    remaining_points = [t for t in range(n) if available.get(t, 0) > 0]
    if remaining_points:
        rem_ratios = [available[t] for t in remaining_points]
        cycles_intervals[cycle_number] = remaining_points
        cycles_ratios[cycle_number] = rem_ratios

    # ---- Post-processing phase: remove original time point 0 (if it exists) and its corresponding ratio;
    # remove all consecutive time points with 0 charge; subtract one from all time nodes ----
    for key in cycles_intervals:
        new_points = []
        new_ratios = []
        for pt, rt in zip(cycles_intervals[key], cycles_ratios[key]):
            if pt == 0:
                continue  # Remove time point 0 and its corresponding ratio
            if pt > 0 and original_values.iloc[pt] == 0 and original_values.iloc[pt - 1] == 0:
                continue  # Remove time points with consecutive 0 charge and their corresponding ratio
            new_points.append(pt - 1)  # Subtract one from all time nodes
            new_ratios.append(rt)
        cycles_intervals[key] = new_points
        cycles_ratios[key] = new_ratios
    charg_xy = []
    for cycle_number, times in cycles_intervals.items():
        charg_xy.append(len(times))
    return cycles_intervals, cycles_ratios, charg_xy
