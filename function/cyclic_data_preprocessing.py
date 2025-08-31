import numpy as np
import pandas as pd

def redistribute_generation(
    df_gen_bus: pd.DataFrame,
    load: pd.DataFrame,
    charger: pd.DataFrame,
    discharger: pd.DataFrame,
    df_gen_bus_carrier_region: pd.DataFrame,
    regions: list[str],
    resources: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Baseline inter-regional redistribution without a start_key constraint:
    withdraw (high → low) from surplus regions and allocate to destinations
    purely in proportion to their shortages.

    Differences vs. `modify_and_redistribute`
    -----------------------------------------
    - No start_key thresholding in the allocation step.
    - Flows per resource are computed by origin-share prorating within the same resource.

    Returns
    -------
    df_final : DataFrame
        Shape (T, 1 + R*U). Per-region/per-resource values after withdraw + proportional allocation.
        Includes 'snapshot'.
    flow_df : DataFrame
        Shape (R, R). Annual inter-regional flow matrix (per resource summed over time),
        with the diagonal replaced by region self-use (= original generation − total withdrawn).
    flows_by_res : np.ndarray
        Shape (T, U, R, R). Time- and resource-resolved origin→destination flows.
    df_remaining : DataFrame
        Shape (T, 1 + U). System-wide remaining energy by resource after serving loads,
        i.e., (G_sub + alloc) minus loads, summed across all regions.
    """
    # Align indices and compute region-level net surplus/shortage (totals)
    load2      = load.set_index("snapshot")
    charge2    = charger.set_index("snapshot")
    discharge2 = discharger.set_index("snapshot")

    gen2       = df_gen_bus.copy()
    gen2.index = load2.index

    surplus = (
        (gen2[regions] + discharge2[regions] - load2[regions] - charge2[regions])
        .clip(lower=0)
        .reset_index().rename(columns={"index": "snapshot"})
    )
    shortage = (
        (load2[regions] + charge2[regions] - gen2[regions] - discharge2[regions])
        .clip(lower=0)
        .reset_index().rename(columns={"index": "snapshot"})
    )

    # Ensure carrier-by-region table has all columns and pick only snapshot + data
    df_reg = df_gen_bus_carrier_region.copy()
    for reg in regions:
        for res in resources:
            col = f"{reg}_{res}"
            if col not in df_reg.columns:
                df_reg[col] = 0
    data_cols = [f"{reg}_{res}" for reg in regions for res in resources]
    df_reg = df_reg[["snapshot"] + data_cols]

    # Build G(t,i,j) tensor
    T_steps = df_reg.shape[0]
    R = len(regions)
    U = len(resources)
    G = df_reg[data_cols].values.reshape(T_steps, R, U)

    # Region-level surplus/shortage arrays
    S_arr = surplus.set_index("snapshot")[regions].values  # (T, R)
    T_arr = shortage.set_index("snapshot")[regions].values  # (T, R)

    # Withdraw from surplus regions, high → low per resource
    withdraw = np.zeros_like(G)
    for t in range(T_steps):
        for i in range(R):
            rem = S_arr[t, i]
            for j in range(U - 1, -1, -1):  # high → low
                take = min(G[t, i, j], rem)
                withdraw[t, i, j] = take
                rem -= take
                if rem <= 0:
                    break

    # Residual after withdraw
    G_sub = G - withdraw

    # Proportional allocation to destinations by shortage share
    W = withdraw.sum(axis=1)                      # (T, U) resource pools
    total_short = T_arr.sum(axis=1, keepdims=True)# (T, 1)
    prop = np.divide(
        T_arr, total_short,
        out=np.zeros_like(T_arr),
        where=total_short != 0
    )                                             # (T, R)
    alloc = prop[:, :, None] * W[:, None, :]      # (T, R, U)

    # Final per-resource matrix (residual + allocated)
    df_final = pd.DataFrame(
        (G_sub + alloc).reshape(T_steps, R * U),
        columns=data_cols
    )
    df_final.insert(0, "snapshot", df_reg["snapshot"])

    # ----------------------
    # System-wide remaining by resource after serving loads
    # ----------------------
    G_fin = G_sub + alloc                    # (T, R, U)
    L_arr = load2[regions].values            # (T, R)

    remaining_by_res = G_fin.copy()
    for t in range(T_steps):
        for i in range(R):
            rem_load = L_arr[t, i]
            for j in range(U):
                if rem_load <= 0:
                    break
                use = min(remaining_by_res[t, i, j], rem_load)
                remaining_by_res[t, i, j] -= use
                rem_load -= use

    remaining_sum = remaining_by_res.sum(axis=1)   # (T, U)
    df_remaining = pd.DataFrame(remaining_sum, columns=resources)
    df_remaining.insert(0, "snapshot", df_reg["snapshot"])

    # ----------------------
    # Inter-regional flows by resource and annual flow matrix
    # ----------------------
    # flows_raw(t, origin→dest, resource) = withdraw(t, origin, resource) * prop(t, dest)
    flows_raw = withdraw[:, :, None, :] * prop[:, None, :, None]  # (T, R_src, R_dst, U)
    flows_by_res = flows_raw.transpose(0, 3, 1, 2)                # (T, U, R_src, R_dst)

    # Annual total flow matrix (sum over time and resources)
    flow_matrix = flows_by_res.sum(axis=(0, 1))  # (R, R)

    # Replace diagonal with self-use = original generation − total withdrawn
    orig_total = G.sum(axis=(0, 2))             # (R,)
    withdrawn_sum = withdraw.sum(axis=(0, 2))   # (R,)
    self_use = orig_total - withdrawn_sum
    np.fill_diagonal(flow_matrix, self_use)

    flow_df = pd.DataFrame(flow_matrix, index=regions, columns=regions)

    return df_final, flow_df, flows_by_res, df_remaining

def modify_and_redistribute(
    df_gen_bus_carrier_region: pd.DataFrame,
    df_gen_remain: pd.DataFrame,
    discharger: pd.DataFrame,
    regions: list[str],
    resources: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Modify per-region/per-resource residual generation and then perform
    a constrained inter-regional redistribution under a start_key rule.

    Workflow
    --------
    1) Determine, for each snapshot and region, a start_key (threshold resource):
       - `resources` is assumed to be sorted from low → high (e.g., low → high carbon/cost).
       - We scan `reversed(resources)` (high → low) to find the first *non-zero* generation
         for that region at that snapshot.
       - For that region/snapshot, all resources *below* this start_key (i.e., with lower carbon/cost)
         are set to zero in `df_gen_remain`, meaning they do not participate in the subsequent
         redistribution.

    2) Withdraw phase (high → low):
       - Compute each region’s net surplus (remain − discharger) as a scalar per region.
       - For regions with surplus, withdraw from their residual generation starting from
         higher to lower resources until the surplus amount is fully withdrawn.

    3) Allocation phase (low → high, with start_key restriction):
       - For each resource (low → high), allocate the resource pool only to regions that
         (a) still have shortage and (b) have start_key ≤ current resource index.
       - Allocation is proportional to remaining shortages within the eligible set,
         up to the available pool for that resource. Any unallocated amount of that
         resource is treated as discarded (no carry-over to other resources).

    Returns
    -------
    df_remain_final : DataFrame
        Shape (T, 1 + R*U). Final per-region, per-resource residual after
        withdraw + constrained allocation. Includes a 'snapshot' column.
    flows_by_res : np.ndarray
        Shape (T, U, R_origin, R_dest). Inter-regional flow tensor per resource,
        derived by splitting each resource pool to destinations, then prorating
        that split back to origins by the origin's share of the pool.
    """
    # ======================
    # A) Build start_key per region/snapshot and zero out resources below it
    # ======================

    # Ensure df_reg has all region_resource columns; keep only snapshot + data columns
    df_reg = df_gen_bus_carrier_region.copy()
    for region in regions:
        for resource in resources:
            col = f"{region}_{resource}"
            if col not in df_reg.columns:
                df_reg[col] = 0
    data_cols = [f"{region}_{resource}" for region in regions for resource in resources]
    df_reg = df_reg[["snapshot"] + data_cols]

    # Ensure df_remain_mod has all region_resource columns
    df_remain_mod = df_gen_remain.copy()
    for region in regions:
        for resource in resources:
            col = f"{region}_{resource}"
            if col not in df_remain_mod.columns:
                df_remain_mod[col] = 0

    # Use snapshot as the index to align row-wise operations
    df_reg.set_index("snapshot", inplace=True)
    df_remain_mod.set_index("snapshot", inplace=True)

    # Determine start_key by scanning high → low; zero out lower (cleaner/cheaper) resources in df_remain_mod
    descending = list(reversed(resources))  # high → low
    start_idx_df = pd.DataFrame(index=df_reg.index, columns=regions, dtype=int)

    for snapshot in df_reg.index:
        for region in regions:
            # First non-zero resource scanning high → low
            start_key = next(
                (src for src in descending if df_reg.at[snapshot, f"{region}_{src}"] != 0),
                None
            )
            # If all zero, default to the lowest index resource
            if start_key is None:
                start_key = resources[0]
            idx = resources.index(start_key)
            start_idx_df.at[snapshot, region] = idx

            # Zero out all lower (cleaner/cheaper) resources in the *residual* matrix
            for src in resources[:idx]:
                colm = f"{region}_{src}"
                if colm in df_remain_mod.columns:
                    df_remain_mod.at[snapshot, colm] = 0

    # Restore snapshot as a column for subsequent reshape/merge
    df_remain_mod = df_remain_mod.reset_index()
    start_idx_df = start_idx_df.reset_index().rename(columns={"snapshot": "snapshot"})

    # ======================
    # B) Withdraw (high → low) and constrained allocation (low → high)
    # ======================

    # Rebuild matrix used to create tensors; ensure all columns exist
    df_rem = df_remain_mod.copy()
    for reg in regions:
        for res in resources:
            col = f"{reg}_{res}"
            if col not in df_rem.columns:
                df_rem[col] = 0

    # Region-level total remain per snapshot
    remain_totals = pd.DataFrame({
        reg: df_rem[[f"{reg}_{res}" for res in resources]].sum(axis=1)
        for reg in regions
    })
    remain_totals["snapshot"] = df_rem["snapshot"]
    remain2 = remain_totals.set_index("snapshot")

    # Align discharger on snapshot index
    dis2 = discharger.set_index("snapshot")

    # S_arr/T_arr: region-level surplus/shortage scalars (per snapshot)
    S_arr = (remain2[regions] - dis2[regions]).clip(lower=0).values  # (T, R)
    T_arr = (dis2[regions]   - remain2[regions]).clip(lower=0).values  # (T, R)

    # Build G(t,i,j): residual generation before withdraw/allocation
    df_rem = df_rem[["snapshot"] + data_cols]
    T_steps, R, U = df_rem.shape[0], len(regions), len(resources)
    G = df_rem[data_cols].values.reshape(T_steps, R, U)

    # Withdraw from surplus regions, high → low per resource
    withdraw = np.zeros_like(G)
    for t in range(T_steps):
        for i in range(R):
            rem = S_arr[t, i]
            for j in range(U - 1, -1, -1):  # high → low
                if rem <= 0:
                    break
                take = min(G[t, i, j], rem)
                withdraw[t, i, j] = take
                rem -= take

    # Residual after withdraw
    G_sub = G - withdraw

    # Resource pools (sum over origins) for each (t, resource)
    W = withdraw.sum(axis=1)  # (T, U)

    # Map start_key indices to the same row order as df_rem
    start_idx_map = start_idx_df.set_index("snapshot")[regions].loc[df_rem["snapshot"]].values  # (T, R)

    # Constrained allocation: for each resource j (low → high),
    # only allocate to destinations with shortage > 0 and start_key ≤ j
    alloc = np.zeros_like(G)
    for t in range(T_steps):
        shortage_rem = T_arr[t].copy()  # (R,)
        for j in range(U):  # low → high
            pool = W[t, j]
            if pool <= 0:
                continue

            eligible_mask = (shortage_rem > 0) & (start_idx_map[t] <= j)
            need = shortage_rem[eligible_mask].sum()
            if need <= 0:
                continue

            take = min(pool, need)  # allocate up to the actual unmet need
            frac = shortage_rem[eligible_mask] / need
            alloc_amounts = take * frac

            idx = np.where(eligible_mask)[0]
            alloc[t, idx, j] += alloc_amounts
            shortage_rem[idx] -= alloc_amounts
            # Numerical guard: avoid tiny negative leftovers
            shortage_rem[idx] = np.maximum(shortage_rem[idx], 0.0)

    # Final per-resource residual after withdraw + constrained allocation
    G_final = G_sub + alloc
    df_remain_final = pd.DataFrame(
        G_final.reshape(T_steps, R * U),
        columns=data_cols
    )
    df_remain_final.insert(0, "snapshot", df_rem["snapshot"])

    # Build per-resource inter-regional flows:
    # For each resource j at time t, split destination allocation `alloc` back to origins
    # by the origin share in `withdraw`.
    W_safe = np.where(W == 0, 1, W)          # avoid division by zero
    ratio = withdraw / W_safe[:, None, :]    # (T, R_origin, U)
    ratio_tr = ratio.transpose(0, 2, 1)      # (T, U, R_origin)
    alloc_tr = alloc.transpose(0, 2, 1)      # (T, U, R_dest)
    flows_by_res = ratio_tr[:, :, :, None] * alloc_tr[:, :, None, :]  # (T, U, R_origin, R_dest)

    return df_remain_final, flows_by_res

def cycle_extraction(values):
    """
    Extract charge/discharge cycles from a storage SOC (state of charge) time series.

    Parameters
    ----------
    values : pd.Series
        SOC time series (monotonic index). Units arbitrary but consistent.

    Returns
    -------
    cycles_intervals : dict[int, list[int]]
        For each cycle id, the list of time indices (after final post-processing)
        that belong to that cycle. Each index participates with the corresponding ratio.
    cycles_ratios : dict[int, list[float]]
        For each cycle id, the per-time share (0..1) of the cycle’s energy at that
        time index; i.e., the fraction of the total charge/discharge at that time
        allocated to this cycle.
    charg_xy : list[int]
        The length (number of time indices) for each extracted cycle.

    Notes
    -----
    • The algorithm first locates local extrema (including endpoints), forming
      LAB segments with magnitudes L between successive extrema.
    • Starting from the 3rd LAB, it tests a candidate cycle whenever
      L[i-1] ≤ L[i-2]. The signed difference d_n over the candidate span is
      evaluated on the (possibly updated) series.
    • If |d_n| > 0.1 (or both ends are zero), a cycle is extracted:
      an end boundary j is searched backwards; a fractional “recorder” is computed
      to split the first step if needed. A global `available`(t)∈[0,1] tracks how
      much of each time index can still be assigned to future cycles, supporting
      overlapping cycles that sum to 1 per time.
    • After extracting a cycle, the adjacent LABs are merged and scanning restarts.
      If |d_n| ≤ 0.1, the small cycle is discarded by merging LABs without recording.
    • Remaining unassigned indices at the end are grouped into one final “largest” cycle.
    • Finally, index 0 (and zero→zero transitions) are dropped and all indices are
      shifted by −1 to align with downstream logic.
    """
    # Keep an untouched copy for later trimming rules
    original_values = values.copy()

    # First differences (used for boundary split ratio “recorder”); fill initial NaN with 0
    diff_df = values.diff().fillna(0)
    n = len(values)

    # ---- 1) Find extrema (local minima/maxima) including endpoints ----
    extrema_indices = [0]
    i = 1
    while i < n - 1:
        if values.iloc[i] == values.iloc[i - 1]:
            # Handle flat plateau: extend to the end of the plateau
            start = i - 1
            while i < n - 1 and values.iloc[i] == values.iloc[i + 1]:
                i += 1
            # If plateau is a peak/valley relative to neighbors, record the end of plateau
            if i < n - 1:
                if (values.iloc[start] > values.iloc[start - 1] and values.iloc[i] > values.iloc[i + 1]) or \
                   (values.iloc[start] < values.iloc[start - 1] and values.iloc[i] < values.iloc[i + 1]):
                    extrema_indices.append(i)
        else:
            # Single-point extremum
            if (values.iloc[i] > values.iloc[i - 1] and values.iloc[i] > values.iloc[i + 1]) or \
               (values.iloc[i] < values.iloc[i - 1] and values.iloc[i] < values.iloc[i + 1]):
                extrema_indices.append(i)
        i += 1
    extrema_indices.append(n - 1)

    # ---- 2) Build LAB segments between successive extrema ----
    L = [abs(values.iloc[extrema_indices[j + 1]] - values.iloc[extrema_indices[j]])
         for j in range(len(extrema_indices) - 1)]
    A = extrema_indices[:-1]  # segment starts
    B = extrema_indices[1:]   # segment ends

    cycles_intervals = {}   # cycle_id -> list of time indices
    cycles_ratios = {}      # cycle_id -> list of ratios at those indices
    cycle_number = 1

    # Global availability: each time index can be used up to 1.0 across cycles
    available = {i: 1 for i in range(n)}

    # ---- 3) Scan LABs and extract cycles whenever L[i-1] ≤ L[i-2] ----
    i = 2
    while i < len(L):
        l_m1 = L[i - 2]
        l_m2 = L[i - 1]
        if l_m2 <= l_m1:
            start_index = A[i - 1]
            end_point = B[i - 1]

            # Signed difference over candidate (computed on current values)
            d_n = values.iloc[A[i - 1]] - values.iloc[B[i - 1]]

            # Threshold to filter tiny cycles; allow special case (both ends 0)
            if abs(d_n) > 0.1 or (values.iloc[A[i - 2]] == 0 and values.iloc[B[i - 1]] == 0):
                found = False
                # Search backward for the boundary j that crosses the endpoint level
                end_index = max(A[i - 2] - 1, 0)
                for j in range(start_index - 1, end_index, -1):
                    cond_down = (d_n > 0 and values.iloc[j] <= values.iloc[end_point])
                    cond_up   = (d_n < 0 and values.iloc[j] >= values.iloc[end_point])
                    if cond_down or cond_up:
                        # Fraction of the step to include at j+1; guard diff == 0
                        recorder = (values.iloc[j + 1] - values.iloc[end_point]) / diff_df.iloc[j + 1] \
                                   if diff_df.iloc[j + 1] != 0 else 0
                        # Temporarily flatten the step (local copy to keep original_values intact)
                        values = values.copy()
                        values.loc[j + 1] = values.loc[end_point]
                        j_first = j + 1
                        found = True
                        break
                if not found:
                    # No clean crossing found: use earliest allowed index and full ratio
                    j_first = end_index
                    recorder = 1

                # ---- Build the cycle (indices and ratios) using `available` ----
                current_cycle_points = []
                current_cycle_ratios = []
                for t in range(j_first, end_point + 1):
                    if t == j_first:
                        avail_ratio = available.get(t, 1)
                        current_cycle_points.append(t)
                        current_cycle_ratios.append(recorder)
                        # Consume a fraction at the first point
                        available[t] = avail_ratio - recorder
                    else:
                        # Use any remaining availability at t
                        if available.get(t, 0) > 0:
                            current_cycle_points.append(t)
                            current_cycle_ratios.append(available[t])
                            available[t] = 0

                cycles_intervals[cycle_number] = current_cycle_points
                cycles_ratios[cycle_number] = current_cycle_ratios
                cycle_number += 1

                # ---- Merge LABs around the extracted cycle and restart scanning ----
                L[i] = abs(values.iloc[B[i]] - values.iloc[A[i - 2]])
                A[i] = A[i - 2]
                del L[i - 2:i]
                del A[i - 2:i]
                del B[i - 2:i]
                i = 2  # restart after modification
            else:
                # Tiny cycle: merge LABs without recording a cycle
                L[i] = abs(values.iloc[B[i]] - values.iloc[A[i - 2]])
                A[i] = A[i - 2]
                del L[i - 2:i]
                del A[i - 2:i]
                del B[i - 2:i]
                i = 2
        else:
            i += 1

    # ---- 4) Any leftover availability becomes one final (largest) cycle ----
    remaining_points = [t for t in range(n) if available.get(t, 0) > 0]
    if remaining_points:
        rem_ratios = [available[t] for t in remaining_points]
        cycles_intervals[cycle_number] = remaining_points
        cycles_ratios[cycle_number] = rem_ratios

    # ---- 5) Final clean-up: drop t==0 and zero→zero transitions; shift all indices by −1 ----
    for key in cycles_intervals:
        new_points = []
        new_ratios = []
        for pt, rt in zip(cycles_intervals[key], cycles_ratios[key]):
            # Remove index 0
            if pt == 0:
                continue
            # Remove points where both current and previous SOC are zero (flat zero run)
            if pt > 0 and original_values.iloc[pt] == 0 and original_values.iloc[pt - 1] == 0:
                continue
            # Shift by −1 to align with downstream time indexing
            new_points.append(pt - 1)
            new_ratios.append(rt)
        cycles_intervals[key] = new_points
        cycles_ratios[key] = new_ratios

    # Cycle lengths
    charg_xy = [len(times) for _, times in cycles_intervals.items()]

    return cycles_intervals, cycles_ratios, charg_xy