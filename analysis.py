import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import pdist


def find_corners(x, y, return_indices=False):
    corner_points = []
    corner_inds = []
    ind = 0
    for a, b in zip(x, y):
        point_qualifies = not np.any((x < a) & (y > b))
        if point_qualifies:
            corner_points.append((a, b))
            corner_inds.append(ind)
        ind += 1

    corner_points.sort(key=lambda x: x[0])
    if return_indices:
        return corner_points, corner_inds
    return corner_points


def generate_corner_data(corner_points, ids=None, formulas=None):
    duplicates = []
    suspicious = []
    internal_corners = []
    used_ids = set()
    for indx, point in enumerate(corner_points):
        if ids is not None:
            id1, id2 = ids[indx]
        if formulas is not None:
            f1, f2 = formulas[indx]

        if point == (0, 0):
            if formulas is not None and (f1 != f2):
                suspicious.append([point[0], point[1], id1, id2, f1, f2])
            else:
                duplicates.append([point[0], point[1], id1, id2, f1, f2])
            continue

        if id1 == id2:
            duplicates.append([point[0], point[1], id1, id2, f1, f2])
            continue

        if point[0] == 0 and point[1] > 0:
            suspicious.append([point[0], point[1], id1, id2, f1, f2])
        else:
            internal_corners.append([point[0], point[1], id1, id2, f1, f2])

    internal_corner_slopes = [i[1] / i[0] for i in internal_corners]
    max_slope = np.max(internal_corner_slopes)
    expr = []
    log_convexity = []
    for i in range(1, len(internal_corner_slopes) - 1):
        expr.append(internal_corner_slopes[i - 1] + internal_corner_slopes[i + 1] - 2 * internal_corner_slopes[i])

    expr = [internal_corner_slopes[1] - 2 * internal_corner_slopes[0]] + expr + [
        internal_corner_slopes[-2] - 2 * internal_corner_slopes[-1]]

    df = pd.DataFrame({"x": [i[0] for i in internal_corners], "y": [i[1] for i in internal_corners],
                       "slope=y/x": internal_corner_slopes, "convexity": expr,
                       "id1": [i[2] for i in internal_corners], "id2": [i[3] for i in internal_corners],
                       "formula1": [i[4] for i in internal_corners], "formula2": [i[5] for i in internal_corners]})

    dup_df = pd.DataFrame({"x": [i[0] for i in duplicates], "y": [i[1] for i in duplicates],
                           "id1": [i[2] for i in duplicates], "id2": [i[3] for i in duplicates],
                           "formula1": [i[4] for i in duplicates], "formula2": [i[5] for i in duplicates]})

    sus_df = pd.DataFrame({"x": [i[0] for i in suspicious], "y": [i[1] for i in suspicious],
                           "id1": [i[2] for i in suspicious], "id2": [i[3] for i in suspicious],
                           "formula1": [i[4] for i in suspicious], "formula2": [i[5] for i in suspicious]})

    df["log(slope)"] = np.log(df["slope=y/x"])
    b_n = list(np.log(df["slope=y/x"]))
    jumps = [internal_corners[0][1]]
    for i in range(1, len(internal_corners)):
        jumps.append(internal_corners[i][1] - internal_corners[i - 1][1])

    for i in range(1, len(b_n) - 1):
        log_convexity.append(b_n[i - 1] + b_n[i + 1] - 2 * b_n[i])
    log_convexity = [b_n[1] - 2 * b_n[0]] + list(log_convexity) + [b_n[len(b_n) - 2] - 2 * b_n[len(b_n) - 3]]
    df["log-convexity"] = log_convexity
    df["normalized angle"] = np.arctan(df["slope=y/x"] / max_slope) * 180 / np.pi
    a_n = df["normalized angle"].to_numpy()
    a_n = np.concatenate([a_n, [0]])
    df["angular jump"] = a_n[1:] - a_n[:-1]
    return df, sus_df, dup_df


def compare_amds(amds, metric='chebyshev'):
    m = len(amds)
    amds = np.asarray(amds)

    if m < 1000:
        pdist(amds, metric=metric)

    cdm = np.empty((m * (m - 1)) // 2, dtype=np.float64)
    ind = 0
    for i in tqdm(range(m), desc="Comparing AMDs..."):
        ind_ = ind + m - i - 1
        cdm[ind:ind_] = np.amax(np.abs(amds[i + 1:] - amds[i]), axis=-1)
        ind = ind_
    return cdm


