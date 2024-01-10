import numpy as np
import pandas as pd


def find_corners(x, y):
    corner_points = []
    for a, b in zip(x, y):
        point_qualifies = not np.any((x < a) & (y > b))
        if point_qualifies:
            corner_points.append((a, b))

    corner_points.sort(key=lambda x: x[0])
    return corner_points


def generate_corner_data(corner_points):
    internal_corners = corner_points.copy()
    internal_corner_slopes = [i[1] / i[0] for i in internal_corners]
    max_slope = np.max(internal_corner_slopes)
    expr = []
    log_convexity = []
    for i in range(1, len(internal_corner_slopes) - 1):
        expr.append(internal_corner_slopes[i - 1] + internal_corner_slopes[i + 1] - 2 * internal_corner_slopes[i])
    expr = [internal_corner_slopes[1] - 2 * internal_corner_slopes[0]] + expr + [
        internal_corner_slopes[-2] - 2 * internal_corner_slopes[-1]]

    df = pd.DataFrame({"x": [i[0] for i in internal_corners], "y": [i[1] for i in internal_corners],
                       "slope=y/x": internal_corner_slopes, "convexity": expr})
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
    return df


def find_suspicious_pairs(distances, diff_in_prop):
    #  find pairs with very small EMD and large
    #  difference in properties
    pass
