import argparse

from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pickle
from data import *
import numpy as np
import seaborn
from analysis import *
from tqdm import tqdm

ALL_PROPS = ["formation_energy_peratom", "optb88vdw_bandgap", "optb88vdw_total_energy",
             "ehull", "mbj_bandgap", "bulk_modulus_kv", "shear_modulus_gv", 'magmom_oszicar',
             'magmom_outcar', "slme", "spillage", "kpoint_length_unit",
             'epsx', 'epsy', 'epsz', 'mepsx', 'mepsy', 'mepsz',
             'dfpt_piezo_max_dij', 'dfpt_piezo_max_eij', "exfoliation_energy", "max_efg",
             "avg_elec_mass", "avg_hole_mass", "n-Seebeck", "n-powerfact",
             "p-Seebeck", "p-powerfact"]


def c(S: amd.PeriodicSet):
    m = S.motif.shape[0]
    Vol = float(abs(np.dot(np.cross(S.cell[0], S.cell[1]), S.cell[2])))
    return (Vol / (m * V_n)) ** 1 / 3


def am_weighted_pdd(ps: amd.PeriodicSet, k=100, df=None):
    pdd, row_groups = amd.PDD(ps, k=k, collapse=False, return_row_groups=True)
    inds_represented = [i[0] for i in row_groups]
    pdd[:, 0] *= df[ps.types[inds_represented] - 1]
    pdd[:, 0] /= np.sum(pdd[:, 0])
    return pdd


def dist_ind_to_pair_ind(d, i):
    b = 1 - 2 * d
    x = np.floor((-b - np.sqrt(b ** 2 - 8 * i)) / 2).astype(int)
    y = (i + x * (b + x + 2) / 2 + 1).astype(int)
    return x, y


def find_intersection(corners, line_fn, current_point, return_index=False, verbose=False):
    i = (0, 0)
    int_indx = -1
    for indx, corner in enumerate(corners):
        y = line_fn(corner[0])
        if corner[0] >= current_point[0]:
            break
        if y < corner[1]:
            i = (corner[0], y)
            int_indx = indx
            if verbose:
                print(f"Intersection found at {int_indx}: {i}")
    if return_index:
        return int_indx
    return i


def find_line_function(p1, p2, verbose=False, return_slope_and_intercept=False):
    slope = (p1[1] - p2[1]) / (p1[0] - p2[0])
    intercept = p1[1] - slope * p1[0]
    fn = lambda x: slope * x + intercept
    if verbose:
        print(f"y = {slope} x + {intercept}")
    if return_slope_and_intercept:
        return slope, intercept
    return fn


def plot_spr(periodic_sets, targets, prop, jids=None, take_closest=10000, distance_threshold=1.0,
             zoomed=False, filename="", show_plot=False, verbose=False, cache_results=True):

    if prop not in targets.keys():
        print(f"Error: {prop} not among properties")
        print(f"Please choose from: {np.sort(targets.keys())}")
        return

    target_values = list(targets[prop])
    to_keep = [i for i in range(len(target_values)) if not np.isnan(target_values[i])]
    formulas = list(targets.formula)

    if len(to_keep) == 0:
        print(f"Zeros entries for this property. Returning..")
        return

    if verbose:
        print(f"{prop} Data size: {len(to_keep)}")

    ps = [periodic_sets[i] for i in to_keep]
    property_values = [target_values[i] for i in to_keep]
    formulas = [formulas[i] for i in to_keep]

    if jids is not None:
        jids = [jids[i] for i in to_keep]

    if os.path.exists(f"./data/jarvis_{prop}_amds"):
        amds = pickle.load(open(f"./data/jarvis_{prop}_amds", "rb"))
    else:
        amds = [amd.AMD(p, k=100) for p in ps]
        if cache_results:
            pickle.dump(amds, open(f"./data/jarvis_{prop}_amds", "wb"))

    distances = amd.AMD_pdist(amds, low_memory=True)
    property_values = np.array(property_values)
    #  fe_diffs = pdist(property_values.reshape((-1, 1)))

    if distances.shape[0] > take_closest:
        inds = np.argsort(distances)[:take_closest]
        d = np.sort(distances)
        samples_inds = [i for i in range(take_closest, len(distances)) if d[i] < distance_threshold]
        inds = np.concatenate([inds, samples_inds])
    else:
        inds = np.argsort(distances)

    #  fe_diffs = fe_diffs[inds]
    m = len(ps)
    pairs = []
    pair_jids = []
    pair_formulas = []

    for i in tqdm(inds, desc="Generating indices..."):
        pairs.append(dist_ind_to_pair_ind(m, i))
        if jids is not None:
            i1, i2 = pairs[-1]
            pair_jids.append((jids[i1], jids[i2]))
            pair_formulas.append((formulas[i1], formulas[i2]))

    if os.path.exists(f"./data/jarvis_{prop}_pairs"):
        distances, fe_diffs = pickle.load(open(f"./data/jarvis_{prop}_pairs", "rb"))
    else:
        print(f"generating pairs for jarvis")
        distances = np.array([amd.EMD(amd.PDD(ps[i], k=100), amd.PDD(ps[j], k=100)) for i, j in pairs])
        fe_diffs = np.array([abs(property_values[i] - property_values[j]) for i, j in pairs])
        pickle.dump((distances, fe_diffs), open(f"./data/jarvis_{prop}_pairs", "wb"))

    plt.figure(figsize=(30, 20))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    pl = seaborn.scatterplot(x=distances, y=fe_diffs, color="black")
    pl.set_xlabel("L-inf based EMD on PDD100", fontsize=46)
    pl.set_ylabel(f"Absolute Difference in {prop}", fontsize=46)

    x = distances
    y = fe_diffs

    corner_points, corner_indices = find_corners(x, y, return_indices=True)
    internal_corners = []
    for i in range(1, len(corner_points)):
        b = corner_points[i - 1][1]
        a = corner_points[i][0]
        internal_corners.append((a, b))

    #internal_corners = corner_points.copy()
    internal_corner_slopes = [i[1] / i[0] for i in internal_corners]
    max_slope = np.max(internal_corner_slopes)
    df = generate_corner_data(corner_points)
    filtered_corners = list(zip(list(df['x']), list(df['y'])))

    if jids is not None:
        corner_jids = [pair_jids[ind] for ind in corner_indices]
        corner_formulas = [pair_formulas[ind] for ind in corner_indices]
        #df["jid1"] = [i[0] for i in corner_jids]
        #df["jid2"] = [i[1] for i in corner_jids]
        #df["formula1"] = [i[0] for i in corner_formulas]
        #df["formula2"] = [i[1] for i in corner_formulas]
        if verbose:
            print("including JIDs in dataframe")

    df.sort_values("x").to_csv(f"./figures/{prop}_external_points.csv", index=False)
    original_corners = corner_points.copy()
    # Plot staircase
    for i, corner_point in enumerate(corner_points):
        if i == 0:
            plt.plot([0, corner_point[0]], [0, 0], color="blue")
            plt.plot([corner_point[0], corner_point[0]], [0, corner_point[1]], color="blue")
        else:
            plt.plot([corner_points[i - 1][0], corner_point[0]], [corner_points[i - 1][1], corner_points[i - 1][1]],
                     color="blue")
            plt.plot([corner_point[0], corner_point[0]], [corner_points[i - 1][1], corner_points[i][1]],
                     color="blue")

    filtered_corners.sort(key=lambda x: x[0])

    #  The corner point producing the largest angular jump determines the slope
    arg_for_SPF = np.argmax(list(df["angular jump"])[1:])
    internal_corner_for_SPF = filtered_corners[1:][arg_for_SPF]
    adj_corner = [c for c in corner_points if c[1] == internal_corner_for_SPF[1]][0]
    if verbose:
        print(f"next_internal_corner: {adj_corner}")
        print(f"internal_corner: {internal_corner_for_SPF}")
    if adj_corner == (0, 0):
        print(f"Skipping {prop}")
        plt.xlim([0, distance_threshold / 10])
        pl.set_yticklabels(np.round(pl.get_yticks(), 2), size=30)
        pl.set_xticklabels(np.round(pl.get_xticks(), 3), size=30)
        plt.savefig(f"./figures/jarvis_{prop}-vs-EMD_PDD100_angular_jump_failed.png")
        if show_plot:
            plt.show()
        return
    SPF = list(df["slope=y/x"])[1:][arg_for_SPF]
    initial_SPF = SPF
    SPD = adj_corner[1] - SPF * adj_corner[0]
    SPD = max(SPD, 0)
    if verbose:
        print(f"y = {np.round(SPF, 3)} x + {np.round(SPD, 3)}")
    line_function = lambda x: SPF * x + SPD
    # corner_points = stage1(corner_points)
    # corner_points = stage2(corner_points)
    SPB = [corner for corner in corner_points if
           line_function(corner[0]) < corner[1] and corner[0] >= adj_corner[0]]
    if len(SPB) == 0:
        SPB = 0
    else:
        SPB = SPB[0][0]
    point_before_SPB = [corner for corner in corner_points if corner[0] < SPB][-1]
    if verbose:
        print(f"Using an SPB of : {np.round(SPB, 4)}")
    potential_SPF = SPF

    if SPD > 0:
        while True:
            intersection_index = find_intersection(original_corners, line_function, (SPB, 0), return_index=True)
            if verbose:
                print(
                    f"previous line equation: y = {np.round(line_function(1) - line_function(0), 3)} x + {np.round(line_function(0), 3)}")
            if intersection_index == -1:
                break
            c, d = original_corners[intersection_index]
            amount_to_shift = (d - line_function(c)) + line_function(0)
            line_function = lambda x: potential_SPF * x + amount_to_shift
            if verbose:
                print(f"y = {np.round(potential_SPF, 3)} x + {np.round(amount_to_shift, 3)}")
                print(f"Intersects at {np.round((c, d), 3)}")
            SPD = line_function(0)
            SPF = potential_SPF

    highlighted_points = [point for point in corner_points if
                          (point[0] == SPB and point[1] >= line_function(SPB)) or line_function(point[0]) == point[
                              1]]
    p1 = adj_corner
    points_to_consider_for_p2 = [point for point in corner_points if point[0] < p1[0]]
    potential_slope_and_intercepts = [find_line_function(p1, point, return_slope_and_intercept=True) for point in
                                      points_to_consider_for_p2]
    min_slope = np.argmin([p[0] for p in potential_slope_and_intercepts])
    SPF, SPD = potential_slope_and_intercepts[min_slope]
    p2 = points_to_consider_for_p2[min_slope]
    print("-------------------------------------")
    print("Optimization:")
    print(f"Using points: {p1} and {p2}")
    line_function = find_line_function(p1, p2, verbose=True)
    print(f"Checking line function: {line_function(p1[0]) - p1[1]} and {line_function(p2[0]) - p2[1]}")
    print("-------------------------------------")
    SPD = line_function(0)
    if SPD < 0:
        SPD = 0
        line_function = find_line_function(p1, (0, 0))
    SPB = [corner[0] for corner in corner_points if
           line_function(corner[0]) < corner[1] and abs(line_function(corner[0]) - corner[1]) > 1e-15][0]
    highlighted_points = [point for point in corner_points if
                          (point[0] == SPB and point[1] >= line_function(SPB)) or abs(
                              line_function(point[0]) - point[1]) < 1e-10]

    print("-----------------------------------")
    print(f"SPD: {SPD} kJ/mol")
    print(f"SPB: {SPB}")
    print(f"SPF: {SPF}")
    # xran = [0, 0.4]
    if zoomed:
        xran = [0, 2 * SPB]
        yran = [0, np.max(fe_diffs[distances < 2 * SPB])]
    else:
        d = np.max([i[1] for i in highlighted_points])
        xran = [0, min(4 * SPB, np.max(distances))]
        yran = [0, min(np.max(fe_diffs[distances < 4 * SPB]), 4 * d)]
    #  Original line
    # plt.plot([0, SPB], [0, initial_SPF * SPB], color="red")
    #  Shifted line
    plt.plot([0, SPB], [SPD, line_function(SPB)], color='springgreen')
    plt.plot([SPB, SPB], [line_function(SPB), np.max(fe_diffs)], color='springgreen')
    plt.scatter([i[0] for i in highlighted_points], [i[1] for i in highlighted_points], c="red", s=110)

    plt.xlim(xran)
    plt.ylim(yran)
    pl.set_yticklabels(np.round(pl.get_yticks(), 2), size=30)
    pl.set_xticklabels(np.round(pl.get_xticks(), 3), size=30)
    z_suffix = "2SPB" if zoomed else "4SPB"
    plt.savefig(f"./figures/jarvis_{prop}-vs-EMD_PDD100_{z_suffix}_angular_jump.png")
    if show_plot:
        plt.show()
    plt.close()


def plot(args):
    db = args.database_name
    data = read_data(db, include_jid=args.include_jid)
    jids = None
    if len(data) > 2:
        periodic_sets, target, jids = data
    else:
        periodic_sets, target = data

    target = target.replace('na', np.nan)
    if args.run_all:
        properties_to_run = target.keys()
        for prop in properties_to_run:
            if target[prop].dtype == np.float64:
                plot_spr(periodic_sets, target, prop, jids=jids, verbose=args.verbose, show_plot=args.show_plot)
    else:
        plot_spr(periodic_sets, target, args.property_name, jids=jids, verbose=args.verbose, show_plot=args.show_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Structure-Property Relationship Plotter',
        description='Plots EMD between PDDs at specified k-NN against difference in property value')
    parser.add_argument('database_name', type=str, help='name of Jarvis-DFT database')
    parser.add_argument('property_name', type=str, help='name of property to plot')
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    parser.add_argument('-i', '--include-jid',
                        action='store_true')
    parser.add_argument('-z', '--zoomed',
                        action='store_true')
    parser.add_argument('-s', '--show-plot',
                        action='store_true')
    parser.add_argument('-a', '--run-all',
                        action='store_true')

    args = parser.parse_args()
    if args.verbose:
        print(f"Using database: {args.database_name}")
        print(f"Include JID: {args.include_jid}")
        print(f"Show plot: {args.show_plot}")
        print(f"Run all: {args.run_all}")
    plot(args)
