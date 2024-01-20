import argparse
import os.path

from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pickle
from data import *
import numpy as np
import seaborn
from analysis import *
from tqdm import tqdm
import warnings
import gc

warnings.filterwarnings('ignore')

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


AXIS_LABELS = {
    "pdd": "L-inf based EMD on PDD100",
    "mpdd": "L-inf based EMD on mPDD100",
    "amd": "L-inf Norm of difference of AMD100"
}


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


def plot_spr(periodic_sets, targets, prop, ids=None, take_closest=10000, distance_threshold=None,
             zoomed=False, zoomed_out=False, filename="", show_plot=False, verbose=False, cache_results=True,
             metric="pdd", weighted_by="AtomicMass"):

    metric = metric.lower()
    if metric == 'mpdd':
        am = pd.read_csv('periodic_table.csv')[weighted_by]

    if prop not in targets.keys():
        print(f"Please choose from: {np.sort(targets.keys())}")
        raise ValueError(f"Error: {prop} not among properties")

    target_values = list(targets[prop])
    to_keep = [i for i in range(len(target_values)) if not np.isnan(target_values[i])]
    formulas = list(targets.formula)

    if len(to_keep) == 0:
        print(f"Zero entries for this property. Returning..")
        return

    if verbose:
        print(f"{prop} Data size: {len(to_keep)}")

    ps = [periodic_sets[i] for i in to_keep]
    property_values = [target_values[i] for i in to_keep]
    formulas = [formulas[i] for i in to_keep]

    if ids is not None:
        ids = [ids[i] for i in to_keep]

    if os.path.exists(f"./data/jarvis_{prop}_amds"):
        amds = pickle.load(open(f"./data/jarvis_{prop}_amds", "rb"))
    else:
        amds = [amd.AMD(p, k=100) for p in ps]
        if cache_results:
            pickle.dump(amds, open(f"./data/jarvis_{prop}_amds", "wb"))

    distances = compare_amds(amds)
    property_values = np.array(property_values)

    avg_dist = np.mean(distances)
    std_dist = np.std(distances)
    if distance_threshold is None:
        distance_threshold = avg_dist - 2 * std_dist

    if verbose:
        print(f"Average AMD distances: {np.mean(distances)}")
        print(f"Standard deviation of AMD distances {np.std(distances)}")

    if verbose:
        print(f"Total number of distances: {distances.shape[0]}")
    if distances.shape[0] > take_closest:
        inds = np.argsort(distances)[:take_closest]
        d = np.sort(distances)
        samples_inds = [i for i in range(take_closest, len(distances)) if d[i] < distance_threshold]
        inds = np.concatenate([inds, samples_inds])
    else:
        inds = np.argsort(distances)

    m = len(ps)
    pairs = []
    pair_jids = []
    pair_formulas = []

    for i in tqdm(inds, desc="Generating indices..."):
        pairs.append(dist_ind_to_pair_ind(m, i))
        if ids is not None:
            i1, i2 = pairs[-1]
            pair_jids.append((ids[i1], ids[i2]))
            pair_formulas.append((formulas[i1], formulas[i2]))

    if os.path.exists(f"./data/jarvis_{prop}_{metric}_pairs"):
        distances, fe_diffs = pickle.load(open(f"./data/jarvis_{prop}_pairs", "rb"))
    else:
        if verbose:
            print(f"Generating pairs for {prop}")
        distances = []

        if metric == "pdd":
            for i, j in tqdm(pairs, desc="Computing Earth Mover's Distances.."):
                distances.append(amd.EMD(amd.PDD(ps[i], k=100), amd.PDD(ps[j], k=100)))
        elif metric == "mpdd":
            for i, j in tqdm(pairs, desc="Computing weighted Earth Mover's Distances.."):
                distances.append(amd.EMD(am_weighted_pdd(ps[i], k=100, df=am), am_weighted_pdd(ps[j], k=100, df=am)))
        elif metric == "amd":
            for i, j in tqdm(pairs, desc="Computing Average Minimum Distances.."):
                distances.append(np.linalg.norm(amd.AMD(ps[i], k=100) - amd.AMD(ps[j], k=100), ord=np.inf))
        distances = np.array(distances)
        fe_diffs = np.array([abs(property_values[i] - property_values[j]) for i, j in pairs])
        pickle.dump((distances, fe_diffs), open(f"./data/jarvis_{prop}_pairs", "wb"))


    create_hist(distances, xlabel=AXIS_LABELS[metric], ylabel=f"Frequency",
                filename=f"./figures/jarvis_{prop}-vs-{metric}_1D_histogram.png")

    create_hist(distances, fe_diffs, xlabel=AXIS_LABELS[metric], ylabel=f"Absolute Difference in {prop}",
                filename=f"./figures/jarvis_{prop}-vs-{metric}_2D_histogram.png")

    plt.figure(figsize=(30, 20))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    pl = seaborn.scatterplot(x=distances, y=fe_diffs, color="black")
    pl.set_xlabel(AXIS_LABELS[metric], fontsize=46)
    pl.set_ylabel(f"Absolute Difference in {prop}", fontsize=46)

    x = distances
    y = fe_diffs

    if verbose:
        print(f"Average difference in {prop}: {np.mean(y)}")

    corner_points, corner_indices = find_corners(x, y, return_indices=True)
    internal_corners = []
    for i in range(1, len(corner_points)):
        b = corner_points[i - 1][1]
        a = corner_points[i][0]
        internal_corners.append((a, b))

    corner_jids = None
    corner_formulas = None
    if ids is not None:
        corner_jids = [pair_jids[ind] for ind in corner_indices]
        corner_formulas = [pair_formulas[ind] for ind in corner_indices]

    df, sus, dup = generate_corner_data(corner_points, ids=corner_jids, formulas=corner_formulas)
    filtered_corners = list(zip(list(df['x']), list(df['y'])))

    df.sort_values("x").to_csv(f"./data/{prop}_external_points_{metric}.csv", index=False)
    sus.sort_values("x").to_csv(f"./data/{prop}_suspicious_points_{metric}.csv", index=False)
    dup.sort_values("x").to_csv(f"./data/{prop}_duplicate_points_{metric}.csv", index=False)

    original_corners = corner_points.copy()

    # Plot staircase
    for i, corner_point in enumerate(filtered_corners):
        if i == 0:
            plt.plot([0, corner_point[0]], [0, 0], color="blue")
            plt.plot([corner_point[0], corner_point[0]], [0, corner_point[1]], color="blue")
        else:
            plt.plot([filtered_corners[i - 1][0], corner_point[0]], [filtered_corners[i - 1][1], filtered_corners[i - 1][1]],
                     color="blue")
            plt.plot([corner_point[0], corner_point[0]], [filtered_corners[i - 1][1], filtered_corners[i][1]],
                     color="blue")

    filtered_corners.sort(key=lambda x: x[0])

    #  The corner point producing the largest angular jump determines the slope
    arg_for_SPF = np.argmax(list(df["angular jump"])[1:])
    internal_corner_for_SPF = filtered_corners[1:][arg_for_SPF]
    adj_corner = [c for c in filtered_corners if c[1] == internal_corner_for_SPF[1]][0]
    if verbose:
        print(f"next_internal_corner: {adj_corner}")
        print(f"internal_corner: {internal_corner_for_SPF}")
    if adj_corner == (0, 0):
        print(f"Skipping {prop}")
        plt.xlim([0, distance_threshold / 10])
        pl.set_yticklabels(np.round(pl.get_yticks(), 2), size=30)
        pl.set_xticklabels(np.round(pl.get_xticks(), 4), size=30)
        plt.savefig(f"./figures/jarvis_{prop}-vs-{metric}_angular_jump_failed.png")
        if show_plot:
            plt.show()
        return
    SPF = list(df["slope=y/x"])[1:][arg_for_SPF]
    SPD = adj_corner[1] - SPF * adj_corner[0]
    SPD = max(SPD, 0)
    if verbose:
        print(f"y = {np.round(SPF, 3)} x + {np.round(SPD, 3)}")
    line_function = lambda x: SPF * x + SPD
    # corner_points = stage1(corner_points)
    # corner_points = stage2(corner_points)
    SPB = [corner for corner in filtered_corners if
           line_function(corner[0]) < corner[1] and corner[0] >= adj_corner[0]]
    if len(SPB) == 0:
        SPB = 0
    else:
        SPB = SPB[0][0]

    if verbose:
        print(f"Using an SPB of : {np.round(SPB, 4)}")
    potential_SPF = SPF

    if SPD > 0:
        while True:
            intersection_index = find_intersection(filtered_corners, line_function, (SPB, 0), return_index=True)
            if verbose:
                print(
                    f"previous line equation: y = {np.round(line_function(1) - line_function(0), 3)} x + {np.round(line_function(0), 3)}")
            if intersection_index == -1:
                break
            c, d = filtered_corners[intersection_index]
            amount_to_shift = (d - line_function(c)) + line_function(0)
            line_function = lambda x: potential_SPF * x + amount_to_shift
            if verbose:
                print(f"y = {np.round(potential_SPF, 3)} x + {np.round(amount_to_shift, 3)}")
                print(f"Intersects at {np.round((c, d), 3)}")
            SPD = line_function(0)
            SPF = potential_SPF

    p1 = adj_corner

    points_to_consider_for_p2 = [point for point in filtered_corners if point[0] < p1[0]]
    potential_slope_and_intercepts = [find_line_function(p1, point, return_slope_and_intercept=True) for point in
                                      points_to_consider_for_p2]

    min_slope = np.argmin([p[0] for p in potential_slope_and_intercepts])
    SPF, SPD = potential_slope_and_intercepts[min_slope]
    p2 = points_to_consider_for_p2[min_slope]
    if verbose:
        print("-------------------------------------")
        print("Optimization:")
        print(f"Using points: {p1} and {p2}")
    line_function = find_line_function(p1, p2, verbose=True)
    if verbose:
        print(f"Checking line function: {line_function(p1[0]) - p1[1]} and {line_function(p2[0]) - p2[1]}")
        print("-------------------------------------")

    SPD = line_function(0)
    if SPD < 0:
        SPD = 0
        line_function = find_line_function(p1, (0, 0))

    SPBs = [corner[0] for corner in filtered_corners if
           line_function(corner[0]) < corner[1] and abs(line_function(corner[0]) - corner[1]) > 1e-15]

    if len(SPBs) == 0:
        SPB = np.max([corner[0] for corner in filtered_corners])
    else:
        SPB = SPBs[0]

    highlighted_points = [point for point in filtered_corners if
                          (point[0] == SPB and point[1] >= line_function(SPB)) or abs(
                              line_function(point[0]) - point[1]) < 1e-10]

    print("-----------------------------------")
    print(f"SPD: {SPD} ")
    print(f"SPB: {SPB}")
    print(f"SPF: {SPF}")
    print("-----------------------------------")

    if zoomed:
        xran = [0, 2 * SPB]
        yran = [0, np.max(fe_diffs[distances < 2 * SPB])]
    elif zoomed_out:
        xran = [0, np.max(distances)]
        yran = [0, np.max(fe_diffs)]
    else:
        d = np.max([i[1] for i in highlighted_points])
        xran = [0, min(4 * SPB, np.max(distances))]
        yran = [0, min(np.max(fe_diffs[distances < 4 * SPB]), 4 * d)]
    #  Original line
    plt.plot([0, SPB], [0, potential_SPF * SPB], color="red")

    #  Shifted line
    plt.plot([0, SPB], [SPD, line_function(SPB)], color='springgreen')
    plt.plot([SPB, SPB], [line_function(SPB), np.max(fe_diffs)], color='springgreen')
    plt.scatter([i[0] for i in highlighted_points], [i[1] for i in highlighted_points], c="red", s=110)

    plt.xlim(xran)
    plt.ylim(yran)
    pl.set_yticklabels(np.round(pl.get_yticks(), 2), size=30)
    pl.set_xticklabels(np.round(pl.get_xticks(), 1 + int(np.nanmax(-np.log10(pl.get_xticks()[1:])))), size=30)
    z_suffix = "2SPB" if zoomed else "4SPB"
    m_suff = 'EMD_PDD100' if metric == 'pdd' else 'EMD_mPDD100'
    m_suff = 'AMD100' if metric == 'amd' else m_suff
    plt.savefig(f"./figures/jarvis_{prop}-vs-{m_suff}_{z_suffix}_angular_jump.png")
    if show_plot:
        plt.show()
    plt.close()


def create_hist(x, y=None, bins=10, xlabel="", ylabel="", filename=""):
    if y is None:
        ax = seaborn.histplot(x=x, bins=bins, kde=True)
    else:
        ax = seaborn.histplot(x=x, y=y, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(filename)
    plt.close()


def plot(args):
    db = args.database_name
    src = args.source_name
    crystal_data = get_data(src, db, include_id=args.include_id, prop=args.property_name)
    #crystal_data = read_jarvis_data(db, include_jid=args.include_jid, verbose=args.verbose)
    ids = None
    if len(crystal_data) > 2:
        periodic_sets, target, ids = crystal_data
    else:
        periodic_sets, target = crystal_data

    target = target.replace('na', np.nan)
    if args.run_all:
        properties_to_run = target.keys()
        for prop in properties_to_run:
            if target[prop].dtype == np.float64:
                plot_spr(periodic_sets, target, prop, ids=ids,
                         verbose=args.verbose, show_plot=args.show_plot, zoomed=args.zoomed,
                         metric=args.metric, weighted_by=args.weighted_by, zoomed_out=args.zoomed_out)
            gc.collect()
    else:
        plot_spr(periodic_sets, target, args.property_name, ids=ids,
                 verbose=args.verbose, show_plot=args.show_plot, zoomed=args.zoomed,
                 metric=args.metric, weighted_by=args.weighted_by, zoomed_out=args.zoomed_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Structure-Property Relationship Plotter',
        description='Plots EMD between PDDs at specified k-NN against difference in property value')
    parser.add_argument('source_name', type=str, help='name of data source')
    parser.add_argument('database_name', type=str, help='name of source database')
    parser.add_argument('property_name', type=str, help='name of property to plot')
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    parser.add_argument('-i', '--include-id',
                        action='store_true')
    parser.add_argument('-z', '--zoomed',
                        action='store_true')
    parser.add_argument('-o', '--zoomed-out',
                        action='store_true')
    parser.add_argument('-s', '--show-plot',
                        action='store_true')
    parser.add_argument('-a', '--run-all',
                        action='store_true')
    parser.add_argument('-m', '--metric')
    parser.add_argument('-w', '--weighted-by')
    parser.add_argument('-n', '--id-col')

    args = parser.parse_args()

    get_or_create_dir("./cache")
    get_or_create_dir("./figures")
    get_or_create_dir("./data")

    if args.zoomed_out and args.zoomed:
        raise ValueError("Please select one of '--zoomed' or '--zoomed-out'")

    if args.verbose:
        print(f"Using source: {args.source_name}")
        print(f"Using database: {args.database_name}")
        print(f"Include ID: {args.include_id}")
        print(f"Show plot: {args.show_plot}")
        print(f"Run all: {args.run_all}")
        print(f"Zoomed: {args.zoomed}")
        print(f"Metric: {args.metric}")
        print(f"Weighted by: {args.weighted_by}")
        print(f"ID col: {args.id_col}")
        print(f"Zoomed out: {args.zoomed_out}")
    plot(args)
