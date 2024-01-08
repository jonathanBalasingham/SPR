import os

import amd
import matplotlib
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
import seaborn
import random
import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
from scipy.special import gamma
import pickle

import math
from matplotlib.ticker import StrMethodFormatter



seaborn.set_style("darkgrid")
gamma_r3 = gamma(3/2 + 1)
V_n = (np.pi ** (3/2)) / gamma_r3


def find_line_function(p1, p2, verbose=False, return_slope_and_intercept=False):
    slope = (p1[1] - p2[1]) / (p1[0] - p2[0])
    intercept = p1[1] - slope * p1[0]
    fn = lambda x: slope * x + intercept  
    if verbose:
        print(f"y = {slope} x + {intercept}")
    if return_slope_and_intercept:
        return (slope, intercept)
    return fn


def bounds(fn, corner_points, limit):
    for point in corner_points:
        if point[0] > limit:
            break
        if fn(point[0]) < point[1]:
            return False
    return True
     

def c(S: amd.PeriodicSet):
    m = S.motif.shape[0]
    Vol = float(abs(np.dot(np.cross(S.cell[0], S.cell[1]), S.cell[2])))
    return (Vol / (m*V_n)) ** 1/3

def am_weighted_pdd(ps: amd.PeriodicSet, k=100, df=None):
    pdd, row_groups = amd.PDD(ps, k=k, collapse=False, return_row_groups=True)
    inds_represented = [i[0] for i in row_groups]
    pdd[:, 0] *= df[ps.types[inds_represented] - 1]
    pdd[:, 0] /= np.sum(pdd[:, 0])
    return pdd


def dist_ind_to_pair_ind(d, i):
    b = 1 - 2 * d
    x = np.floor((-b - np.sqrt(b**2 - 8*i))/2).astype(int)
    y = (i + x * (b + x + 2) / 2 + 1).astype(int)
    return (x,y)

def find_intersection(corners, line_fn, current_point, return_index=False):
    i = (0, 0)
    int_indx = -1
    for indx, corner in enumerate(corners):
        y = line_fn(corner[0])
        if corner[0] >= current_point[0]:
            break
        if y < corner[1]:
            i = (corner[0], y)
            int_indx = indx
            print(f"Intersection found at {int_indx}: {i}")
    if return_index:
        return int_indx
    return i

def plot_dist_vs_lattice_energy_v5_internal_corner(datasets=["./data/T0_Predicted_Structures.cif", "./data/T1_Predicted_Structures.cif", "./data/T2_Predicted_Structures.cif"], zoomed=False):
    for data_path in datasets:
        crystal_set_name = os.path.basename(data_path).split(".")[0].split("_")[0]
        ps = pickle.load(open(f"./data/{crystal_set_name}_ps", "rb"))
        used_names = set()
        unique_ps = []
        for p in ps:
            if p.name not in used_names:
                unique_ps.append(p)
                used_names.add(p.name)
                
        ps = unique_ps
        names = [p.name for p in ps]
        le = [float(n.split("_")[0]) for n in names]
        if os.path.exists(f"./data/{crystal_set_name}_amds"):
            amds = pickle.load(open(f"./data/{crystal_set_name}_amds", "rb"))
        else:
            amds = [amd.AMD(p, k=100) for p in ps]
            pickle.dump(amds, open(f"./data/{crystal_set_name}_amds", "wb"))

        distances = amd.AMD_pdist(amds, low_memory=True)
        le = np.array(le)
        fe_diffs = pdist(le.reshape((-1, 1)))

        inds = np.argsort(distances)[:20000]
        d = np.sort(distances)
        samples_inds = [i for i in range(20000, len(distances)) if d[i] < 0.6 and i % 10000 == 0]
        inds = np.concatenate([inds, samples_inds])
                
        fe_diffs = fe_diffs[inds]
        m = len(ps)
        pairs = []
        for i in inds:
            pairs.append(dist_ind_to_pair_ind(m, i))

        if os.path.exists(f"./data/{crystal_set_name}_pairs"):
            distances, fe_diffs = pickle.load(open(f"./data/{crystal_set_name}_pairs", "rb"))
        else:
            print(f"generating pairs for {crystal_set_name}")
            distances = np.array([amd.EMD(amd.PDD(ps[i], k=100), amd.PDD(ps[j], k=100)) for i, j in pairs])
            fe_diffs = np.array([abs((float(ps[i].name.split("_")[0])) - (float(ps[j].name.split("_")[0]))) for i,j in pairs])
            pickle.dump((distances, fe_diffs), open(f"./data/{crystal_set_name}_pairs", "wb"))

        plt.figure(figsize=(30,20))
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        pl = seaborn.scatterplot(x=distances, y=fe_diffs, color="black")
        pl.set_xlabel("L-inf based EMD on PDD100", fontsize=46)
        pl.set_ylabel("Absolute Difference in Lattice Energy (kJ/mol)", fontsize=46)
        
        x = distances
        y = fe_diffs

        corner_points = []
        internal_corners = []
        for a, b in zip(x, y):
            point_qualifies = not np.any((x < a) & (y > b))
            if point_qualifies:
                corner_points.append((a,b))

        corner_points.sort(key=lambda x: x[0])
        for i in range(1, len(corner_points)):
            b = corner_points[i-1][1]
            a = corner_points[i][0]
            internal_corners.append((a,b))
        
        internal_corners = corner_points.copy()
        internal_corner_slopes = [i[1] / i[0] for i in internal_corners]
        max_slope = np.max(internal_corner_slopes)
        expr = []
        log_convexity = []
        for i in range(1, len(internal_corner_slopes)-1):
            expr.append(internal_corner_slopes[i-1] + internal_corner_slopes[i+1] - 2*internal_corner_slopes[i])
        expr = [internal_corner_slopes[1] - 2*internal_corner_slopes[0]] + expr + [internal_corner_slopes[-2] - 2*internal_corner_slopes[-1]]

        df = pd.DataFrame({"x":[i[0] for i in internal_corners], "y": [i[1] for i in internal_corners], "slope=y/x": internal_corner_slopes, "convexity": expr})
        df["log(slope)"] = np.log(df["slope=y/x"])
        b_n = list(np.log(df["slope=y/x"]))
        jumps = [internal_corners[0][1]]
        for i in range(1, len(internal_corners)):
            jumps.append(internal_corners[i][1] - internal_corners[i-1][1])
        
        for i in range(1, len(b_n)-1):
            log_convexity.append(b_n[i-1] + b_n[i+1] - 2*b_n[i])  
        log_convexity = [b_n[1] - 2*b_n[0]] + list(log_convexity) + [b_n[len(b_n)-2] - 2*b_n[len(b_n)-3]]
        df["log-convexity"] = log_convexity
        df["normalized angle"] = np.arctan(df["slope=y/x"] / max_slope) * 180 / np.pi
        a_n = df["normalized angle"].to_numpy()
        a_n = np.concatenate([a_n, [0]])
        df["angular jump"] = a_n[1:] - a_n[:-1]
        df.sort_values("x").to_csv(f"./figures/{crystal_set_name}_external_points.csv", index=False)
        original_corners = corner_points.copy()
        # Plot staircase
        for i, corner_point in enumerate(corner_points):
            if i == 0:
                plt.plot([0, corner_point[0]], [0, 0], color="blue")
                plt.plot([corner_point[0], corner_point[0]], [0, corner_point[1]], color="blue")
            else:
                plt.plot([corner_points[i-1][0], corner_point[0]], [corner_points[i-1][1], corner_points[i-1][1]], color="blue")
                plt.plot([corner_point[0], corner_point[0]], [corner_points[i-1][1], corner_points[i][1]], color="blue")
                
        internal_corner_slopes = np.array(internal_corner_slopes)  
        internal_corners.sort(key=lambda x: x[0])  
        arg_for_SPF = np.argmax(list(df["angular jump"])[1:])
        internal_corner_for_SPF = internal_corners[1:][arg_for_SPF]
        adj_corner = [c for c in corner_points if c[1] == internal_corner_for_SPF[1]][0]
        print(f"next_internal_corner: {adj_corner}")
        print(f"internal_corner: {internal_corners[1:][arg_for_SPF]}")  
        SPF = list(df["slope=y/x"])[1:][arg_for_SPF]
        initial_SPF = SPF
        SPD = adj_corner[1] - SPF*adj_corner[0]
        SPD = max(SPD, 0)
        print(f"y = {np.round(SPF, 3)} x + {np.round(SPD, 3)}")
        line_function = lambda x: SPF * x + SPD
        #corner_points = stage1(corner_points)
        #corner_points = stage2(corner_points)
        SPB = [corner for corner in corner_points if line_function(corner[0]) < corner[1] and corner[0] >= adj_corner[0]][0][0]
        point_before_SPB = [corner for corner in corner_points if corner[0] < SPB][-1]
        print(f"Using an SPB of : {np.round(SPB, 4)}")
        potential_SPF = SPF
        
        
        if SPD > 0:
            while True:
                intersection_index = find_intersection(original_corners, line_function, (SPB, 0), return_index=True)
                print(f"previous line equation: y = {np.round(line_function(1) - line_function(0), 3)} x + {np.round(line_function(0), 3)}")
                if intersection_index == -1:
                    break
                c, d = original_corners[intersection_index]
                amount_to_shift = (d - line_function(c)) + line_function(0)
                line_function = lambda x: potential_SPF * x + amount_to_shift
                print(f"y = {np.round(potential_SPF, 3)} x + {np.round(amount_to_shift, 3)}")
                print(f"Intersects at {np.round((c,d), 3)}")
                SPD = line_function(0)
                SPF = potential_SPF
                
           
        highlighted_points = [point for point in corner_points if (point[0] == SPB and point[1] >= line_function(SPB)) or line_function(point[0]) == point[1]]  
        p1 = adj_corner
        points_to_consider_for_p2 = [point for point in corner_points if point[0] < p1[0]]
        potential_slope_and_intercepts = [find_line_function(p1, point, return_slope_and_intercept=True) for point in points_to_consider_for_p2]
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
            line_function = find_line_function(p1, (0,0))
        SPB = [corner[0] for corner in corner_points if line_function(corner[0]) < corner[1] and abs(line_function(corner[0]) - corner[1]) > 1e-15 ][0]
        highlighted_points = [point for point in corner_points if (point[0] == SPB and point[1] >= line_function(SPB)) or abs(line_function(point[0]) - point[1]) < 1e-10]  

        print("-----------------------------------")
        print(f"SPD: {SPD} kJ/mol")
        print(f"SPB: {SPB}")
        print(f"SPF: {SPF}")
        #xran = [0, 0.4]
        if zoomed:
            xran = [0, 2*SPB]
            yran = [0, np.max(fe_diffs[distances < 2*SPB])]
        else:
            d = np.max([i[1] for i in highlighted_points])
            xran = [0, min(4*SPB, np.max(distances))]
            yran = [0, min(np.max(fe_diffs[distances < 4*SPB]), 4*d)]
        #  Original line
        #plt.plot([0, SPB], [0, initial_SPF * SPB], color="red")
        #  Shifted line
        plt.plot([0, SPB], [SPD, line_function(SPB)], color='springgreen')
        plt.plot([SPB, SPB], [line_function(SPB), np.max(fe_diffs)], color='springgreen')
        plt.scatter([i[0] for i in highlighted_points], [i[1] for i in highlighted_points], c="red", s=110)

        plt.xlim(xran)
        plt.ylim(yran)
        pl.set_yticklabels(np.round(pl.get_yticks(), 2), size=30)
        pl.set_xticklabels(np.round(pl.get_xticks(), 3), size=30)
        z_suffix = "2SPB" if zoomed else "4SPB"
        plt.savefig(f"./figures/{crystal_set_name}_LE-vs-EMD_PDD100_{z_suffix}_angular_jump.png")
        plt.show()
        plt.close()
        


def plot_SPR_jarvis(dataset="/home/jonathan/PhD/Periodic-Set-Transformer_v2/Periodic-Set-Transformer-matbench/jarvis_dft_3d_pymatgen_structures_old", zoomed=False):
    properties = ["formation_energy_peratom", "optb88vdw_bandgap", "optb88vdw_total_energy", 
                "ehull", "mbj_bandgap", "bulk_modulus_kv", "shear_modulus_gv", 'magmom_oszicar', 
                'magmom_outcar', "slme", "spillage", "kpoint_length_unit",
                'epsx', 'epsy', 'epsz', 'mepsx', 'mepsy', 'mepsz',
                'dfpt_piezo_max_dij', 'dfpt_piezo_max_eij', "exfoliation_energy", "max_efg",
                "avg_elec_mass", "avg_hole_mass", "n-Seebeck", "n-powerfact",
                "p-Seebeck", "p-powerfact"]
    periodic_sets = pickle.load(open(dataset + "_ps", "rb"))
    _, targets = pickle.load(open(dataset, "rb"))
    
    for prop in properties:
        target_values = list(targets[prop])
        to_keep = [i for i in range(len(target_values)) if target_values[i] != "na"]
        print(f"{prop} Data size: {len(to_keep)}")
        if len(to_keep) > 10000:
            continue
        ps = [periodic_sets[i] for i in to_keep]
        le = [target_values[i] for i in to_keep]

        if os.path.exists(f"./data/jarvis_amds"):
            amds = pickle.load(open(f"./data/jarvis_{prop}_amds", "rb"))
        else:
            amds = [amd.AMD(p, k=100) for p in ps]
            pickle.dump(amds, open(f"./data/jarvis_{prop}_amds", "wb"))

        distances = amd.AMD_pdist(amds, low_memory=True)
        le = np.array(le)
        fe_diffs = pdist(le.reshape((-1, 1)))

        inds = np.argsort(distances)[:20000]
        d = np.sort(distances)
        samples_inds = [i for i in range(20000, len(distances)) if d[i] < 0.6 and i % 10000 == 0]
        inds = np.concatenate([inds, samples_inds])
                
        fe_diffs = fe_diffs[inds]
        m = len(ps)
        pairs = []
        for i in inds:
            pairs.append(dist_ind_to_pair_ind(m, i))

        if os.path.exists(f"./data/jarvis_pairs"):
            distances, fe_diffs = pickle.load(open(f"./data/jarvis_{prop}_pairs", "rb"))
        else:
            print(f"generating pairs for jarvis")
            distances = np.array([amd.EMD(amd.PDD(ps[i], k=100), amd.PDD(ps[j], k=100)) for i, j in pairs])
            fe_diffs = np.array([abs(le[i] - le[j]) for i,j in pairs])
            pickle.dump((distances, fe_diffs), open(f"./data/jarvis_{prop}_pairs", "wb"))

        plt.figure(figsize=(30,20))
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        pl = seaborn.scatterplot(x=distances, y=fe_diffs, color="black")
        pl.set_xlabel("L-inf based EMD on PDD100", fontsize=46)
        pl.set_ylabel(f"Absolute Difference in {prop}", fontsize=46)
                
        x = distances
        y = fe_diffs

        corner_points = []
        internal_corners = []
        for a, b in zip(x, y):
            point_qualifies = not np.any((x < a) & (y > b))
            if point_qualifies:
                corner_points.append((a,b))

        corner_points.sort(key=lambda x: x[0])
        for i in range(1, len(corner_points)):
            b = corner_points[i-1][1]
            a = corner_points[i][0]
            internal_corners.append((a,b))
        
        internal_corners = corner_points.copy()
        internal_corner_slopes = [i[1] / i[0] for i in internal_corners]
        max_slope = np.max(internal_corner_slopes)
        expr = []
        log_convexity = []
        for i in range(1, len(internal_corner_slopes)-1):
            expr.append(internal_corner_slopes[i-1] + internal_corner_slopes[i+1] - 2*internal_corner_slopes[i])
        expr = [internal_corner_slopes[1] - 2*internal_corner_slopes[0]] + expr + [internal_corner_slopes[-2] - 2*internal_corner_slopes[-1]]

        df = pd.DataFrame({"x":[i[0] for i in internal_corners], "y": [i[1] for i in internal_corners], "slope=y/x": internal_corner_slopes, "convexity": expr})
        df["log(slope)"] = np.log(df["slope=y/x"])
        b_n = list(np.log(df["slope=y/x"]))
        jumps = [internal_corners[0][1]]
        for i in range(1, len(internal_corners)):
            jumps.append(internal_corners[i][1] - internal_corners[i-1][1])
        
        for i in range(1, len(b_n)-1):
            log_convexity.append(b_n[i-1] + b_n[i+1] - 2*b_n[i])  
        log_convexity = [b_n[1] - 2*b_n[0]] + list(log_convexity) + [b_n[len(b_n)-2] - 2*b_n[len(b_n)-3]]
        df["log-convexity"] = log_convexity
        df["normalized angle"] = np.arctan(df["slope=y/x"] / max_slope) * 180 / np.pi
        a_n = df["normalized angle"].to_numpy()
        a_n = np.concatenate([a_n, [0]])
        df["angular jump"] = a_n[1:] - a_n[:-1]
        df.sort_values("x").to_csv(f"./figures/{prop}_external_points.csv", index=False)
        original_corners = corner_points.copy()
        # Plot staircase
        for i, corner_point in enumerate(corner_points):
            if i == 0:
                plt.plot([0, corner_point[0]], [0, 0], color="blue")
                plt.plot([corner_point[0], corner_point[0]], [0, corner_point[1]], color="blue")
            else:
                plt.plot([corner_points[i-1][0], corner_point[0]], [corner_points[i-1][1], corner_points[i-1][1]], color="blue")
                plt.plot([corner_point[0], corner_point[0]], [corner_points[i-1][1], corner_points[i][1]], color="blue")
                
        internal_corner_slopes = np.array(internal_corner_slopes)  
        internal_corners.sort(key=lambda x: x[0])  
        arg_for_SPF = np.argmax(list(df["angular jump"])[1:])
        internal_corner_for_SPF = internal_corners[1:][arg_for_SPF]
        adj_corner = [c for c in corner_points if c[1] == internal_corner_for_SPF[1]][0]
        print(f"next_internal_corner: {adj_corner}")
        print(f"internal_corner: {internal_corners[1:][arg_for_SPF]}")
        if adj_corner == (0,0):
            print(f"Skipping {prop}")
            continue
        SPF = list(df["slope=y/x"])[1:][arg_for_SPF]
        initial_SPF = SPF
        SPD = adj_corner[1] - SPF*adj_corner[0]
        SPD = max(SPD, 0)
        print(f"y = {np.round(SPF, 3)} x + {np.round(SPD, 3)}")
        line_function = lambda x: SPF * x + SPD
        #corner_points = stage1(corner_points)
        #corner_points = stage2(corner_points)
        SPB = [corner for corner in corner_points if line_function(corner[0]) < corner[1] and corner[0] >= adj_corner[0]]
        if len(SPB) == 0:
            SPB = 0
        else:
            SPB = SPB[0][0]
        point_before_SPB = [corner for corner in corner_points if corner[0] < SPB][-1]
        print(f"Using an SPB of : {np.round(SPB, 4)}")
        potential_SPF = SPF
        
        
        if SPD > 0:
            while True:
                intersection_index = find_intersection(original_corners, line_function, (SPB, 0), return_index=True)
                print(f"previous line equation: y = {np.round(line_function(1) - line_function(0), 3)} x + {np.round(line_function(0), 3)}")
                if intersection_index == -1:
                    break
                c, d = original_corners[intersection_index]
                amount_to_shift = (d - line_function(c)) + line_function(0)
                line_function = lambda x: potential_SPF * x + amount_to_shift
                print(f"y = {np.round(potential_SPF, 3)} x + {np.round(amount_to_shift, 3)}")
                print(f"Intersects at {np.round((c,d), 3)}")
                SPD = line_function(0)
                SPF = potential_SPF
                
           
        highlighted_points = [point for point in corner_points if (point[0] == SPB and point[1] >= line_function(SPB)) or line_function(point[0]) == point[1]]  
        p1 = adj_corner
        points_to_consider_for_p2 = [point for point in corner_points if point[0] < p1[0]]
        potential_slope_and_intercepts = [find_line_function(p1, point, return_slope_and_intercept=True) for point in points_to_consider_for_p2]
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
            line_function = find_line_function(p1, (0,0))
        SPB = [corner[0] for corner in corner_points if line_function(corner[0]) < corner[1] and abs(line_function(corner[0]) - corner[1]) > 1e-15 ][0]
        highlighted_points = [point for point in corner_points if (point[0] == SPB and point[1] >= line_function(SPB)) or abs(line_function(point[0]) - point[1]) < 1e-10]  

        print("-----------------------------------")
        print(f"SPD: {SPD} kJ/mol")
        print(f"SPB: {SPB}")
        print(f"SPF: {SPF}")
        #xran = [0, 0.4]
        if zoomed:
            xran = [0, 2*SPB]
            yran = [0, np.max(fe_diffs[distances < 2*SPB])]
        else:
            d = np.max([i[1] for i in highlighted_points])
            xran = [0, min(4*SPB, np.max(distances))]
            yran = [0, min(np.max(fe_diffs[distances < 4*SPB]), 4*d)]
        #  Original line
        #plt.plot([0, SPB], [0, initial_SPF * SPB], color="red")
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
        plt.show()
        plt.close()


def plot_dist_vs_density_v5(datasets=["./data/T0_Predicted_Structures.cif", "./data/T1_Predicted_Structures.cif", "./data/T2_Predicted_Structures.cif"], zoomed=False):
    for data_path in datasets:
        crystal_set_name = os.path.basename(data_path).split(".")[0].split("_")[0]
        ps = pickle.load(open(f"./data/{crystal_set_name}_ps", "rb"))
        if os.path.exists(f"./data/{crystal_set_name}_pymatgen_structures"):
            crystals = pickle.load(open(f"./data/{crystal_set_name}_pymatgen_structures", "rb"))
        else:
            crystals = CifParser(data_path).get_structures()
            pickle.dump(crystals, open(f"./data/{crystal_set_name}_pymatgen_structures", "wb"))
        
        densities = [c.density for c in crystals]
        used_names = set()
        unique_ps = []
        for p in ps:
            if p.name not in used_names:
                unique_ps.append(p)
                used_names.add(p.name)
                
        ps = unique_ps
        names = [p.name for p in ps]
        #le = [float(n.split("_")[0]) for n in names]
        if os.path.exists(f"./data/{crystal_set_name}_amds"):
            amds = pickle.load(open(f"./data/{crystal_set_name}_amds", "rb"))
        else:
            amds = [amd.AMD(p, k=100) for p in ps]
            pickle.dump(amds, open(f"./data/{crystal_set_name}_amds", "wb"))

        distances = amd.AMD_pdist(amds, low_memory=True)
        #le = np.array(le)
        fe_diffs = pdist(np.array(densities).reshape((-1, 1))) #pdist(le.reshape((-1, 1)))

        inds = np.argsort(distances)[:20000]
        d = np.sort(distances)
        samples_inds = [i for i in range(20000, len(distances)) if d[i] < 0.6 and i % 10000 == 0]
        inds = np.concatenate([inds, samples_inds])
                
        fe_diffs = fe_diffs[inds]
        m = len(ps)
        pairs = []
        for i in inds:
            pairs.append(dist_ind_to_pair_ind(m, i))

        if os.path.exists(f"./data/{crystal_set_name}_pairs"):
            distances, fe_diffs = pickle.load(open(f"./data/{crystal_set_name}_pairs", "rb"))
            fe_diffs = np.array([abs(crystals[i].density - crystals[j].density) for i,j in pairs])
        else:
            print(f"generating pairs for {crystal_set_name}")
            distances = np.array([amd.EMD(amd.PDD(ps[i], k=100), amd.PDD(ps[j], k=100)) for i, j in pairs])
            fe_diffs = np.array([abs(crystals[i].density - crystals[j].density) for i,j in pairs])
            pickle.dump((distances, fe_diffs), open(f"./data/{crystal_set_name}_pairs", "wb"))

        fe_diffs = np.array([abs(crystals[i].density - crystals[j].density) for i,j in pairs])
        plt.figure(figsize=(30,20))
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        pl = seaborn.scatterplot(x=distances, y=fe_diffs, color="black")
        pl.set_xlabel("L-inf based EMD on PDD100", fontsize=46)
        pl.set_ylabel("Absolute Difference in Density (g/cm^3)", fontsize=46)
        
    
        x = distances
        y = fe_diffs

        corner_points = []
        internal_corners = []
        for a, b in zip(x, y):
            point_qualifies = not np.any((x < a) & (y > b))
            if point_qualifies:
                corner_points.append((a,b))

        corner_points.sort(key=lambda x: x[0])
        for i in range(1, len(corner_points)):
            b = corner_points[i-1][1]
            a = corner_points[i][0]
            internal_corners.append((a,b))
        
        internal_corners = corner_points.copy()
        internal_corner_slopes = [i[1] / i[0] for i in internal_corners]
        max_slope = np.max(internal_corner_slopes)
        expr = []
        log_convexity = []
        for i in range(1, len(internal_corner_slopes)-1):
            expr.append(internal_corner_slopes[i-1] + internal_corner_slopes[i+1] - 2*internal_corner_slopes[i])
        expr = [internal_corner_slopes[1] - 2*internal_corner_slopes[0]] + expr + [internal_corner_slopes[-2] - 2*internal_corner_slopes[-1]]

        df = pd.DataFrame({"x":[i[0] for i in internal_corners], "y": [i[1] for i in internal_corners], "slope=y/x": internal_corner_slopes, "convexity": expr})
        df["log(slope)"] = np.log(df["slope=y/x"])
        b_n = list(np.log(df["slope=y/x"]))
        jumps = [internal_corners[0][1]]
        for i in range(1, len(internal_corners)):
            jumps.append(internal_corners[i][1] - internal_corners[i-1][1])
        
        for i in range(1, len(b_n)-1):
            log_convexity.append(b_n[i-1] + b_n[i+1] - 2*b_n[i])  
        log_convexity = [b_n[1] - 2*b_n[0]] + list(log_convexity) + [b_n[len(b_n)-2] - 2*b_n[len(b_n)-3]]
        df["log-convexity"] = log_convexity
        df["normalized angle"] = np.arctan(df["slope=y/x"] / max_slope) * 180 / np.pi
        a_n = df["normalized angle"].to_numpy()
        a_n = np.concatenate([a_n, [0]])
        df["angular jump"] = a_n[1:] - a_n[:-1]
        df.sort_values("x").to_csv(f"./figures/{crystal_set_name}_external_points.csv", index=False)
        original_corners = corner_points.copy()
        # Plot staircase
        for i, corner_point in enumerate(corner_points):
            if i == 0:
                plt.plot([0, corner_point[0]], [0, 0], color="blue")
                plt.plot([corner_point[0], corner_point[0]], [0, corner_point[1]], color="blue")
            else:
                plt.plot([corner_points[i-1][0], corner_point[0]], [corner_points[i-1][1], corner_points[i-1][1]], color="blue")
                plt.plot([corner_point[0], corner_point[0]], [corner_points[i-1][1], corner_points[i][1]], color="blue")
                
        internal_corner_slopes = np.array(internal_corner_slopes)  
        internal_corners.sort(key=lambda x: x[0])  
        arg_for_SPF = np.argmax(list(df["angular jump"])[1:])
        internal_corner_for_SPF = internal_corners[1:][arg_for_SPF]
        adj_corner = [c for c in corner_points if c[1] == internal_corner_for_SPF[1]][0]
        print(f"next_internal_corner: {adj_corner}")
        print(f"internal_corner: {internal_corners[1:][arg_for_SPF]}")  
        SPF = list(df["slope=y/x"])[1:][arg_for_SPF]
        initial_SPF = SPF
        SPD = adj_corner[1] - SPF*adj_corner[0]
        SPD = max(SPD, 0)
        print(f"y = {np.round(SPF, 3)} x + {np.round(SPD, 3)}")
        line_function = lambda x: SPF * x + SPD
        #corner_points = stage1(corner_points)
        #corner_points = stage2(corner_points)
        SPB = [corner for corner in corner_points if line_function(corner[0]) < corner[1] and corner[0] >= adj_corner[0]][0][0]
        point_before_SPB = [corner for corner in corner_points if corner[0] < SPB][-1]
        print(f"Using an SPB of : {np.round(SPB, 4)}")
        potential_SPF = SPF
        
        
        if SPD > 0:
            while True:
                intersection_index = find_intersection(original_corners, line_function, (SPB, 0), return_index=True)
                print(f"previous line equation: y = {np.round(line_function(1) - line_function(0), 3)} x + {np.round(line_function(0), 3)}")
                if intersection_index == -1:
                    break
                c, d = original_corners[intersection_index]
                amount_to_shift = (d - line_function(c)) + line_function(0)
                line_function = lambda x: potential_SPF * x + amount_to_shift
                print(f"y = {np.round(potential_SPF, 3)} x + {np.round(amount_to_shift, 3)}")
                print(f"Intersects at {np.round((c,d), 3)}")
                SPD = line_function(0)
                SPF = potential_SPF
                
           
        highlighted_points = [point for point in corner_points if (point[0] == SPB and point[1] >= line_function(SPB)) or line_function(point[0]) == point[1]]  
        p1 = adj_corner
        points_to_consider_for_p2 = [point for point in corner_points if point[0] < p1[0]]
        potential_slope_and_intercepts = [find_line_function(p1, point, return_slope_and_intercept=True) for point in points_to_consider_for_p2]
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
            line_function = find_line_function(p1, (0,0))
        SPB = [corner[0] for corner in corner_points if line_function(corner[0]) < corner[1] and abs(line_function(corner[0]) - corner[1]) > 1e-15 ][0]
        highlighted_points = [point for point in corner_points if (point[0] == SPB and point[1] >= line_function(SPB)) or abs(line_function(point[0]) - point[1]) < 1e-10]  

        print("-----------------------------------")
        print(f"SPD: {SPD} kJ/mol")
        print(f"SPB: {SPB}")
        print(f"SPF: {SPF}")
        #xran = [0, 0.4]
        if zoomed:
            xran = [0, 2*SPB]
            yran = [0, np.max(fe_diffs[distances < 2*SPB])]
        else:
            d = np.max([i[1] for i in highlighted_points])
            xran = [0, min(4*SPB, np.max(distances))]
            yran = [0, min(np.max(fe_diffs[distances < 4*SPB]), 4*d)]
        #  Original line
        #plt.plot([0, SPB], [0, initial_SPF * SPB], color="red")
        #  Shifted line
        plt.plot([0, SPB], [SPD, line_function(SPB)], color='springgreen')
        plt.plot([SPB, SPB], [line_function(SPB), np.max(fe_diffs)], color='springgreen')
        plt.scatter([i[0] for i in highlighted_points], [i[1] for i in highlighted_points], c="red", s=110)

        plt.xlim(xran)
        plt.ylim(yran)
        pl.set_yticklabels(np.round(pl.get_yticks(), 2), size=30)
        pl.set_xticklabels(np.round(pl.get_xticks(), 3), size=30)
        z_suffix = "2SPB" if zoomed else "4SPB"
        plt.savefig(f"./figures/{crystal_set_name}_Density-vs-EMD_PDD100_{z_suffix}_angular_jump.png")
        #plt.show()
        plt.close()



def plot_matbench():
    from matbench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False)
    tasks = [
        #mb.matbench_dielectric,
        #mb.matbench_log_gvrh,
        #mb.matbench_log_kvrh,
        #mb.matbench_phonons,
        #mb.matbench_mp_e_form,
        mb.matbench_mp_gap    
    ]
    prediction_files = {
        "matbench_dielectric": "/home/jonathan/Downloads/results_dielectric.json.gz",
        "matbench_log_gvrh": "/home/jonathan/Downloads/results_gvrh.json.gz",
        "matbench_log_kvrh": "/home/jonathan/Downloads/results_kvrh.json.gz",
        "matbench_phonons": "/home/jonathan/Downloads/results_phonons.json.gz",
        "matbench_mp_e_form": "/home/jonathan/Downloads/matbench_form.json.gz",
        "matbench_mp_gap": "/home/jonathan/Downloads/matbench_band_gap.json.gz"
    }
    for task in tasks:
        task.load()
        task_name = task.dataset_name
        print(task_name)
        with open(prediction_files[task_name]) as f:
            import json
            predictions = json.load(f)
        all_predictions = []
        all_ground_truth = []
        print("Starting folds..")
        for fold in task.folds:
            print(fold)
            _, ground_truth = task.get_test_data(fold, include_target=True)
            ground_truth = np.array(ground_truth)
            fold_predictions = predictions['tasks'][task_name]['results']["fold_" + str(fold)]["data"].values()
            fold_predictions = np.array(list(fold_predictions))
            all_predictions.append(fold_predictions)
            all_ground_truth.append(ground_truth) 
        print("concatenating..")
        all_predictions = np.concatenate(all_predictions)
        all_ground_truth = np.concatenate(all_ground_truth)
        print(all_predictions.shape)
        print(all_ground_truth.shape)
        plot_truth_vs_prediction(all_predictions, all_ground_truth)
