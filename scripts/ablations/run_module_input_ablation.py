#!/usr/bin/env python3
"""Standalone clean module/input ablation benchmark for fullerene gap prediction.

The script reads only the label CSV and GJF geometry tree, or the validated raw
dataset cache produced by the current multiseed benchmark.  It does not import
another model script.  Every configuration uses the same validated 7,035
samples, split definitions, training settings, checkpoint rule, eV target unit,
and five-seed protocol as the current Proposed-model benchmark.  Post-processing
is disabled and raw model predictions are the only primary paper results.
"""

import argparse
import gc
import hashlib
import os
from pathlib import Path
import random
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import networkx as nx
import scipy.sparse as sp
import scipy.linalg
from scipy.sparse.linalg import eigsh
from scipy.spatial import ConvexHull

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# =========================
# 🔧 Global Configuration
# =========================
# ⚠️ Ensure these match your server paths
ROOT_FOLDER = "data/structures"
DATA_FILE = "data/dataset.csv"
# A clean cache must never reuse samples created by the legacy path matcher.
CACHE_SCHEMA_VERSION = 4
CACHE_FILE = os.path.join("results", "ablations", "validated_raw_dataset_cache.pt")

# Gaussian orbital-energy differences in sorted_fullerene_data.csv are in Hartree.
HARTREE_TO_EV = 27.211386245988
TARGET_UNIT = "eV"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 7e-4
MAX_EPOCHS = 420
PATIENCE = 70
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.08
NOISE_STD = 0.0
GROUP_VAL_SIZES = (58, 60)
BASELINE_MIN_RES_STD = 0.003 * HARTREE_TO_EV
RESIDUAL_CLIP_Q = (0.002, 0.998)
RESIDUAL_LOWER_MARGIN_SIGMA = 0.15
RESIDUAL_UPPER_MARGIN_SIGMA = 0.60
RESIDUAL_SCALE_CANDIDATES = (1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35)
CALIBRATION_SLOPE_CLAMP = (0.94, 1.34)
CALIBRATION_INTERCEPT_CLAMP = 0.006 * HARTREE_TO_EV
NEGATIVE_VAL_PENALTY = 0.12
TAIL_BLEND_CANDIDATES = (0.00, 0.08, 0.12, 0.16, 0.20, 0.24)
TAIL_BLEND_N_START = 72
TAIL_BLEND_N_FULL = 84
TAIL_SCALE_BONUS = 0.10
TAIL_CAL_SLOPE_SHRINK = 0.55
PRED_RANGE_GUARD_Q = (0.001, 0.999)
PRED_RANGE_GUARD_MARGIN = 0.60
HIGH_TAIL_ANCHOR_Q_CANDIDATES = (0.95, 0.98)
HIGH_TAIL_GAIN_CANDIDATES = (0.00, 0.05, 0.10, 0.15)
HIGH_GAP_SHIFT_CANDIDATES = tuple(x * HARTREE_TO_EV for x in (0.0, 0.001, 0.002, 0.003))
TAIL_SCORE_Q = 0.90
TAIL_R2_SCORE_WEIGHT = 0.0
TAIL_BIAS_SCORE_WEIGHT = 0.0
USE_SIZE_BALANCED_LOSS = True
SIZE_WEIGHT_POWER = 0.30
SIZE_WEIGHT_CLAMP = (0.6, 2.4)
TARGET_TAIL_WEIGHT = 0.26
TARGET_TAIL_CLAMP = 3.0
HUBER_MIX = 0.30
STD_MATCH_WEIGHT = 0.10
CORR_LOSS_WEIGHT = 0.06
HIGH_GAP_UNDER_WEIGHT = 0.08
HIGH_GAP_LOSS_Q = 0.80
TOP_STD_MATCH_WEIGHT = 0.06
TOP_STD_MATCH_Q = 0.80
HIGH_GAP_MEAN_UNDER_WEIGHT = 0.0
TOP_RANGE_MATCH_WEIGHT = 0.0
# Match the current multiseed Proposed benchmark: checkpoint selection and all
# reported ablation metrics use unmodified raw model predictions.
ENABLE_POSTPROCESSING = False
POSTPROCESSING_MODE = "disabled"
PREPROCESS_MAX_WORKERS = 4
PREPROCESS_PROGRESS_STEP = 250

# Innovation Config
NUM_CLUSTERS = 3
USE_SAM = True
SAM_RHO = 0.05
USE_LAPPE_AUG = False
LAP_PE_DIM =0

# Model Architecture (DimeNetLite)
HIDDEN_DIM = 192
NUM_LAYERS = 5
RBF_DIM = 32
ANGLE_BASIS_K = 16
MIN_DIST = 0.1

SEED = 2024
BASE_SAVE_DIR = "clean_module_input_ablation"
DEFAULT_SEEDS = (2024, 2025, 2026, 2027, 2028)
EXPECTED_SPLIT_COUNTS = {
    "random": {"train": 5628, "val": 703, "test": 704},
    "group": {"train": 2752, "val": 3017, "test": 1266},
}

# The default configuration reproduces the corrected proposed model.  Every
# ablation changes exactly the switches declared in EXPERIMENT_CONFIGS.
DEFAULT_ABLATION_CONFIG = {
    "baseline_only": False,
    "use_size_baseline": True,
    "use_global_descriptors": True,
    "use_size_descriptors": True,
    "use_topology_inputs": True,
    "use_geometry_inputs": True,
    "use_directional_geometry": True,
    "use_auxiliary_loss": True,
    "use_size_balanced_weighting": True,
    "use_sam": True,
}
ACTIVE_ABLATION_CONFIG = dict(DEFAULT_ABLATION_CONFIG)

MODULE_CONFIGS = {
    "full": {},
    "no_size_baseline": {"use_size_baseline": False},
    "no_global_descriptors": {"use_global_descriptors": False},
    "no_directional_geometry": {"use_directional_geometry": False},
    "no_auxiliary_loss": {"use_auxiliary_loss": False},
    "no_size_balanced_weighting": {"use_size_balanced_weighting": False},
    "no_sam": {"use_sam": False},
}

INPUT_CONFIGS = {
    # size_only is evaluated as the Ridge size baseline without a GNN.
    "size_only": {
        "baseline_only": True,
        "use_global_descriptors": False,
        "use_topology_inputs": False,
        "use_geometry_inputs": False,
        "use_directional_geometry": False,
        "use_auxiliary_loss": False,
        "use_size_balanced_weighting": False,
        "use_sam": False,
    },
    "size_topology": {
        "use_geometry_inputs": False,
        "use_directional_geometry": False,
    },
    "size_geometry": {
        "use_topology_inputs": False,
    },
    "full": {},
}


def set_ablation_config(overrides):
    ACTIVE_ABLATION_CONFIG.clear()
    ACTIVE_ABLATION_CONFIG.update(DEFAULT_ABLATION_CONFIG)
    ACTIVE_ABLATION_CONFIG.update(dict(overrides))


def ablation_enabled(name):
    return bool(ACTIVE_ABLATION_CONFIG.get(name, DEFAULT_ABLATION_CONFIG[name]))


# =========================
# 🛠️ Basic Tools
# =========================
def seed_everything(seed=2024):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except:
        return torch.load(path, map_location=map_location)


def configure_worker_runtime():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def apply_lappe_aug(node_feat: torch.Tensor, lap_dim: int) -> torch.Tensor:
    """LapPE Sign Flipping Augmentation"""
    if lap_dim <= 0: return node_feat
    pe = node_feat[:, -lap_dim:]
    signs = (torch.randint(0, 2, (1, lap_dim), device=pe.device, dtype=pe.dtype) * 2 - 1)
    node_feat[:, -lap_dim:] = pe * signs
    return node_feat


def build_baseline_features(atom_count: int) -> np.ndarray:
    n = float(atom_count)
    return np.array([
        1.0 / n,
        1.0 / np.sqrt(n),
        1.0 / (n ** (2.0 / 3.0)),
        np.log(n + 1.0) / n,
    ], dtype=np.float32)


def split_for_atom_count(atom_count: int) -> str:
    if 20 <= atom_count <= 56:
        return "train"
    if atom_count in GROUP_VAL_SIZES:
        return "val"
    if 70 <= atom_count <= 100:
        return "test"
    raise ValueError(f"C{atom_count} is outside the declared strict group protocol.")


def build_baseline_features_from_item(item) -> np.ndarray:
    return build_baseline_features(int(item["intended_N"]))


def build_structural_global_feature(item, cluster_label: int) -> np.ndarray:
    """Build only descriptors available from the validated molecular graph/geometry.

    The KMeans label is assigned by a model fitted on the training split only.
    No DFT electronic properties (for example HOMO, LUMO, or dipole moment) are used.
    """
    atom_count = float(item["intended_N"])
    size_part = np.array([
        1.0 / atom_count,
        1.0 / np.sqrt(atom_count),
        float(np.log1p(atom_count) / atom_count),
    ], dtype=np.float32)
    topology_part = np.array([float(item["ipr_penalty"])], dtype=np.float32)
    geometry_part = np.asarray(item["phys_feat"], dtype=np.float32)

    if not ablation_enabled("use_size_descriptors"):
        size_part = np.zeros_like(size_part)
    if not ablation_enabled("use_topology_inputs"):
        topology_part = np.zeros_like(topology_part)
    if not ablation_enabled("use_geometry_inputs"):
        geometry_part = np.zeros_like(geometry_part)

    base = np.concatenate([size_part, topology_part, geometry_part])
    cluster_one_hot = np.zeros(NUM_CLUSTERS, dtype=np.float32)
    if ablation_enabled("use_geometry_inputs"):
        cluster_one_hot[int(cluster_label)] = 1.0
    result = np.concatenate([base, cluster_one_hot]).astype(np.float32)
    if not ablation_enabled("use_global_descriptors"):
        result[:] = 0.0
    return result


def build_augmented_global_feature(item, base_y: float, cluster_label: int) -> np.ndarray:
    raw_g = build_structural_global_feature(item, cluster_label)
    atom_count = float(item["intended_N"])
    extra = np.array([base_y, base_y * np.log1p(atom_count)], dtype=np.float32)
    if not ablation_enabled("use_global_descriptors") or not ablation_enabled("use_size_baseline"):
        extra[:] = 0.0
    return np.concatenate([raw_g, extra], axis=0)


def restore_target_value(target_norm, atom_count, base_model, global_mu_res, global_sigma_res):
    base_y = (
        float(base_model.predict(build_baseline_features(atom_count).reshape(1, -1))[0])
        if base_model is not None else 0.0
    )
    raw_res = float(target_norm * global_sigma_res + global_mu_res)
    return base_y + raw_res


def transform_residual(raw_res, res_clip_low=None, res_clip_high=None, residual_scale=1.0, use_guard=False):
    raw_res = float(raw_res)
    if use_guard and res_clip_low is not None and res_clip_high is not None:
        raw_res = float(np.clip(raw_res, res_clip_low, res_clip_high))
    return raw_res * residual_scale


def restore_prediction_value(pred_norm, atom_count, base_model, global_mu_res, global_sigma_res,
                             res_clip_low=None, res_clip_high=None, residual_scale=1.0,
                             cal_slope=1.0, cal_intercept=0.0, use_guard=False):
    base_y = (
        float(base_model.predict(build_baseline_features(atom_count).reshape(1, -1))[0])
        if base_model is not None else 0.0
    )
    raw_res = float(pred_norm * global_sigma_res + global_mu_res)
    pred = base_y + transform_residual(
        raw_res,
        res_clip_low=res_clip_low,
        res_clip_high=res_clip_high,
        residual_scale=residual_scale,
        use_guard=use_guard,
    )
    pred = cal_slope * pred + cal_intercept
    return max(pred, 0.0)


# =========================
# ⚡ Optimizer & Early Stopping
# =========================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = [p.grad.norm(p=2).to(shared_device).view(-1) for group in self.param_groups for p in group["params"] if
                 p.grad is not None]
        if not norms: return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.cat(norms), p=2)


class EarlyStopping:
    def __init__(self, patience=20, delta=0, mode='max', verbose=False, path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose and self.counter % 10 == 0:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        torch.save(model.state_dict(), self.path)


# =========================
# 🧬 Feature Extraction
# =========================
class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=3.0, n_gaussians=32):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        self.coeff = -0.5 / ((stop - start) / (n_gaussians - 1)) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist_1d):
        x = dist_1d.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(x, 2))


class AngleBasis(nn.Module):
    def __init__(self, K=8):
        super().__init__()
        self.K = K

    def forward(self, theta):
        m = torch.arange(1, self.K + 1, device=theta.device, dtype=theta.dtype).view(1, -1)
        t = theta.view(-1, 1) * m
        return torch.cat([torch.sin(t), torch.cos(t)], dim=1)


def compute_lap_pe(adj_matrix, k=8):
    try:
        N = adj_matrix.shape[0]
        G = nx.from_numpy_array(adj_matrix)
        L = nx.normalized_laplacian_matrix(G).astype(np.float32)
        k_eff = min(k + 1, N - 1)
        evals, evecs = eigsh(L, k=k_eff, which='SM', tol=1e-2)
        pe = evecs[:, 1:k + 1]
        if pe.shape[1] < k:
            pad = np.zeros((N, k - pe.shape[1]), dtype=np.float32)
            pe = np.hstack([pe, pad])
        return torch.tensor(pe, dtype=torch.float32)
    except Exception as e:
        return torch.zeros((adj_matrix.shape[0], k), dtype=torch.float32)


def calculate_comprehensive_physics(adj_matrix, coords, num_atoms):
    phys_feat = []
    try:
        evals = scipy.linalg.eigvalsh(adj_matrix)
        evals = np.sort(evals)
        n_occ = num_atoms // 2
        gap_est = evals[n_occ] - evals[n_occ - 1] if n_occ < len(evals) else 0.0
        phys_feat.extend([np.sum(evals[:n_occ]) * 2 / num_atoms, np.std(evals), gap_est])
    except:
        phys_feat.extend([0.0, 0.0, 0.0])

    try:
        hull = ConvexHull(coords)
        vol, area = hull.volume, hull.area
        d_ipq = 1.0 - (36.0 * np.pi * (vol ** 2)) / (area ** 3 + 1e-8)
        centroid = np.mean(coords, axis=0)
        radii = np.linalg.norm(coords - centroid, axis=1) # 这里用 radii
        inertia = np.dot((coords - centroid).T, (coords - centroid))
        evals_I = np.linalg.eigvalsh(inertia)
        asphericity = (evals_I[-1] - evals_I[0]) / (evals_I[-1] + 1e-8)

        # 核心修正：消除尺寸依赖 (Extensive -> Intensive)
        vol_norm = vol / num_atoms
        area_norm = area / (num_atoms ** (2 / 3))  # 面积随 N^(2/3) 缩放

        phys_feat.extend([np.mean(radii), np.std(radii), asphericity, vol_norm, area_norm, d_ipq])
    except:
        phys_feat.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return phys_feat


def calculate_topology_features(graph, node_ring_features, edge_ring_tensor):
    feats = []
    try:
        degrees = np.array([deg for _, deg in graph.degree()], dtype=np.float32)
        feats.extend([
            float(degrees.mean()),
            float(degrees.std()),
            float(np.mean(np.abs(degrees - 3.0))),
            float(np.mean(degrees != 3.0)),
        ])
    except Exception:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    try:
        clustering = np.array(list(nx.clustering(graph).values()), dtype=np.float32)
        feats.extend([float(clustering.mean()), float(clustering.std())])
    except Exception:
        feats.extend([0.0, 0.0])

    try:
        if nx.is_connected(graph):
            component = graph
        else:
            largest_nodes = max(nx.connected_components(graph), key=len)
            component = graph.subgraph(largest_nodes).copy()
        feats.extend([
            float(nx.average_shortest_path_length(component)),
            float(nx.diameter(component)),
        ])
    except Exception:
        feats.extend([0.0, 0.0])

    try:
        node_np = node_ring_features.detach().cpu().numpy()
        n5 = node_np[:, 0]
        n6 = node_np[:, 1]
        feats.extend([
            float(n5.mean()), float(n5.std()),
            float(n6.mean()), float(n6.std()),
        ])
    except Exception:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    try:
        er = edge_ring_tensor.detach().cpu().numpy()
        if er.ndim == 3 and er.shape[-1] >= 4:
            upper = np.triu(np.ones(er.shape[:2], dtype=bool), k=1)
            active = er[upper]
            edge_mask = active.sum(axis=1) > 0
            active = active[edge_mask]
            if active.size:
                ratios = active[:, :4].mean(axis=0)
                feats.extend([float(x) for x in ratios])
            else:
                feats.extend([0.0, 0.0, 0.0, 0.0])
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])
    except Exception:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    return np.nan_to_num(np.asarray(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


class MatrixFullereneFeature:
    def __init__(self, graph):
        self.graph = graph
        self.ring_types = [5, 6]
        self.one_hot_dim = len(self.ring_types) + 1

    def _get_ring_one_hot(self, ring_size):
        vec = np.zeros(self.one_hot_dim, dtype=np.float32)
        if ring_size in self.ring_types:
            idx = self.ring_types.index(ring_size)
            vec[idx] = 1.0
        else:
            vec[-1] = 1.0
        return vec

    def extract_features(self):
        """Extract facial rings from a planar fullerene graph.

        A minimum cycle basis is not a set of molecular faces.  For a connected
        planar fullerene, traversing the embedding gives every face exactly once
        and allows us to validate the required 12 pentagons and N/2-10 hexagons.
        """
        n_nodes = self.graph.number_of_nodes()
        if not nx.is_connected(self.graph):
            raise ValueError("reconstructed graph is disconnected")
        if any(degree != 3 for _, degree in self.graph.degree()):
            raise ValueError("reconstructed graph is not three-regular")
        if self.graph.number_of_edges() != 3 * n_nodes // 2:
            raise ValueError("reconstructed graph has an invalid number of edges")

        is_planar, embedding = nx.check_planarity(self.graph)
        if not is_planar:
            raise ValueError("reconstructed graph is not planar")

        visited_half_edges = set()
        faces = []
        for u, v in embedding.edges():
            if (u, v) not in visited_half_edges:
                face = embedding.traverse_face(u, v, visited_half_edges)
                if len(face) >= 3:
                    faces.append(face)

        expected_faces = n_nodes // 2 + 2
        expected_hexagons = n_nodes // 2 - 10
        face_sizes = [len(face) for face in faces]
        if len(faces) != expected_faces:
            raise ValueError(f"expected {expected_faces} faces, found {len(faces)}")
        if any(size not in self.ring_types for size in face_sizes):
            raise ValueError("face set contains a non-5/6-membered ring")
        if face_sizes.count(5) != 12 or face_sizes.count(6) != expected_hexagons:
            raise ValueError("face counts do not satisfy fullerene topology")

        edge_to_faces = {}
        node_to_faces = {n: [] for n in self.graph.nodes()}
        for face in faces:
            face_size = len(face)
            for i, u in enumerate(face):
                v = face[(i + 1) % face_size]
                edge_to_faces.setdefault(tuple(sorted((u, v))), []).append(face_size)
                node_to_faces[u].append(face_size)

        nodes_list = sorted(list(self.graph.nodes()))
        node_map = {n: i for i, n in enumerate(nodes_list)}
        node_features = []

        for node in nodes_list:
            rings = node_to_faces[node]
            n5 = rings.count(5)
            n6 = rings.count(6)
            if n5 + n6 != 3:
                raise ValueError("node is not incident to exactly three fullerene faces")
            node_features.append([n5 / 3.0, n6 / 3.0])

        N = len(nodes_list)
        edge_tensor = torch.zeros((N, N, 4), dtype=torch.float32)

        for u, v in self.graph.edges():
            if u not in node_map or v not in node_map: continue
            idx_u, idx_v = node_map[u], node_map[v]
            rings = edge_to_faces.get(tuple(sorted((u, v))), [])

            feat = np.zeros(4, dtype=np.float32)
            if len(rings) == 2:
                r1, r2 = rings[0], rings[1]
                if (r1 == 5 and r2 == 5):
                    feat[0] = 1.0
                elif (r1 == 5 and r2 == 6) or (r1 == 6 and r2 == 5):
                    feat[1] = 1.0
                elif (r1 == 6 and r2 == 6):
                    feat[2] = 1.0
                else:
                    feat[3] = 1.0
            else:
                raise ValueError("edge is not incident to exactly two fullerene faces")

            edge_tensor[idx_u, idx_v] = torch.tensor(feat)
            edge_tensor[idx_v, idx_u] = torch.tensor(feat)
            # ... 上面是 edge_tensor[idx_v, idx_u] = torch.tensor(feat)

         # IPR penalty: the number of edges shared by two pentagons.
        adj_pentagons_count = 0
        for edge_u, edge_v in self.graph.edges():
            rings = edge_to_faces.get(tuple(sorted((edge_u, edge_v))), [])
            if rings.count(5) == 2:
                adj_pentagons_count += 1
        self.adj_pentagons = float(adj_pentagons_count)

        return torch.tensor(node_features, dtype=torch.float32), edge_tensor
# =========================
# 🔄 GJF Parsing (Enhanced Version)
# =========================
def get_coords_robust(path):
    """
    智能读取 .gjf 文件坐标 (移植自 generate_features.py)
    逻辑：寻找 '电荷 多重度' (如 0 1) 行，其后紧接着就是坐标。
    """
    try:
        with open(path, 'r', errors='ignore') as f:
            lines = [line.strip() for line in f.readlines()]

        start_idx = -1

        # 遍历每一行，寻找 "整数 整数" 模式 (Charge Multiplicity)
        for i, line in enumerate(lines):
            parts = line.split()
            # 电荷和多重度通常是两个整数
            if len(parts) == 2:
                try:
                    c = int(parts[0])
                    m = int(parts[1])

                    # 关键验证：检查下一行是否像坐标 (包含浮点数)
                    if i + 1 < len(lines):
                        next_line_parts = lines[i + 1].split()
                        # 坐标行通常至少有4列 (原子 x y z) 或 5列 (原子 0 x y z)
                        if len(next_line_parts) >= 4:
                            float(next_line_parts[-1])
                            float(next_line_parts[-2])
                            float(next_line_parts[-3])
                            # 验证通过！找到坐标起始位
                            start_idx = i + 1
                            break
                except ValueError:
                    continue

        if start_idx == -1:
            return np.array([])  # 未找到坐标

        # 开始读取坐标
        coords = []
        for i in range(start_idx, len(lines)):
            line = lines[i]
            if not line: break  # 遇到空行停止
            parts = line.split()
            # 兼容标准格式: "C x y z" 或 "C 0 x y z"
            if len(parts) >= 4:
                try:
                    # 取最后三列作为 x, y, z
                    x = float(parts[-3])
                    y = float(parts[-2])
                    z = float(parts[-1])
                    coords.append([x, y, z])
                except ValueError:
                    continue

        return np.array(coords, dtype=np.float32)

    except Exception:
        return np.array([])


def _parse_numeric_token(token):
    """Return an integer represented by a directory token such as '60' or 'C60'."""
    match = re.fullmatch(r"(?:C)?(\d+)", str(token), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_fullerene_file_key(full_path):
    """Read exactly one (atom_count, isomer_id) key from the directory layout.

    Supported layout: Organized_Fullerenes/<atom_count>/<split>/<isomer_id>/file.gjf.
    The file name is intentionally ignored so that numeric text in a file name cannot
    corrupt the atom-count or isomer-ID key.
    """
    rel_parts = os.path.normpath(os.path.relpath(full_path, ROOT_FOLDER)).split(os.sep)
    dir_parts = rel_parts[:-1]
    if not dir_parts:
        raise ValueError(f"file is not below ROOT_FOLDER: {full_path}")

    atom_count = _parse_numeric_token(dir_parts[0])
    if atom_count is None:
        raise ValueError(f"cannot parse atom-count directory in {full_path}")

    split_tokens = {"train", "val", "valid", "validation", "test"}
    start = 1
    if len(dir_parts) > 1 and dir_parts[1].lower() in split_tokens:
        start = 2
    if len(dir_parts) <= start:
        match = re.fullmatch(rf"C{atom_count}_(\d+)\.gjf", rel_parts[-1], flags=re.IGNORECASE)
        if len(dir_parts) == 1 and match:
            return atom_count, int(match.group(1))
        raise ValueError(f"cannot parse isomer-ID directory in {full_path}")

    isomer_id = _parse_numeric_token(dir_parts[start])
    if isomer_id is None:
        raise ValueError(f"cannot parse isomer-ID directory in {full_path}")
    return atom_count, isomer_id


def process_single_file(args):
    path, target_gap, intended_n, isomer_id = args
    try:
        configure_worker_runtime()
        coords = get_coords_robust(path)

        if len(coords) < 10:
            raise ValueError(f"fewer than 10 atomic coordinates were parsed from {path}")

        num = len(coords)
        if num != int(intended_n):
            raise ValueError(
                f"coordinate atom count ({num}) does not match CSV AtomCount ({intended_n}) for {path}"
            )
        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(coords)
        _, indices = nbrs.kneighbors(coords)

        adj_mat = np.zeros((num, num))
        G_nx = nx.Graph()
        G_nx.add_nodes_from(range(num))

        for i in range(num):
            for n_idx in indices[i, 1:]:
                adj_mat[i, n_idx] = 1.0
                adj_mat[n_idx, i] = 1.0
                G_nx.add_edge(i, int(n_idx))

        extractor = MatrixFullereneFeature(G_nx)
        n_feat_topo, e_ring_dense = extractor.extract_features()
        phys_feat = calculate_comprehensive_physics(adj_mat, coords, num)
        topo_feat = calculate_topology_features(G_nx, n_feat_topo, e_ring_dense)

        # ==========================================
        # 🎯 核心修改：用局部相对起伏替代绝对尺寸
        # ==========================================
        centroid = np.mean(coords, axis=0)
        # 算出所有原子的绝对半径
        radii_abs = np.linalg.norm(coords - centroid, axis=1)
        # 减去平均半径！彻底消除尺寸依赖
        node_radii_rel = (radii_abs - np.mean(radii_abs)).reshape(-1, 1)
        node_radii_tensor = torch.tensor(node_radii_rel, dtype=torch.float32)

        ipr_penalty = extractor.adj_pentagons  # 接出相邻五元环数量

        # 拼接拓扑特征与物理相对起伏
        node_feat = torch.cat([n_feat_topo, node_radii_tensor], dim=1)

        return {
            "node_feat": node_feat,
            "coords": torch.tensor(coords),
            "adj": torch.tensor(adj_mat),
            "e_ring": e_ring_dense,
            "y_target": float(target_gap),
            "phys_feat": phys_feat,
            "topo_feat": topo_feat,
            "ipr_penalty": ipr_penalty,
            "N": int(num),
            "intended_N": int(intended_n),
            "sample_id": int(isomer_id),
            "source_path": str(path),
            "unit": TARGET_UNIT,
        }
    except Exception as e:
        return {"error": str(e), "source_path": str(path)}


class FastDataset(Dataset):
    def __init__(self, data_list, is_train=False):
        self.data_list = data_list
        self.is_train = is_train

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        it = self.data_list[idx]
        out = dict(it)

        # 深拷贝坐标和节点特征，以免修改原始缓存数据
        out["coords"] = out["coords"].clone()
        out["node_feat"] = out["node_feat"].clone()

        if self.is_train:
            # 去除 LapPE 增强逻辑，只保留坐标噪声增强
            noise = torch.randn_like(out["coords"]) * NOISE_STD
            out["coords"] += noise

        return out
def collate_graphs(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None

    node_feats, coords, batch_node = [], [], []
    edge_src, edge_dst, edge_ring, edge_dist = [], [], [], []
    triplet_ea, triplet_eb = [], []
    y, global_feats, Ns, sample_weights, base_feats = [], [], [], [], []

    node_offset = 0
    edge_offset = 0

    for g_idx, item in enumerate(batch):
        nf = item["node_feat"]
        pos = item["coords"]
        adj = item["adj"]
        er = item["e_ring"]
        N = nf.shape[0]

        node_feats.append(nf)
        coords.append(pos)
        batch_node.append(torch.full((N,), g_idx, dtype=torch.long))
        y.append(item["y_target"])
        global_feats.append(item["g"])
        # CSV AtomCount is the authoritative cage size.  Equality with the
        # parsed geometry size is validated before any split is created.
        Ns.append(item["intended_N"])
        sample_weights.append(item.get("sample_weight", 1.0))
        base_feats.append(item.get("base_feat", torch.tensor(build_baseline_features_from_item(item))))

        src, dst = torch.where(adj > 0)
        mask = src != dst
        src, dst = src[mask], dst[mask]
        E = src.numel()

        if E == 0:
            node_offset += N
            continue

        edge_src.append(src + node_offset)
        edge_dst.append(dst + node_offset)
        edge_ring.append(er[src, dst, :])

        dist = torch.norm(pos[dst] - pos[src], dim=1).clamp(min=MIN_DIST)
        edge_dist.append(dist)

        local_src, local_dst = src.tolist(), dst.tolist()
        edge_map = {(local_src[i], local_dst[i]): i for i in range(E)}

        adj_np = adj.numpy()
        for i in range(N):
            neigh = np.where(adj_np[i])[0]
            if len(neigh) < 2: continue
            for j in neigh:
                for k in neigh:
                    if j == k: continue
                    ea = edge_map.get((i, j))
                    eb = edge_map.get((i, k))
                    if ea is not None and eb is not None:
                        triplet_ea.append(ea + edge_offset)
                        triplet_eb.append(eb + edge_offset)

        node_offset += N
        edge_offset += E

    if not edge_src: return None

    return {
        "x": torch.cat(node_feats, dim=0),
        "pos": torch.cat(coords, dim=0),
        "batch_node": torch.cat(batch_node, dim=0),
        "edge_src": torch.cat(edge_src, dim=0),
        "edge_dst": torch.cat(edge_dst, dim=0),
        "edge_ring": torch.cat(edge_ring, dim=0),
        "edge_dist": torch.cat(edge_dist, dim=0),
        "triplet_ea": torch.tensor(triplet_ea, dtype=torch.long),
        "triplet_eb": torch.tensor(triplet_eb, dtype=torch.long),
        "y": torch.tensor(y, dtype=torch.float32),
        "g": torch.stack(global_feats, dim=0),
        "N": torch.tensor(Ns),
        "sample_weight": torch.tensor(sample_weights, dtype=torch.float32),
        "base_feat": torch.stack(base_feats, dim=0).float()
    }


# =========================
# 🧠 DimeNetLite Model
# =========================
class DimeNetLite(nn.Module):
    def __init__(self, node_in, edge_ring_in, global_dim, hidden_dim=128, rbf_dim=32, angle_k=16, num_layers=4,
                 dropout=0.1, use_directional_geometry=True):
        super().__init__()
        self.hidden = hidden_dim
        self.use_directional_geometry = bool(use_directional_geometry)
        self.rbf = GaussianSmearing(start=0.5, stop=3.0, n_gaussians=rbf_dim)
        self.ang = AngleBasis(K=angle_k)
        self.node_emb = nn.Linear(node_in, hidden_dim)
        self.edge_emb = nn.Sequential(nn.Linear(edge_ring_in + rbf_dim, hidden_dim), nn.SiLU(),
                                      nn.Linear(hidden_dim, hidden_dim))
        self.triplet_mlp = nn.Sequential(nn.Linear(hidden_dim + 2 * rbf_dim + 2 * angle_k, hidden_dim), nn.SiLU(),
                                         nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)
        self.readout = nn.Sequential(nn.Linear(hidden_dim * 2 + global_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                     nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, data):
        x, pos, batch_node = data["x"], data["pos"], data["batch_node"]
        edge_src, edge_dst = data["edge_src"], data["edge_dst"]

        h = self.node_emb(x)
        rbf_e = self.rbf(data["edge_dist"])
        if not self.use_directional_geometry:
            rbf_e = torch.zeros_like(rbf_e)
        e_feat = torch.cat([data["edge_ring"], rbf_e], dim=1)
        m = self.edge_emb(e_feat)

        for ln in self.layers:
            agg = torch.zeros_like(h)
            agg.index_add_(0, edge_dst, m)
            h = ln(h + self.drop(F.silu(agg)))

            hi, hj = h[edge_src], h[edge_dst]
            if self.use_directional_geometry and data["triplet_ea"].numel() > 0:
                va = pos[data["edge_dst"][data["triplet_ea"]]] - pos[data["edge_src"][data["triplet_ea"]]]
                vb = pos[data["edge_dst"][data["triplet_eb"]]] - pos[data["edge_src"][data["triplet_eb"]]]
                na = torch.norm(va, dim=1).clamp(min=MIN_DIST)
                nb = torch.norm(vb, dim=1).clamp(min=MIN_DIST)
                cosang = torch.clamp((va * vb).sum(dim=1) / (na * nb), -0.99, 0.99)
                ang_feat = self.ang(torch.acos(cosang))

                tri_in = torch.cat(
                    [m[data["triplet_eb"]], rbf_e[data["triplet_ea"]], rbf_e[data["triplet_eb"]], ang_feat], dim=1)
                update = torch.zeros_like(m)
                update.index_add_(0, data["triplet_ea"], self.triplet_mlp(tri_in))
                m = m + update

        B = data["g"].shape[0]
        sum_h = torch.zeros((B, self.hidden), device=h.device)
        sum_h.index_add_(0, batch_node, h)
        cnt = torch.zeros((B,), device=h.device)
        cnt.index_add_(0, batch_node, torch.ones_like(batch_node, dtype=h.dtype))
        mean_h = sum_h / (cnt.unsqueeze(1) + 1e-8)

        max_h = torch.full((B, self.hidden), -1e9, device=h.device)
        for b in range(B):
            mask = batch_node == b
            if mask.any(): max_h[b] = h[mask].max(dim=0)[0]

        out = self.readout(torch.cat([mean_h, max_h, data["g"]], dim=1))
        return out.squeeze(-1)


# =========================
# 🔄 Training Routine with Strict Splitting
# =========================
# =========================
# 🔄 Training Routine with Strict Splitting
# =========================
def _run_training_experiment_legacy(all_data, groups, split_mode="random"):
    EXP_NAME = f"mode_{split_mode}"
    EXP_DIR = os.path.join(BASE_SAVE_DIR, EXP_NAME)
    os.makedirs(EXP_DIR, exist_ok=True)

    print(f"\n🚀 START: {split_mode.upper()} SPLIT -> {EXP_DIR}")

    n = len(all_data)
    idx_all = np.arange(n)

    if split_mode == "group_folder":
        train_idx, val_idx, test_idx = [], [], []
        for i, item in enumerate(all_data):
            atom_cnt = int(item["intended_N"])
            if atom_cnt != int(item["N"]):
                raise RuntimeError(
                    f"label/geometry mismatch for ID={item.get('sample_id')}: "
                    f"CSV C{atom_cnt}, geometry C{item['N']}"
                )
            if 20 <= atom_cnt <= 56:
                train_idx.append(i)
            elif atom_cnt in GROUP_VAL_SIZES:
                val_idx.append(i)
            elif 70 <= atom_cnt <= 100:
                test_idx.append(i)
            else:
                raise RuntimeError(
                    f"C{atom_cnt} is outside the declared strict group protocol."
                )

        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        test_idx = np.array(test_idx)
        val_label = ", ".join(f"C{n}" for n in GROUP_VAL_SIZES)
        print(
            f"   [Group Logic] Train(C20-56): {len(train_idx)}, "
            f"Val({val_label}): {len(val_idx)}, Test(Rest): {len(test_idx)}"
        )
    else:
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(idx_all)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        train_idx = perm[:n_train]
        val_idx = perm[n_train: n_train + n_val]
        test_idx = perm[n_train + n_val:]
        print(f"   [Random Logic 8:1:1] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    from sklearn.linear_model import Ridge

    all_train_N = np.array([all_data[i]["intended_N"] for i in train_idx])
    all_train_y = np.array([all_data[i]["y_target"] for i in train_idx])
    train_y_low, train_y_high = np.quantile(all_train_y, PRED_RANGE_GUARD_Q)
    train_y_span = float(max(train_y_high - train_y_low, 1e-6))
    pred_guard_low = float(max(0.0, train_y_low - PRED_RANGE_GUARD_MARGIN * train_y_span))
    pred_guard_high = float(train_y_high + PRED_RANGE_GUARD_MARGIN * train_y_span)

    X_base_train = np.stack([build_baseline_features(n) for n in all_train_N], axis=0)
    base_model = Ridge(alpha=1.0).fit(X_base_train, all_train_y)

    train_residuals = all_train_y - base_model.predict(X_base_train)
    global_mu_res = float(np.mean(train_residuals))
    global_sigma_res = float(max(np.std(train_residuals), BASELINE_MIN_RES_STD))
    res_clip_low, res_clip_high = np.quantile(train_residuals, RESIDUAL_CLIP_Q)
    res_clip_low = float(res_clip_low)
    res_clip_high = float(res_clip_high)
    if res_clip_high <= res_clip_low:
        res_clip_low = float(np.min(train_residuals))
        res_clip_high = float(np.max(train_residuals))
    if res_clip_high <= res_clip_low:
        res_clip_low = global_mu_res - 2.0 * global_sigma_res
        res_clip_high = global_mu_res + 2.0 * global_sigma_res

    g_vals = [all_data[i]["g"].numpy() for i in train_idx]
    scaler_g = StandardScaler().fit(np.array(g_vals))
    train_count_map = pd.Series(all_train_N).value_counts().to_dict()
    mean_train_count = float(np.mean(list(train_count_map.values())))

    def prepare_dataset(indices, is_train):
        ds = []
        for i in indices:
            it = dict(all_data[i])
            N_val = int(it["N"])
            base_y = float(base_model.predict(build_baseline_features(N_val).reshape(1, -1))[0])
            it["y_target"] = float((it["y_target"] - base_y - global_mu_res) / global_sigma_res)

            g_numpy = np.nan_to_num(it["g"].numpy().reshape(1, -1))
            it["g"] = torch.tensor(scaler_g.transform(g_numpy)[0], dtype=torch.float32)
            if split_mode == "group_folder" and is_train and USE_SIZE_BALANCED_LOSS:
                raw_weight = (mean_train_count / max(float(train_count_map.get(N_val, 1.0)), 1.0)) ** SIZE_WEIGHT_POWER
                it["sample_weight"] = float(np.clip(raw_weight, SIZE_WEIGHT_CLAMP[0], SIZE_WEIGHT_CLAMP[1]))
            else:
                it["sample_weight"] = 1.0
            ds.append(it)
        return FastDataset(ds, is_train)

    train_loader = DataLoader(prepare_dataset(train_idx, True), batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_graphs, num_workers=0)
    val_loader = DataLoader(prepare_dataset(val_idx, False), batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_graphs, num_workers=0)
    test_loader = DataLoader(prepare_dataset(test_idx, False), batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_graphs, num_workers=0)

    sample = all_data[0]
    node_dim = sample["node_feat"].shape[1]
    edge_dim = sample["e_ring"].shape[2]
    g_dim = sample["g"].shape[0]

    model = DimeNetLite(node_dim, edge_dim, g_dim, HIDDEN_DIM, RBF_DIM, ANGLE_BASIS_K, NUM_LAYERS, DROPOUT_RATE).to(
        DEVICE)

    if USE_SAM:
        optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, rho=SAM_RHO)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer.base_optimizer if USE_SAM else optimizer,
        T_0=10, T_mult=2, eta_min=1e-6
    )
    criterion = nn.MSELoss(reduction="none")

    save_path = os.path.join(EXP_DIR, "best_model.pth")
    early_stop = EarlyStopping(patience=PATIENCE, verbose=False, path=save_path)

    def compute_batch_loss(batch):
        pred = model(batch)
        per_sample_loss = criterion(pred, batch["y"])
        if split_mode == "group_folder" and USE_SIZE_BALANCED_LOSS:
            return (per_sample_loss * batch["sample_weight"]).mean()
        return per_sample_loss.mean()

    def collect_rows(loader, split_name):
        rows_local = []
        with torch.no_grad():
            for batch in loader:
                if not batch:
                    continue
                for k in batch:
                    if torch.is_tensor(batch[k]):
                        batch[k] = batch[k].to(DEVICE)
                pred = model(batch).cpu().numpy()
                true = batch["y"].cpu().numpy()
                Ns = batch["N"].cpu().numpy()

                for i in range(len(Ns)):
                    atom_count = int(Ns[i])
                    p = restore_prediction_value(
                        pred[i],
                        atom_count,
                        base_model,
                        global_mu_res,
                        global_sigma_res,
                        res_clip_low=res_clip_low,
                        res_clip_high=res_clip_high,
                        use_guard=(split_mode == "group_folder"),
                    )
                    t = restore_target_value(true[i], atom_count, base_model, global_mu_res, global_sigma_res)
                    rows_local.append({"true": t, "pred": p, "N": atom_count, "split": split_name})
        return rows_local

    def summarize_validation(rows_local):
        val_df = pd.DataFrame(rows_local)
        overall_r2 = r2_score(val_df["true"], val_df["pred"])
        if split_mode != "group_folder":
            return overall_r2, overall_r2, {}

        size_scores = {}
        for size in GROUP_VAL_SIZES:
            sub_df = val_df[val_df["N"] == size]
            if len(sub_df) >= 2:
                size_scores[size] = r2_score(sub_df["true"], sub_df["pred"])
            else:
                size_scores[size] = -1.0

        score = float(np.mean(list(size_scores.values())))
        if any(v <= 0.0 for v in size_scores.values()):
            score = min(score, -1.0)
        return score, overall_r2, size_scores

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_sum = 0
        for batch in train_loader:
            if not batch:
                continue
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(DEVICE)

            optimizer.zero_grad()
            if USE_SAM:
                loss = compute_batch_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.first_step(zero_grad=True)

                loss2 = compute_batch_loss(batch)
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.second_step(zero_grad=True)
            else:
                loss = compute_batch_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
            loss_sum += loss.item()

        model.eval()
        val_rows = collect_rows(val_loader, "val")
        val_score, val_r2_overall, val_size_scores = summarize_validation(val_rows)
        early_stop(val_score, model)

        if epoch % 10 == 0:
            if split_mode == "group_folder":
                size_msg = " | ".join(
                    [f"Val R2 C{size}: {val_size_scores.get(size, -1.0):.4f}" for size in GROUP_VAL_SIZES]
                )
                print(
                    f"Ep {epoch:03d} | Loss: {loss_sum / len(train_loader):.4f} | "
                    f"Val Score: {val_score:.4f} | Val R2 Overall: {val_r2_overall:.4f} | {size_msg}"
                )
            else:
                print(f"Ep {epoch:03d} | Loss: {loss_sum / len(train_loader):.4f} | Val R2: {val_score:.4f}")

        if early_stop.early_stop:
            print("Early Stopping!")
            break
        scheduler.step()

    model.load_state_dict(safe_torch_load(save_path, map_location=DEVICE))
    model.eval()
    rows = []
    rows.extend(collect_rows(train_loader, "train"))
    rows.extend(collect_rows(val_loader, "val"))
    rows.extend(collect_rows(test_loader, "test"))

    df = pd.DataFrame(rows)
    csv_file = os.path.join(EXP_DIR, "results.csv")
    df.to_csv(csv_file, index=False)

    test_df = df[df["split"] == "test"]
    final_r2 = r2_score(test_df["true"], test_df["pred"])
    print(f"🏁 Test R2: {final_r2:.4f}")
    return csv_file, final_r2


def run_training_experiment(all_data, groups, split_mode="random"):
    EXP_NAME = f"mode_{split_mode}"
    EXP_DIR = os.path.join(BASE_SAVE_DIR, EXP_NAME)
    os.makedirs(EXP_DIR, exist_ok=True)
    status_path = os.path.join(EXP_DIR, "run_status.txt")

    def write_exp_status(*lines):
        with open(status_path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(str(line) + "\n")

    print(f"\nSTART: {split_mode.upper()} SPLIT -> {EXP_DIR}")
    write_exp_status(f"START split={split_mode}", f"EXP_DIR={os.path.abspath(EXP_DIR)}")

    n = len(all_data)
    idx_all = np.arange(n)

    if split_mode == "group_folder":
        train_idx, val_idx, test_idx = [], [], []
        for i, item in enumerate(all_data):
            atom_cnt = int(item["intended_N"])
            if atom_cnt != int(item["N"]):
                raise RuntimeError(
                    f"label/geometry mismatch for ID={item.get('sample_id')}: "
                    f"CSV C{atom_cnt}, geometry C{item['N']}"
                )
            if 20 <= atom_cnt <= 56:
                train_idx.append(i)
            elif atom_cnt in GROUP_VAL_SIZES:
                val_idx.append(i)
            else:
                test_idx.append(i)

        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        test_idx = np.array(test_idx)
        val_label = ", ".join(f"C{size}" for size in GROUP_VAL_SIZES)
        print(
            f"   [Group Logic] Train(C20-56): {len(train_idx)}, "
            f"Val({val_label}): {len(val_idx)}, Test(Rest): {len(test_idx)}"
        )
        write_exp_status(
            f"SPLIT train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}",
            f"TRAIN_RANGE=C20-C56 VAL={val_label}",
        )
    else:
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(idx_all)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]
        print(f"   [Random Logic 8:1:1] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        write_exp_status(f"SPLIT train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        msg = f"Invalid split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        write_exp_status("ERROR " + msg)
        raise RuntimeError(msg)

    protocol_name = "group" if split_mode == "group_folder" else "random"
    actual_counts = {
        "train": int(len(train_idx)),
        "val": int(len(val_idx)),
        "test": int(len(test_idx)),
    }
    if actual_counts != EXPECTED_SPLIT_COUNTS[protocol_name]:
        msg = (
            f"Unexpected {protocol_name} split counts: {actual_counts}; "
            f"expected {EXPECTED_SPLIT_COUNTS[protocol_name]}."
        )
        write_exp_status("ERROR " + msg)
        raise RuntimeError(msg)

    split_lookup = {}
    for split_name, split_values in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for sample_index in split_values:
            split_lookup[int(sample_index)] = split_name
    manifest = pd.DataFrame([
        {
            "sample_id": int(item["sample_id"]),
            "intended_N": int(item["intended_N"]),
            "geometry_N": int(item["N"]),
            "split": split_lookup[index],
            "gap_unit": item["unit"],
            "source_path": item["source_path"],
        }
        for index, item in enumerate(all_data)
    ])
    manifest_file = os.path.join(EXP_DIR, "dataset_manifest.csv")
    manifest.to_csv(manifest_file, index=False)

    from sklearn.linear_model import Ridge

    all_train_N = np.array([all_data[i]["N"] for i in train_idx])
    all_train_y = np.array([all_data[i]["y_target"] for i in train_idx])

    X_base_train = np.stack([build_baseline_features_from_item(all_data[i]) for i in train_idx], axis=0)
    if ablation_enabled("use_size_baseline"):
        base_model = Ridge(alpha=1.0).fit(X_base_train, all_train_y)
        train_residuals = all_train_y - base_model.predict(X_base_train)
    else:
        base_model = None
        train_residuals = all_train_y.copy()
    global_mu_res = float(np.mean(train_residuals))
    global_sigma_res = float(max(np.std(train_residuals), BASELINE_MIN_RES_STD))

    # Fit every unsupervised transformation on the training split only.
    if ablation_enabled("use_global_descriptors") and ablation_enabled("use_geometry_inputs"):
        train_phys = np.asarray([all_data[i]["phys_feat"] for i in train_idx], dtype=np.float32)
        all_phys = np.asarray([item["phys_feat"] for item in all_data], dtype=np.float32)
        phys_scaler = StandardScaler().fit(train_phys)
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=SEED, n_init=20)
        kmeans.fit(phys_scaler.transform(train_phys))
        cluster_labels = kmeans.predict(phys_scaler.transform(all_phys))
    else:
        cluster_labels = np.zeros(len(all_data), dtype=np.int64)

    g_vals = []
    for i in train_idx:
        base_feat = build_baseline_features_from_item(all_data[i])
        base_y = float(base_model.predict(base_feat.reshape(1, -1))[0]) if base_model is not None else 0.0
        g_vals.append(build_augmented_global_feature(all_data[i], base_y, int(cluster_labels[i])))
    scaler_g = StandardScaler().fit(np.array(g_vals))
    train_count_map = pd.Series(all_train_N).value_counts().to_dict()
    mean_train_count = float(np.mean(list(train_count_map.values())))

    def prepare_dataset(indices, is_train):
        ds = []
        for i in indices:
            it = dict(all_data[i])
            it["node_feat"] = it["node_feat"].clone()
            it["e_ring"] = it["e_ring"].clone()
            if not ablation_enabled("use_topology_inputs"):
                if it["node_feat"].shape[1] > 1:
                    it["node_feat"][:, :-1] = 0.0
                it["e_ring"].zero_()
            if not ablation_enabled("use_geometry_inputs"):
                it["node_feat"][:, -1] = 0.0
            atom_count = int(it["intended_N"])
            base_feat = build_baseline_features_from_item(all_data[i])
            base_y = float(base_model.predict(base_feat.reshape(1, -1))[0]) if base_model is not None else 0.0
            it["y_target"] = float((it["y_target"] - base_y - global_mu_res) / global_sigma_res)
            g_numpy = build_augmented_global_feature(it, base_y, int(cluster_labels[i])).reshape(1, -1)
            it["g"] = torch.tensor(scaler_g.transform(g_numpy)[0], dtype=torch.float32)
            it["base_feat"] = torch.tensor(base_feat, dtype=torch.float32)
            if (
                split_mode == "group_folder"
                and is_train
                and ablation_enabled("use_size_balanced_weighting")
            ):
                raw_weight = (mean_train_count / max(float(train_count_map.get(atom_count, 1.0)), 1.0)) ** SIZE_WEIGHT_POWER
                tail_boost = 1.0 + TARGET_TAIL_WEIGHT * min(abs(it["y_target"]), TARGET_TAIL_CLAMP)
                it["sample_weight"] = float(np.clip(raw_weight * tail_boost, SIZE_WEIGHT_CLAMP[0], SIZE_WEIGHT_CLAMP[1]))
            else:
                it["sample_weight"] = 1.0
            ds.append(it)
        return FastDataset(ds, is_train)

    train_loader = DataLoader(
        prepare_dataset(train_idx, True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=0,
    )
    val_loader = DataLoader(
        prepare_dataset(val_idx, False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=0,
    )
    test_loader = DataLoader(
        prepare_dataset(test_idx, False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=0,
    )

    sample = all_data[0]
    node_dim = sample["node_feat"].shape[1]
    edge_dim = sample["e_ring"].shape[2]
    g_dim = int(scaler_g.mean_.shape[0])

    model = DimeNetLite(
        node_dim,
        edge_dim,
        g_dim,
        HIDDEN_DIM,
        RBF_DIM,
        ANGLE_BASIS_K,
        NUM_LAYERS,
        DROPOUT_RATE,
        use_directional_geometry=ablation_enabled("use_directional_geometry"),
    ).to(DEVICE)
    write_exp_status(
        "MODEL_INITIALIZED",
        f"DEVICE={DEVICE}",
        f"node_dim={node_dim} edge_dim={edge_dim} global_dim={g_dim}",
    )

    use_sam = ablation_enabled("use_sam")
    if use_sam:
        optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, rho=SAM_RHO)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer.base_optimizer if use_sam else optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6,
    )
    criterion = nn.MSELoss(reduction="none")

    save_path = os.path.join(EXP_DIR, "best_model.pth")
    last_save_path = os.path.join(EXP_DIR, "last_model.pth")
    early_stop = EarlyStopping(patience=PATIENCE, verbose=False, path=save_path)
    def compute_batch_loss(batch):
        pred = model(batch)
        mse_loss = criterion(pred, batch["y"])
        huber_loss = F.smooth_l1_loss(pred, batch["y"], reduction="none", beta=0.5)
        per_sample_loss = (1.0 - HUBER_MIX) * mse_loss + HUBER_MIX * huber_loss
        if split_mode == "group_folder" and ablation_enabled("use_size_balanced_weighting"):
            base_loss = (per_sample_loss * batch["sample_weight"]).mean()
        else:
            base_loss = per_sample_loss.mean()

        if pred.numel() > 1 and ablation_enabled("use_auxiliary_loss"):
            pred_std = pred.std(unbiased=False)
            target_std = batch["y"].std(unbiased=False)
            std_loss = (pred_std - target_std).pow(2)

            pred_center = pred - pred.mean()
            target_center = batch["y"] - batch["y"].mean()
            denom = pred_center.std(unbiased=False) * target_center.std(unbiased=False) + 1e-8
            corr = (pred_center * target_center).mean() / denom
            extra_loss = STD_MATCH_WEIGHT * std_loss + CORR_LOSS_WEIGHT * (1.0 - corr)

            high_gap_threshold = torch.quantile(batch["y"].detach(), HIGH_GAP_LOSS_Q)
            high_gap_mask = batch["y"] >= high_gap_threshold
            if high_gap_mask.sum() >= 2:
                under_error = F.relu(batch["y"][high_gap_mask] - pred[high_gap_mask])
                extra_loss = extra_loss + HIGH_GAP_UNDER_WEIGHT * under_error.pow(2).mean()
                if HIGH_GAP_MEAN_UNDER_WEIGHT > 0.0:
                    mean_under = F.relu(batch["y"][high_gap_mask].mean() - pred[high_gap_mask].mean())
                    extra_loss = extra_loss + HIGH_GAP_MEAN_UNDER_WEIGHT * mean_under.pow(2)

            top_std_threshold = torch.quantile(batch["y"].detach(), TOP_STD_MATCH_Q)
            top_mask = batch["y"] >= top_std_threshold
            if top_mask.sum() >= 3:
                top_pred_std = pred[top_mask].std(unbiased=False)
                top_target_std = batch["y"][top_mask].std(unbiased=False)
                extra_loss = extra_loss + TOP_STD_MATCH_WEIGHT * (top_pred_std - top_target_std).pow(2)
                if TOP_RANGE_MATCH_WEIGHT > 0.0:
                    top_pred_range = pred[top_mask].max() - pred[top_mask].min()
                    top_target_range = batch["y"][top_mask].max() - batch["y"][top_mask].min()
                    extra_loss = extra_loss + TOP_RANGE_MATCH_WEIGHT * (top_pred_range - top_target_range).pow(2)

            return base_loss + extra_loss

        return base_loss

    def collect_raw_outputs(loader):
        trues, base_preds, raw_residuals, atom_sizes = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                if not batch:
                    continue
                for k in batch:
                    if torch.is_tensor(batch[k]):
                        batch[k] = batch[k].to(DEVICE)
                pred = model(batch).cpu().numpy()
                true = batch["y"].cpu().numpy()
                Ns = batch["N"].cpu().numpy()
                batch_base_features = batch["base_feat"].cpu().numpy()

                for i in range(len(Ns)):
                    atom_count = int(Ns[i])
                    base_y = (
                        float(base_model.predict(batch_base_features[i].reshape(1, -1))[0])
                        if base_model is not None else 0.0
                    )
                    trues.append(float(base_y + true[i] * global_sigma_res + global_mu_res))
                    base_preds.append(base_y)
                    raw_residuals.append(float(pred[i] * global_sigma_res + global_mu_res))
                    atom_sizes.append(atom_count)

        return (
            np.asarray(trues, dtype=np.float32),
            np.asarray(base_preds, dtype=np.float32),
            np.asarray(raw_residuals, dtype=np.float32),
            np.asarray(atom_sizes, dtype=np.int32),
        )

    def summarize_predictions(trues, preds, atom_sizes):
        overall_r2 = r2_score(trues, preds)
        if split_mode != "group_folder":
            return overall_r2, overall_r2, {}

        size_scores = {}
        for size in GROUP_VAL_SIZES:
            mask = atom_sizes == size
            if mask.sum() >= 2:
                size_scores[size] = r2_score(trues[mask], preds[mask])
            else:
                size_scores[size] = -1.0

        mean_size_r2 = float(np.mean(list(size_scores.values())))
        negative_count = sum(score <= 0.0 for score in size_scores.values())

        tail_threshold = float(np.quantile(trues, TAIL_SCORE_Q))
        tail_mask = trues >= tail_threshold
        if tail_mask.sum() >= 8 and not np.isclose(np.var(trues[tail_mask]), 0.0):
            tail_r2 = float(r2_score(trues[tail_mask], preds[tail_mask]))
            tail_bias = float(np.mean(preds[tail_mask] - trues[tail_mask]))
            tail_under = float(np.mean(np.maximum(trues[tail_mask] - preds[tail_mask], 0.0)))
        else:
            tail_r2 = 0.0
            tail_bias = 0.0
            tail_under = 0.0

        score = (
            0.25 * mean_size_r2
            + 0.63 * overall_r2
            + TAIL_R2_SCORE_WEIGHT * tail_r2
            - TAIL_BIAS_SCORE_WEIGHT * tail_under
            - NEGATIVE_VAL_PENALTY * negative_count
        )
        if overall_r2 <= 0.0 or negative_count > 0:
            score = min(score, -1.0 - NEGATIVE_VAL_PENALTY * negative_count)
        return score, overall_r2, size_scores

    def evaluate_raw_validation(trues, base_preds, raw_residuals, atom_sizes):
        """Evaluate the unmodified model output for checkpoint selection."""
        raw_preds = np.asarray(base_preds, dtype=np.float32) + np.asarray(raw_residuals, dtype=np.float32)
        return summarize_predictions(trues, raw_preds, atom_sizes)

    def collect_rows(loader, split_name):
        trues, base_preds, raw_residuals, atom_sizes = collect_raw_outputs(loader)
        raw_preds = np.asarray(base_preds, dtype=np.float32) + np.asarray(raw_residuals, dtype=np.float32)
        rows_local = []
        for i in range(len(raw_preds)):
            raw_pred = float(raw_preds[i])
            rows_local.append(
                {
                    "true": float(trues[i]),
                    # ``pred`` is a backward-compatible alias of ``pred_raw``.
                    "pred": raw_pred,
                    "pred_raw": raw_pred,
                    "N": int(atom_sizes[i]),
                    "split": split_name,
                    "unit": TARGET_UNIT,
                }
            )
        return rows_local

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        for batch in train_loader:
            if not batch:
                continue
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(DEVICE)

            optimizer.zero_grad()
            if use_sam:
                loss = compute_batch_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.first_step(zero_grad=True)

                loss2 = compute_batch_loss(batch)
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.second_step(zero_grad=True)
            else:
                loss = compute_batch_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
            loss_sum += loss.item()
            batch_count += 1

        model.eval()
        val_trues, val_base_preds, val_raw_residuals, val_atom_sizes = collect_raw_outputs(val_loader)
        val_score, val_r2_overall, val_size_scores = evaluate_raw_validation(
            val_trues,
            val_base_preds,
            val_raw_residuals,
            val_atom_sizes,
        )
        prev_best = -np.inf if early_stop.best_score is None else early_stop.best_score
        early_stop(val_score, model)
        torch.save(model.state_dict(), last_save_path)
        write_exp_status(
            f"EPOCH={epoch}",
            f"loss={loss_sum / max(batch_count, 1):.6f}",
            f"val_score={val_score:.6f}",
            f"val_r2_overall={val_r2_overall:.6f}",
            f"best_score={early_stop.best_score:.6f}",
            f"saved_last={last_save_path}",
        )
        if early_stop.best_score is not None and early_stop.best_score > prev_best:
            write_exp_status(f"saved_best={save_path}")

        if epoch % 10 == 0:
            avg_loss = loss_sum / max(batch_count, 1)
            if split_mode == "group_folder":
                size_msg = " | ".join(
                    [f"Val R2 C{size}: {val_size_scores.get(size, -1.0):.4f}" for size in GROUP_VAL_SIZES]
                )
                print(
                    f"Ep {epoch:03d} | Loss: {avg_loss:.4f} | Val Score: {val_score:.4f} | "
                    f"Val R2 Overall (raw): {val_r2_overall:.4f} | {size_msg}"
                )
            else:
                print(f"Ep {epoch:03d} | Loss: {avg_loss:.4f} | Val R2: {val_score:.4f}")

        if early_stop.early_stop:
            print("Early Stopping!")
            break
        scheduler.step()

    if not os.path.exists(save_path):
        if os.path.exists(last_save_path):
            torch.save(safe_torch_load(last_save_path, map_location=DEVICE), save_path)
            write_exp_status(f"WARNING best_model_missing; copied last_model to {save_path}")
        else:
            msg = "No checkpoint was saved. Check run_status.txt and terminal traceback."
            write_exp_status("ERROR " + msg)
            raise RuntimeError(msg)

    model.load_state_dict(safe_torch_load(save_path, map_location=DEVICE))
    model.eval()

    write_exp_status(
        "POSTPROCESSING_MODE=disabled",
        "PRIMARY_PREDICTION=pred_raw",
    )

    rows = []
    rows.extend(collect_rows(train_loader, "train"))
    rows.extend(collect_rows(val_loader, "val"))
    rows.extend(collect_rows(test_loader, "test"))

    df = pd.DataFrame(rows)
    csv_file = os.path.join(EXP_DIR, "results.csv")
    df.to_csv(csv_file, index=False)
    write_exp_status(f"RESULTS={csv_file}")

    def safe_r2(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if y_true.size < 2 or np.isclose(np.var(y_true), 0.0):
            return np.nan
        return float(r2_score(y_true, y_pred))

    metric_rows = []
    for (split_name, atom_count), sub_df in df.groupby(["split", "N"]):
        err = sub_df["pred"].to_numpy(dtype=np.float64) - sub_df["true"].to_numpy(dtype=np.float64)
        raw_err = sub_df["pred_raw"].to_numpy(dtype=np.float64) - sub_df["true"].to_numpy(dtype=np.float64)
        metric_rows.append({
            "split": split_name,
            "N": int(atom_count),
            "count": int(len(sub_df)),
            "unit": TARGET_UNIT,
            "r2": safe_r2(sub_df["true"], sub_df["pred"]),
            "mae": float(mean_absolute_error(sub_df["true"], sub_df["pred"])),
            "rmse": float(np.sqrt(mean_squared_error(sub_df["true"], sub_df["pred"]))),
            "r2_raw": safe_r2(sub_df["true"], sub_df["pred_raw"]),
            "mae_raw": float(mean_absolute_error(sub_df["true"], sub_df["pred_raw"])),
            "rmse_raw": float(np.sqrt(mean_squared_error(sub_df["true"], sub_df["pred_raw"]))),
            "true_mean": float(sub_df["true"].mean()),
            "pred_mean": float(sub_df["pred"].mean()),
            "bias_pred_minus_true": float(err.mean()),
            "raw_bias_pred_minus_true": float(raw_err.mean()),
        })
    metrics_by_n = pd.DataFrame(metric_rows).sort_values(["split", "N"])
    metrics_by_n_file = os.path.join(EXP_DIR, "metrics_by_N.csv")
    metrics_by_n.to_csv(metrics_by_n_file, index=False)

    split_summary = (
        df.groupby(["split", "N"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "N"])
    )
    split_summary["unit"] = TARGET_UNIT
    split_summary_file = os.path.join(EXP_DIR, "split_summary.csv")
    split_summary.to_csv(split_summary_file, index=False)

    postprocess_params_file = os.path.join(EXP_DIR, "postprocess_params.txt")
    with open(postprocess_params_file, "w", encoding="utf-8") as f:
        f.write(f"target_unit={TARGET_UNIT}\n")
        f.write(f"postprocessing_enabled={ENABLE_POSTPROCESSING}\n")
        f.write(f"postprocessing_mode={POSTPROCESSING_MODE if ENABLE_POSTPROCESSING else 'disabled'}\n")
        f.write("checkpoint_selection_predictions=raw\n")
        f.write("primary_metrics=raw_model_predictions\n")
        f.write("secondary_metrics=none\n")
        f.write("legacy_residual_tail_heuristics_enabled=False\n")
        f.write(f"baseline_feature_dim={X_base_train.shape[1]}\n")

    tail_diag_file = os.path.join(EXP_DIR, "tail_diagnostics.txt")
    with open(tail_diag_file, "w", encoding="utf-8") as f:
        for split_name in ("train", "val", "test"):
            sub_df = df[df["split"] == split_name]
            if sub_df.empty:
                continue
            threshold = float(sub_df["true"].quantile(0.90))
            top_df = sub_df[sub_df["true"] >= threshold]
            f.write(f"[{split_name}]\n")
            f.write(f"count={len(sub_df)}\n")
            f.write(f"overall_r2={safe_r2(sub_df['true'], sub_df['pred'])}\n")
            f.write(f"overall_mae={mean_absolute_error(sub_df['true'], sub_df['pred'])}\n")
            f.write(f"overall_rmse={np.sqrt(mean_squared_error(sub_df['true'], sub_df['pred']))}\n")
            f.write(f"overall_r2_raw={safe_r2(sub_df['true'], sub_df['pred_raw'])}\n")
            f.write(f"overall_mae_raw={mean_absolute_error(sub_df['true'], sub_df['pred_raw'])}\n")
            f.write(f"overall_rmse_raw={np.sqrt(mean_squared_error(sub_df['true'], sub_df['pred_raw']))}\n")
            f.write(f"top10_true_threshold={threshold}\n")
            f.write(f"top10_count={len(top_df)}\n")
            f.write(f"top10_r2={safe_r2(top_df['true'], top_df['pred'])}\n")
            f.write(f"top10_mae={mean_absolute_error(top_df['true'], top_df['pred'])}\n")
            f.write(f"top10_rmse={np.sqrt(mean_squared_error(top_df['true'], top_df['pred']))}\n")
            f.write(f"top10_bias_pred_minus_true={(top_df['pred'] - top_df['true']).mean()}\n")
            f.write(f"top10_mean_underprediction={np.maximum(top_df['true'] - top_df['pred'], 0.0).mean()}\n\n")

    segment_rows = []
    segment_specs = [
        ("all", None, None),
        ("C70_C80", 70, 80),
        ("C82_C90", 82, 90),
        ("C92_C100", 92, 100),
        ("high_gap_top10", None, None),
    ]
    for split_name in ("train", "val", "test"):
        split_df = df[df["split"] == split_name]
        if split_df.empty:
            continue
        high_threshold = float(split_df["true"].quantile(0.90))
        for segment_name, n_low, n_high in segment_specs:
            if segment_name == "all":
                sub_df = split_df
            elif segment_name == "high_gap_top10":
                sub_df = split_df[split_df["true"] >= high_threshold]
            else:
                sub_df = split_df[(split_df["N"] >= n_low) & (split_df["N"] <= n_high)]
            if sub_df.empty:
                continue
            err = sub_df["pred"].to_numpy(dtype=np.float64) - sub_df["true"].to_numpy(dtype=np.float64)
            raw_err = sub_df["pred_raw"].to_numpy(dtype=np.float64) - sub_df["true"].to_numpy(dtype=np.float64)
            segment_rows.append({
                "split": split_name,
                "segment": segment_name,
                "count": int(len(sub_df)),
                "n_min": int(sub_df["N"].min()),
                "n_max": int(sub_df["N"].max()),
                "r2": safe_r2(sub_df["true"], sub_df["pred"]),
                "mae": float(mean_absolute_error(sub_df["true"], sub_df["pred"])),
                "rmse": float(np.sqrt(mean_squared_error(sub_df["true"], sub_df["pred"]))),
                "true_mean": float(sub_df["true"].mean()),
                "pred_mean": float(sub_df["pred"].mean()),
                "true_std": float(sub_df["true"].std(ddof=0)),
                "pred_std": float(sub_df["pred"].std(ddof=0)),
                "bias_pred_minus_true": float(err.mean()),
                "r2_raw": safe_r2(sub_df["true"], sub_df["pred_raw"]),
                "mae_raw": float(mean_absolute_error(sub_df["true"], sub_df["pred_raw"])),
                "rmse_raw": float(np.sqrt(mean_squared_error(sub_df["true"], sub_df["pred_raw"]))),
                "raw_bias_pred_minus_true": float(raw_err.mean()),
            })
    segment_diag_file = os.path.join(EXP_DIR, "segment_diagnostics.csv")
    pd.DataFrame(segment_rows).to_csv(segment_diag_file, index=False)

    write_exp_status(
        f"METRICS_BY_N={metrics_by_n_file}",
        f"SPLIT_SUMMARY={split_summary_file}",
        f"POSTPROCESS_PARAMS={postprocess_params_file}",
        f"TAIL_DIAGNOSTICS={tail_diag_file}",
        f"SEGMENT_DIAGNOSTICS={segment_diag_file}",
    )

    test_df = df[df["split"] == "test"]
    final_r2_raw = r2_score(test_df["true"], test_df["pred_raw"])
    print(f"Test R2 (raw, primary): {final_r2_raw:.4f}")
    write_exp_status(f"TEST_R2_RAW_PRIMARY={final_r2_raw:.6f}", "SUCCESS")
    return csv_file, final_r2_raw


def _legacy_main_unvalidated():
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    seed_everything(SEED)

    if os.path.exists(CACHE_FILE):
        print(f"📥 Loading shared cache: {CACHE_FILE}")
        cache = safe_torch_load(CACHE_FILE)
        all_data, groups = cache["data"], cache["groups"]
    else:
        print(f"⚡ Building Dataset from GJF...")

        if not os.path.exists(DATA_FILE):
            print(f"❌ Error: CSV file '{DATA_FILE}' not found!")
            return

        df = pd.read_csv(DATA_FILE)

        # === 1. 建立超级索引 (针对深层嵌套结构优化) ===
        # 目标结构: Organized_Fullerenes/20/test/000001/xxx.gjf
        print(f"📂 Scanning directory tree: {os.path.abspath(ROOT_FOLDER)}")

        # 索引键: (原子数字符串, ID字符串) -> 文件绝对路径
        file_index = {}

        scan_count = 0
        if not os.path.exists(ROOT_FOLDER):
            print(f"❌ Error: Root folder '{ROOT_FOLDER}' does not exist.")
            return

        # 遍历所有文件
        for root, dirs, files in os.walk(ROOT_FOLDER):
            for f in files:
                if f.lower().endswith(".gjf"):
                    full_path = os.path.join(root, f)

                    # === 智能解析路径 ===
                    parts = os.path.normpath(full_path).split(os.sep)

                    possible_ids = []
                    possible_atoms = []

                    # 收集路径里所有看起来像数字的部分
                    for part in parts:
                        if part.isdigit():
                            val = str(int(part))
                            possible_ids.append(val)
                            possible_atoms.append(val)

                    # 将这个文件注册到所有可能的 (Atom, ID) 组合中
                    for a in possible_atoms:
                        for i in possible_ids:
                            key = (a, i)
                            if key not in file_index:
                                file_index[key] = full_path

                    scan_count += 1

        print(f"   Scanned {scan_count} .gjf files.")
        print(f"   Indexed {len(file_index)} unique (Atom, ID) combinations.")

        # === 2. 匹配 CSV 数据 ===
        tasks = []
        matched_count = 0
        debug_missing = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Matching Files"):
            atom_cnt = str(row['AtomCount'])  # e.g. "20"
            mol_id = str(row['ID'])  # e.g. "1"
            gap = float(row['Gap'])

            # 直接查表
            key = (atom_cnt, mol_id)

            if key in file_index:
                path = file_index[key]
                tasks.append((path, gap, atom_cnt, f"{mol_id}.gjf"))
                matched_count += 1
            else:
                if len(debug_missing) < 5:
                    debug_missing.append(f"Atom={atom_cnt}, ID={mol_id}")

        print(f"✅ Matched {matched_count} files out of {len(df)}.")

        if matched_count == 0:
            print("❌ CRITICAL: No files matched!")
            print("   Debug Info - First 5 missing requests:", debug_missing)
            print("   Debug Info - Sample keys in index:", list(file_index.keys())[:10])
            return

        all_data = []
        print("🚀 Processing files...")
        with ProcessPoolExecutor() as ex:
            for r in tqdm(ex.map(process_single_file, tasks), total=len(tasks)):
                if r: all_data.append(r)

        if len(all_data) == 0:
            print("❌ Error: Feature extraction returned empty data.")
            return

        print("🔍 SML: Clustering...")
        phys_matrix = np.array([d["phys_feat"] for d in all_data])
        phys_matrix = np.nan_to_num(phys_matrix)

        if phys_matrix.shape[0] < NUM_CLUSTERS:
            print(f"⚠️ Warning: Not enough samples for {NUM_CLUSTERS} clusters.")
            labels = np.zeros(phys_matrix.shape[0], dtype=int)
        else:
            labels = KMeans(n_clusters=NUM_CLUSTERS, random_state=SEED).fit_predict(
                StandardScaler().fit_transform(phys_matrix))

        for i, d in enumerate(all_data):
            # === 🎯 融合方案 3 & 2: 丰富的标度律先验 + IPR 拓扑先验 ===
            N_f = float(d["N"])
            g_orig = np.concatenate([
                np.array([1.0 / N_f, 1.0 / np.sqrt(N_f), float(np.log1p(N_f) / N_f)], dtype=np.float32),
                np.array([float(d["ipr_penalty"])], dtype=np.float32),
                np.asarray(d["phys_feat"], dtype=np.float32),
            ])
            cluster_one_hot = np.zeros(NUM_CLUSTERS)
            cluster_one_hot[labels[i]] = 1.0
            d["g"] = torch.tensor(np.concatenate([g_orig, cluster_one_hot]), dtype=torch.float32)
        groups = [x["folder"] for x in all_data]
        torch.save({"data": all_data, "groups": groups}, CACHE_FILE)
        print(f"💾 Cache saved to {CACHE_FILE}")

    # 2. 依次运行实验
    print("----------------------------------------------------------------")
    csv_group, r2_group = run_training_experiment(all_data, groups, split_mode="group_folder")
    print("----------------------------------------------------------------")

    print("Final Report:")
    print(f"Group (Physics) R2: {r2_group:.4f}")

def _load_or_build_validated_data_legacy():
    """Build or load the one-to-one validated CSV/GJF dataset.

    The source CSV is the sole source of labels.  Every (AtomCount, ID) key must
    map to exactly one geometry and the parsed geometry must contain that many
    atoms.  Any failure is fatal, so a run can never silently use mispaired data.
    """
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    seed_everything(SEED)

    all_data = None
    if os.path.exists(CACHE_FILE):
        try:
            cache = safe_torch_load(CACHE_FILE)
            cached_data = cache.get("data", [])
            cached_keys = [
                (int(item.get("intended_N", -1)), int(item.get("sample_id", -1)))
                for item in cached_data
            ]
            required_cache_fields = {
                "N", "intended_N", "sample_id", "source_path", "unit", "y_target",
                "coords", "adj", "node_feat", "e_ring", "phys_feat", "ipr_penalty",
            }
            cache_is_current = (
                cache.get("schema_version") == CACHE_SCHEMA_VERSION
                and cache.get("target_unit") == TARGET_UNIT
                and bool(cached_data)
                and len(cached_keys) == len(set(cached_keys))
                and all(
                    required_cache_fields.issubset(item)
                    and int(item["N"]) == int(item["intended_N"])
                    and item["unit"] == TARGET_UNIT
                    for item in cached_data
                )
            )
            if cache_is_current:
                all_data = cached_data
                print(f"Loading validated cache: {CACHE_FILE}")
            else:
                print("Existing cache has an obsolete schema or unit; rebuilding it.")
        except Exception as exc:
            print(f"Could not validate existing cache ({exc}); rebuilding it.")

    if all_data is None:
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"CSV file not found: {DATA_FILE}")
        if not os.path.exists(ROOT_FOLDER):
            raise FileNotFoundError(f"Geometry root folder not found: {ROOT_FOLDER}")

        df = pd.read_csv(DATA_FILE).copy()
        required_columns = {"ID", "AtomCount", "Gap"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

        df["AtomCount"] = pd.to_numeric(df["AtomCount"], errors="raise").astype(int)
        df["ID"] = pd.to_numeric(df["ID"], errors="raise").astype(int)
        df["Gap"] = pd.to_numeric(df["Gap"], errors="raise")
        if df[["AtomCount", "ID", "Gap"]].isna().any().any():
            raise ValueError("CSV contains missing AtomCount, ID, or Gap values.")
        if df.duplicated(["AtomCount", "ID"]).any():
            duplicates = df.loc[df.duplicated(["AtomCount", "ID"], keep=False), ["AtomCount", "ID"]]
            raise ValueError(f"CSV contains duplicate (AtomCount, ID) keys: {duplicates.head().to_dict('records')}")

        print(f"Scanning geometry tree: {os.path.abspath(ROOT_FOLDER)}")
        file_index = {}
        for root, dirs, files in os.walk(ROOT_FOLDER):
            dirs.sort()
            for filename in sorted(files):
                if not filename.lower().endswith(".gjf"):
                    continue
                full_path = os.path.join(root, filename)
                key = parse_fullerene_file_key(full_path)
                if key in file_index:
                    raise RuntimeError(
                        f"Duplicate geometry for (AtomCount, ID)={key}: "
                        f"{file_index[key]} and {full_path}"
                    )
                file_index[key] = full_path

        source_keys = [(int(row.AtomCount), int(row.ID)) for row in df.itertuples(index=False)]
        source_key_set = set(source_keys)
        missing_files = sorted(source_key_set - set(file_index))
        if missing_files:
            raise RuntimeError(
                f"Missing {len(missing_files)} geometry files required by the CSV. "
                f"First missing keys: {missing_files[:10]}"
            )
        print(f"Matched every CSV label to one geometry file ({len(source_keys)} samples).")

        tasks = [
            (
                file_index[(int(row.AtomCount), int(row.ID))],
                float(row.Gap) * HARTREE_TO_EV,
                int(row.AtomCount),
                int(row.ID),
            )
            for row in df.itertuples(index=False)
        ]

        all_data, processing_errors = [], []
        print("Extracting validated graph features...")
        with ProcessPoolExecutor(max_workers=PREPROCESS_MAX_WORKERS) as executor:
            for result in tqdm(executor.map(process_single_file, tasks), total=len(tasks), desc="Preprocessing"):
                if result is None:
                    processing_errors.append({"error": "Worker returned no result"})
                elif "error" in result:
                    processing_errors.append(result)
                else:
                    all_data.append(result)

        if processing_errors:
            error_path = os.path.join(BASE_SAVE_DIR, "preprocess_errors.json")
            with open(error_path, "w", encoding="utf-8") as handle:
                json.dump(processing_errors, handle, ensure_ascii=False, indent=2)
            raise RuntimeError(
                f"Feature extraction failed for {len(processing_errors)} samples. "
                f"Details were written to {error_path}."
            )

        processed_keys = {(int(item["intended_N"]), int(item["sample_id"])) for item in all_data}
        if len(all_data) != len(df) or len(processed_keys) != len(df) or processed_keys != source_key_set:
            raise RuntimeError("Processed data do not form a one-to-one match with the source CSV.")
        if any(int(item["N"]) != int(item["intended_N"]) for item in all_data):
            raise RuntimeError("At least one geometry atom count differs from its CSV AtomCount.")

        groups = [split_for_atom_count(int(item["intended_N"])) for item in all_data]
        torch.save(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "target_unit": TARGET_UNIT,
                "data": all_data,
                "groups": groups,
            },
            CACHE_FILE,
        )
        print(f"Validated cache saved to {CACHE_FILE}")

    return all_data


def scan_validated_raw_sources():
    """Return the source table, exact CSV-to-geometry mapping and fingerprint."""
    csv_path = Path(DATA_FILE)
    geometry_root = Path(ROOT_FOLDER)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not geometry_root.is_dir():
        raise FileNotFoundError(f"Geometry root folder not found: {geometry_root}")

    frame = pd.read_csv(csv_path).copy()
    required_columns = {"ID", "AtomCount", "Gap"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")
    frame["AtomCount"] = pd.to_numeric(frame["AtomCount"], errors="raise").astype(int)
    frame["ID"] = pd.to_numeric(frame["ID"], errors="raise").astype(int)
    frame["Gap"] = pd.to_numeric(frame["Gap"], errors="raise")
    if frame[["AtomCount", "ID", "Gap"]].isna().any().any():
        raise ValueError("CSV contains missing AtomCount, ID, or Gap values.")
    if frame.duplicated(["AtomCount", "ID"]).any():
        duplicates = frame.loc[
            frame.duplicated(["AtomCount", "ID"], keep=False), ["AtomCount", "ID"]
        ]
        raise ValueError(
            "CSV contains duplicate (AtomCount, ID) keys: "
            f"{duplicates.head().to_dict('records')}"
        )

    print(f"Scanning geometry tree: {geometry_root.resolve()}")
    file_index = {}
    for path in sorted(geometry_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".gjf":
            continue
        key = parse_fullerene_file_key(str(path))
        if key in file_index:
            raise RuntimeError(
                f"Duplicate geometry for (AtomCount, ID)={key}: "
                f"{file_index[key]} and {path}"
            )
        file_index[key] = path

    source_keys = [
        (int(row.AtomCount), int(row.ID)) for row in frame.itertuples(index=False)
    ]
    missing_files = sorted(set(source_keys) - set(file_index))
    if missing_files:
        raise RuntimeError(
            f"Missing {len(missing_files)} geometry files required by the CSV. "
            f"First missing keys: {missing_files[:10]}"
        )

    # Identical to the current raw-dataset multiseed benchmark fingerprint.
    digest = hashlib.sha256(csv_path.read_bytes())
    for key in source_keys:
        path = file_index[key]
        stat = path.stat()
        relative = path.relative_to(geometry_root).as_posix()
        digest.update(
            f"{key[0]}|{key[1]}|{relative}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
        )
    return frame, file_index, source_keys, digest.hexdigest()


def load_or_build_validated_data(rebuild_cache=False, gap_unit="hartree"):
    """Load the current raw-dataset cache or rebuild an equivalent clean cache."""
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    seed_everything(SEED)
    frame, file_index, source_keys, source_fingerprint = scan_validated_raw_sources()
    required_cache_fields = {
        "N", "intended_N", "sample_id", "source_path", "unit", "y_target",
        "coords", "adj", "node_feat", "e_ring", "phys_feat", "ipr_penalty",
    }

    all_data = None
    if os.path.exists(CACHE_FILE) and not rebuild_cache:
        try:
            cache = safe_torch_load(CACHE_FILE)
            cached_data = cache.get("data", []) if isinstance(cache, dict) else []
            cached_keys = [
                (int(item.get("intended_N", -1)), int(item.get("sample_id", -1)))
                for item in cached_data
            ]
            cache_is_current = (
                isinstance(cache, dict)
                and cache.get("schema_version") == CACHE_SCHEMA_VERSION
                and cache.get("target_unit") == TARGET_UNIT
                and cache.get("source_fingerprint") == source_fingerprint
                and cached_keys == source_keys
                and len(cached_keys) == len(set(cached_keys))
                and all(
                    required_cache_fields.issubset(item)
                    and int(item["N"]) == int(item["intended_N"])
                    and item["unit"] == TARGET_UNIT
                    for item in cached_data
                )
            )
            if cache_is_current:
                all_data = cached_data
                print(f"Loading current validated raw-dataset cache: {CACHE_FILE}")
            else:
                print("Existing cache does not match the current raw dataset; rebuilding it.")
        except Exception as exc:
            print(f"Could not validate existing cache ({exc}); rebuilding it.")

    if all_data is None:
        factor = HARTREE_TO_EV if str(gap_unit).lower() == "hartree" else 1.0
        tasks = [
            (
                str(file_index[(int(row.AtomCount), int(row.ID))]),
                float(row.Gap) * factor,
                int(row.AtomCount),
                int(row.ID),
            )
            for row in frame.itertuples(index=False)
        ]
        print(f"Extracting {len(tasks)} validated graph samples from CSV/GJF pairs...")
        if PREPROCESS_MAX_WORKERS <= 1:
            raw_results = [
                process_single_file(task) for task in tqdm(tasks, desc="Preprocessing")
            ]
        else:
            with ProcessPoolExecutor(max_workers=PREPROCESS_MAX_WORKERS) as executor:
                raw_results = list(
                    tqdm(
                        executor.map(process_single_file, tasks),
                        total=len(tasks),
                        desc="Preprocessing",
                    )
                )

        all_data, processing_errors = [], []
        for task, result in zip(tasks, raw_results):
            if result is None:
                processing_errors.append(
                    {"source_path": task[0], "error": "worker returned no result"}
                )
            elif "error" in result:
                processing_errors.append(result)
            else:
                all_data.append(result)
        if processing_errors:
            error_path = Path(BASE_SAVE_DIR) / "preprocess_errors.json"
            error_path.write_text(
                json.dumps(processing_errors, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            raise RuntimeError(
                f"Feature extraction failed for {len(processing_errors)} samples. "
                f"Details were written to {error_path}."
            )

        processed_keys = [
            (int(item["intended_N"]), int(item["sample_id"])) for item in all_data
        ]
        if processed_keys != source_keys:
            raise RuntimeError(
                "Processed data do not preserve the one-to-one CSV row/key order."
            )
        if any(int(item["N"]) != int(item["intended_N"]) for item in all_data):
            raise RuntimeError("At least one geometry atom count differs from CSV AtomCount.")
        if any(item.get("unit") != TARGET_UNIT for item in all_data):
            raise RuntimeError("At least one processed target is not in eV.")

        cache_path = Path(CACHE_FILE)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "target_unit": TARGET_UNIT,
                "source_gap_unit": gap_unit,
                "source_csv": str(Path(DATA_FILE).resolve()),
                "geometry_root": str(Path(ROOT_FOLDER).resolve()),
                "source_fingerprint": source_fingerprint,
                "data": all_data,
                "groups": [
                    split_for_atom_count(int(item["intended_N"])) for item in all_data
                ],
            },
            cache_path,
        )
        print(f"Validated raw-dataset cache saved to: {cache_path}")

    if len(all_data) != 7035:
        raise RuntimeError(f"Expected 7035 validated samples, found {len(all_data)}.")
    return all_data, source_fingerprint


def _write_run_status_disabled(lines):
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    status_path = os.path.join(BASE_SAVE_DIR, "run_status.txt")
    with open(status_path, "w", encoding="utf-8") as f:
        if isinstance(lines, str):
            f.write(lines + "\n")
        else:
            for line in lines:
                f.write(str(line) + "\n")


def write_run_status(lines):
    _write_run_status_disabled(lines)


def _main_gapstyle_disabled():
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    seed_everything(SEED)
    configure_worker_runtime()
    status_lines = [
        "START",
        f"DATA_FILE={DATA_FILE}",
        f"ROOT_FOLDER={ROOT_FOLDER}",
        f"CACHE_FILE={CACHE_FILE}",
        f"BASE_SAVE_DIR={BASE_SAVE_DIR}",
    ]
    write_run_status(status_lines)

    if os.path.exists(CACHE_FILE):
        print(f"Loading shared cache: {CACHE_FILE}")
        cache = safe_torch_load(CACHE_FILE)
        all_data, groups = cache["data"], cache["groups"]
        status_lines.append(f"CACHE_LOADED={CACHE_FILE}")
        write_run_status(status_lines)
    else:
        print("Building dataset from GJF...")
        if not os.path.exists(DATA_FILE):
            msg = f"Error: CSV file '{DATA_FILE}' not found!"
            print(msg)
            status_lines.append(msg)
            write_run_status(status_lines)
            return
        if not os.path.exists(ROOT_FOLDER):
            msg = f"Error: Root folder '{ROOT_FOLDER}' does not exist."
            print(msg)
            status_lines.append(msg)
            write_run_status(status_lines)
            return

        df = pd.read_csv(DATA_FILE)

        # === 1. 寤虹珛瓒呯骇绱㈠紩 (閽堝娣卞眰宓屽缁撴瀯浼樺寲) ===
        # 鐩爣缁撴瀯: Organized_Fullerenes/20/test/000001/xxx.gjf
        print(f"Scanning directory tree: {os.path.abspath(ROOT_FOLDER)}")
        file_index = {}
        scan_count = 0

        for root, dirs, files in os.walk(ROOT_FOLDER):
            for f in files:
                if f.lower().endswith(".gjf"):
                    full_path = os.path.join(root, f)
                    parts = os.path.normpath(full_path).split(os.sep)

                    possible_ids = []
                    possible_atoms = []

                    for part in parts:
                        if part.isdigit():
                            val = str(int(part))
                            possible_ids.append(val)
                            possible_atoms.append(val)

                    for atom_cnt in possible_atoms:
                        for mol_id in possible_ids:
                            key = (atom_cnt, mol_id)
                            if key not in file_index:
                                file_index[key] = full_path

                    scan_count += 1

        print(f"   Scanned {scan_count} .gjf files.")
        print(f"   Indexed {len(file_index)} unique (Atom, ID) combinations.")
        status_lines.extend([f"SCANNED_GJF={scan_count}", f"INDEXED_KEYS={len(file_index)}"])
        write_run_status(status_lines)

        tasks = []
        matched_count = 0
        debug_missing = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Matching Files"):
            atom_cnt = str(row["AtomCount"])
            mol_id = str(row["ID"])
            gap = float(row["Gap"])
            key = (atom_cnt, mol_id)
            if key in file_index:
                path = file_index[key]
                tasks.append((path, gap, atom_cnt, f"{mol_id}.gjf"))
                matched_count += 1
            else:
                if len(debug_missing) < 5:
                    debug_missing.append(f"Atom={atom_cnt}, ID={mol_id}")

        print(f"Matched {matched_count} files out of {len(df)}.")
        status_lines.append(f"MATCHED_TASKS={matched_count}")
        write_run_status(status_lines)
        if matched_count == 0:
            print("CRITICAL: No files matched!")
            print("   Debug Info - First 5 missing requests:", debug_missing)
            print("   Debug Info - Sample keys in index:", list(file_index.keys())[:10])
            status_lines.append(f"FIRST_MISSING={debug_missing}")
            status_lines.append(f"SAMPLE_INDEX_KEYS={list(file_index.keys())[:10]}")
            write_run_status(status_lines)
            return

        all_data = []
        print("Processing files...")
        status_lines.append(f"PREPROCESS_MAX_WORKERS={PREPROCESS_MAX_WORKERS}")
        write_run_status(status_lines)
        try:
            status_lines.append("PREPROCESS_MODE=parallel")
            write_run_status(status_lines)
            with ProcessPoolExecutor(max_workers=PREPROCESS_MAX_WORKERS) as ex:
                for idx, result in enumerate(tqdm(ex.map(process_single_file, tasks), total=len(tasks)), start=1):
                    if result is not None:
                        all_data.append(result)
                    if idx % PREPROCESS_PROGRESS_STEP == 0 or idx == len(tasks):
                        status_lines.append(f"PREPROCESS_PROGRESS={idx}/{len(tasks)}")
                        status_lines.append(f"PREPROCESS_COLLECTED={len(all_data)}")
                        write_run_status(status_lines)
        except Exception as exc:
            status_lines.append(f"PARALLEL_PREPROCESS_FAILED={type(exc).__name__}: {exc}")
            status_lines.append("PREPROCESS_MODE=serial_fallback")
            write_run_status(status_lines)
            all_data = []
            for idx, task in enumerate(tqdm(tasks, total=len(tasks), desc="Serial Processing"), start=1):
                result = process_single_file(task)
                if result is not None:
                    all_data.append(result)
                if idx % PREPROCESS_PROGRESS_STEP == 0 or idx == len(tasks):
                    status_lines.append(f"PREPROCESS_PROGRESS={idx}/{len(tasks)}")
                    status_lines.append(f"PREPROCESS_COLLECTED={len(all_data)}")
                    write_run_status(status_lines)

        if not all_data:
            msg = "Error: Feature extraction returned empty data."
            print(msg)
            status_lines.append("PROCESSED_SAMPLES=0")
            status_lines.append(msg)
            write_run_status(status_lines)
            return
        status_lines.append(f"PROCESSED_SAMPLES={len(all_data)}")
        write_run_status(status_lines)

        print("Clustering on physics features...")
        phys_matrix = np.nan_to_num(np.array([d["phys_feat"] for d in all_data], dtype=np.float32))
        if phys_matrix.shape[0] < NUM_CLUSTERS:
            print(f"Warning: not enough samples for {NUM_CLUSTERS} clusters.")
            labels = np.zeros(phys_matrix.shape[0], dtype=int)
        else:
            labels = KMeans(n_clusters=NUM_CLUSTERS, random_state=SEED).fit_predict(
                StandardScaler().fit_transform(phys_matrix)
            )

        for i, item in enumerate(all_data):
            n_val = float(item["N"])
            g_orig = np.concatenate([
                np.array([1.0 / n_val, 1.0 / np.sqrt(n_val), float(np.log1p(n_val) / n_val)], dtype=np.float32),
                np.array([float(item["ipr_penalty"])], dtype=np.float32),
                np.asarray(item["phys_feat"], dtype=np.float32),
            ])
            cluster_one_hot = np.zeros(NUM_CLUSTERS, dtype=np.float32)
            cluster_one_hot[labels[i]] = 1.0
            item["g"] = torch.tensor(np.concatenate([g_orig, cluster_one_hot]), dtype=torch.float32)

        groups = [x["folder"] for x in all_data]
        torch.save({"data": all_data, "groups": groups}, CACHE_FILE)
        print(f"Cache saved to {CACHE_FILE}")
        status_lines.append(f"CACHE_SAVED={CACHE_FILE}")
        write_run_status(status_lines)

    print("----------------------------------------------------------------")
    csv_group, r2_group = run_training_experiment(all_data, groups, split_mode="group_folder")
    print("----------------------------------------------------------------")
    print("Final Report:")
    print(f"Group (Physics) R2: {r2_group:.4f}")
    status_lines.extend([
        f"GROUP_R2={r2_group:.4f}",
        "SUCCESS",
    ])
    write_run_status(status_lines)


def parse_ablation_args():
    parser = argparse.ArgumentParser(
        description="Run clean five-seed module and input-feature ablations."
    )
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--csv", default=DATA_FILE)
    parser.add_argument("--geometry-root", default=ROOT_FOLDER)
    parser.add_argument("--cache", default=CACHE_FILE)
    parser.add_argument(
        "--gap-unit", choices=("hartree", "eV"), default="hartree",
        help="Unit of the Gap column in the source CSV.",
    )
    parser.add_argument(
        "--rebuild-cache", action="store_true",
        help="Ignore the validated cache and rebuild it from the CSV/GJF pairs.",
    )
    parser.add_argument("--output-dir", default="results/ablations")
    parser.add_argument(
        "--families", nargs="+", choices=("module", "input"),
        default=("module", "input"),
    )
    parser.add_argument(
        "--protocols", nargs="+", choices=("random", "group"),
        default=("group",),
        help=(
            "The publication default is the strict group protocol.  Pass "
            "--protocols random group to run both current paper protocols."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Optional subset of configuration names. Names must belong to the selected families.",
    )
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--preprocess-workers", type=int, default=PREPROCESS_MAX_WORKERS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--check-only", action="store_true",
        help="Validate the cache, dataset fingerprints, splits and run matrix without training.",
    )
    return parser.parse_args()


def make_split_indices(data, protocol, seed):
    if protocol == "random":
        permutation = np.random.RandomState(int(seed)).permutation(len(data))
        n_train = int(0.8 * len(data))
        n_val = int(0.1 * len(data))
        result = {
            "train": permutation[:n_train],
            "val": permutation[n_train:n_train + n_val],
            "test": permutation[n_train + n_val:],
        }
    else:
        result = {"train": [], "val": [], "test": []}
        for index, item in enumerate(data):
            n_atoms = int(item["intended_N"])
            if n_atoms != int(item["N"]):
                raise RuntimeError(
                    f"Label/geometry mismatch for C{n_atoms}, ID={item.get('sample_id')}."
                )
            if 20 <= n_atoms <= 56:
                result["train"].append(index)
            elif n_atoms in GROUP_VAL_SIZES:
                result["val"].append(index)
            elif 70 <= n_atoms <= 100:
                result["test"].append(index)
            else:
                raise RuntimeError(f"C{n_atoms} is outside the declared group protocol.")
        result = {name: np.asarray(values, dtype=np.int64) for name, values in result.items()}

    merged = np.concatenate([result["train"], result["val"], result["test"]])
    if len(merged) != len(data) or len(np.unique(merged)) != len(data):
        raise RuntimeError("The data split is incomplete or overlapping.")
    actual_counts = {name: int(len(values)) for name, values in result.items()}
    expected_counts = EXPECTED_SPLIT_COUNTS[protocol]
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Unexpected {protocol} split counts: {actual_counts}; "
            f"expected {expected_counts}."
        )
    return result


def validated_dataset_fingerprint(data):
    records = sorted(
        (
            int(item["intended_N"]),
            int(item["sample_id"]),
            round(float(item["y_target"]), 12),
            str(item.get("unit", "")),
        )
        for item in data
    )
    payload = json.dumps(records, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_ablation_manifest(data, indices, path, protocol, seed):
    split_lookup = {}
    for split_name, values in indices.items():
        for index in values:
            split_lookup[int(index)] = split_name
    rows = []
    for index, item in enumerate(data):
        rows.append({
            "row_index": index,
            "sample_id": int(item["sample_id"]),
            "intended_N": int(item["intended_N"]),
            "geometry_N": int(item["N"]),
            "split": split_lookup[index],
            "protocol": str(protocol),
            "seed": int(seed),
            "unit": TARGET_UNIT,
            "source_path": str(item["source_path"]),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    digest = hashlib.sha256(
        frame[["sample_id", "intended_N", "split"]]
        .to_csv(index=False)
        .encode("utf-8")
    ).hexdigest()
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")


def safe_r2_value(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2 or np.isclose(np.var(y_true), 0.0):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def metric_frame_from_results(frame, config_name, protocol, seed, run_fingerprint):
    required = {"true", "pred_raw", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Result table is missing columns: {sorted(missing)}")
    rows = []
    for split_name in ("train", "val", "test"):
        subset = frame[frame["split"] == split_name]
        if subset.empty:
            raise RuntimeError(f"Result table has no {split_name} rows.")
        true = subset["true"].to_numpy(dtype=np.float64)
        raw = subset["pred_raw"].to_numpy(dtype=np.float64)
        raw_error = raw - true
        rows.append({
            "config": config_name,
            "protocol": protocol,
            "seed": int(seed),
            "split": split_name,
            "count": int(len(subset)),
            "unit": TARGET_UNIT,
            "r2_raw": safe_r2_value(true, raw),
            "mae_raw": float(mean_absolute_error(true, raw)),
            "rmse_raw": float(np.sqrt(mean_squared_error(true, raw))),
            "bias_raw": float(raw_error.mean()),
            "run_fingerprint": run_fingerprint,
        })
    return pd.DataFrame(rows)


def run_size_only_baseline(data, indices, run_dir, config_name, protocol, seed, run_fingerprint):
    from sklearn.linear_model import Ridge

    run_dir.mkdir(parents=True, exist_ok=True)
    mode_name = "mode_random" if protocol == "random" else "mode_group_folder"
    mode_dir = run_dir / mode_name
    mode_dir.mkdir(parents=True, exist_ok=True)
    train_idx = indices["train"]
    x_train = np.stack([build_baseline_features(int(data[i]["intended_N"])) for i in train_idx])
    y_train = np.asarray([data[i]["y_target"] for i in train_idx], dtype=np.float64)
    model = Ridge(alpha=1.0).fit(x_train, y_train)

    rows = []
    for split_name in ("train", "val", "test"):
        for index in indices[split_name]:
            item = data[int(index)]
            features = build_baseline_features(int(item["intended_N"]))
            raw = float(model.predict(features.reshape(1, -1))[0])
            rows.append({
                "true": float(item["y_target"]),
                "pred": raw,
                "pred_raw": raw,
                "N": int(item["intended_N"]),
                "sample_id": int(item["sample_id"]),
                "split": split_name,
                "unit": TARGET_UNIT,
            })
    results = pd.DataFrame(rows)
    results.to_csv(mode_dir / "results.csv", index=False)
    write_ablation_manifest(
        data, indices, mode_dir / "dataset_manifest.csv", protocol, seed
    )
    (mode_dir / "postprocess_params.txt").write_text(
        "target_unit=eV\n"
        "postprocessing_enabled=False\n"
        "primary_output=pred_raw\n"
        "secondary_output=none\n",
        encoding="utf-8",
    )
    return metric_frame_from_results(
        results, config_name, protocol, seed, run_fingerprint
    )


def experiment_fingerprint(config_name, config, protocol, seed, dataset_fingerprint):
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = {
        "script_sha256": script_hash,
        "dataset_fingerprint": dataset_fingerprint,
        "config_name": config_name,
        "config": config,
        "protocol": protocol,
        "seed": int(seed),
        "expected_split_counts": EXPECTED_SPLIT_COUNTS[protocol],
        "target_unit": TARGET_UNIT,
        "postprocessing_enabled": False,
        "primary_prediction": "pred_raw",
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT_RATE,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "rbf_dim": RBF_DIM,
        "angle_basis_k": ANGLE_BASIS_K,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), payload


def load_completed_ablation(
    metrics_path, config_path, expected_fingerprint, expected_protocol
):
    if not metrics_path.exists() or not config_path.exists():
        return None
    try:
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(metrics_path)
    except Exception:
        return None
    if metadata.get("run_fingerprint") != expected_fingerprint:
        return None
    if set(frame.get("split", [])) != {"train", "val", "test"}:
        return None
    if set(frame.get("run_fingerprint", [])) != {expected_fingerprint}:
        return None
    if "count" not in frame.columns:
        return None
    counts = {
        str(row.split): int(row.count) for row in frame.itertuples(index=False)
    }
    if counts != EXPECTED_SPLIT_COUNTS[expected_protocol]:
        return None
    required_metrics = {"r2_raw", "mae_raw", "rmse_raw", "bias_raw"}
    if not required_metrics.issubset(frame.columns):
        return None
    if not np.isfinite(frame[list(required_metrics)].to_numpy(dtype=float)).all():
        return None
    return frame


def run_one_ablation(data, config_name, overrides, protocol, seed, output_dir,
                     dataset_fingerprint, resume):
    global BASE_SAVE_DIR, SEED

    set_ablation_config(overrides)
    config = dict(ACTIVE_ABLATION_CONFIG)
    run_dir = output_dir / "runs" / config_name / protocol / f"seed_{seed}"
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "run_config.json"
    run_fingerprint, metadata = experiment_fingerprint(
        config_name, config, protocol, seed, dataset_fingerprint
    )
    if resume:
        completed = load_completed_ablation(
            metrics_path, config_path, run_fingerprint, protocol
        )
        if completed is not None:
            print(f"Skipping completed run: {config_name}/{protocol}/seed_{seed}")
            return completed

    run_dir.mkdir(parents=True, exist_ok=True)
    metadata.update({
        "run_fingerprint": run_fingerprint,
        "primary_metrics": "raw predictions",
        "postprocessing_enabled": False,
        "secondary_metrics": "none",
    })
    config_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    indices = make_split_indices(data, protocol, seed)

    print(f"\n===== config={config_name} protocol={protocol} seed={seed} =====")
    if config["baseline_only"]:
        metrics = run_size_only_baseline(
            data, indices, run_dir, config_name, protocol, seed, run_fingerprint
        )
    else:
        old_save, old_seed = BASE_SAVE_DIR, SEED
        try:
            BASE_SAVE_DIR = str(run_dir)
            SEED = int(seed)
            seed_everything(SEED)
            groups = [split_for_atom_count(int(item["intended_N"])) for item in data]
            split_mode = "random" if protocol == "random" else "group_folder"
            results_path, _ = run_training_experiment(data, groups, split_mode=split_mode)
            results = pd.read_csv(results_path)
            metrics = metric_frame_from_results(
                results, config_name, protocol, seed, run_fingerprint
            )
            mode_dir = run_dir / f"mode_{split_mode}"
            write_ablation_manifest(
                data, indices, mode_dir / "dataset_manifest.csv", protocol, seed
            )
        finally:
            BASE_SAVE_DIR, SEED = old_save, old_seed
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    metrics.to_csv(metrics_path, index=False)
    return metrics


def aggregate_ablation_metrics(output_dir, families, selected_configs, protocols, seeds):
    frames = []
    for path in sorted((output_dir / "runs").glob("*/*/seed_*/metrics.csv")):
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            continue
    if not frames:
        raise RuntimeError("No completed ablation metrics were found.")
    all_metrics = pd.concat(frames, ignore_index=True)
    all_metrics = all_metrics.drop_duplicates(
        subset=["config", "protocol", "seed", "split"], keep="last"
    )
    all_metrics = all_metrics[
        all_metrics["config"].isin(selected_configs)
        & all_metrics["protocol"].isin(protocols)
        & all_metrics["seed"].isin(seeds)
    ].copy()
    all_metrics.to_csv(output_dir / "per_seed_metrics.csv", index=False)

    expected = {
        (config, protocol, int(seed), split)
        for config in selected_configs
        for protocol in protocols
        for seed in seeds
        for split in ("train", "val", "test")
    }
    observed = {
        (str(row.config), str(row.protocol), int(row.seed), str(row.split))
        for row in all_metrics.itertuples(index=False)
    }
    missing = sorted(expected - observed)
    if missing:
        (output_dir / "incomplete_runs.json").write_text(
            json.dumps(missing, indent=2), encoding="utf-8"
        )
        raise RuntimeError(
            f"Ablation benchmark is incomplete: {len(missing)} metric rows are missing."
        )
    incomplete_path = output_dir / "incomplete_runs.json"
    if incomplete_path.exists():
        incomplete_path.unlink()

    family_map = {}
    if "module" in families:
        for name in MODULE_CONFIGS:
            if name in selected_configs:
                family_map.setdefault(name, []).append("module")
    if "input" in families:
        for name in INPUT_CONFIGS:
            if name in selected_configs:
                family_map.setdefault(name, []).append("input")

    expanded = []
    for config_name, family_names in family_map.items():
        subset = all_metrics[all_metrics["config"] == config_name]
        for family in family_names:
            copy_frame = subset.copy()
            copy_frame.insert(0, "family", family)
            expanded.append(copy_frame)
    family_metrics = pd.concat(expanded, ignore_index=True)
    family_metrics.to_csv(output_dir / "family_per_seed_metrics.csv", index=False)

    summary_rows = []
    metric_names = ("r2_raw", "mae_raw", "rmse_raw", "bias_raw")
    for (family, config, protocol, split), subset in family_metrics.groupby(
        ["family", "config", "protocol", "split"]
    ):
        row = {
            "family": family,
            "config": config,
            "protocol": protocol,
            "split": split,
            "n_seeds": int(subset["seed"].nunique()),
            "unit": TARGET_UNIT,
        }
        for metric in metric_names:
            values = subset[metric].astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_mean_pm_std"] = (
                f"{mean:.6f} +/- {std:.6f}" if len(values) > 1 else f"{mean:.6f}"
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["family", "protocol", "split", "config"]
    )
    summary.to_csv(output_dir / "metrics_mean_std.csv", index=False)
    test_summary = summary[summary["split"] == "test"].copy()
    test_summary.to_csv(output_dir / "test_metrics_mean_std.csv", index=False)
    return test_summary


def plot_ablation_summaries(test_summary, output_dir):
    if test_summary.empty:
        return
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for (family, protocol), subset in test_summary.groupby(["family", "protocol"]):
        subset = subset.copy()
        order = list(MODULE_CONFIGS) if family == "module" else list(INPUT_CONFIGS)
        subset["order"] = subset["config"].map({name: i for i, name in enumerate(order)})
        subset = subset.sort_values("order")
        x = np.arange(len(subset))
        means = subset["r2_raw_mean"].to_numpy(dtype=float)
        errors = subset["r2_raw_std"].fillna(0.0).to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(max(8.0, 1.45 * len(subset)), 5.5))
        ax.bar(x, means, yerr=errors, capsize=4, color="#4472C4", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(subset["config"], rotation=25, ha="right")
        ax.set_ylabel(r"Test $R^2$ (raw prediction)")
        ax.set_title(f"{family.title()} ablation — {protocol} protocol")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{family}_{protocol}_raw_r2_mean_std.png", dpi=300)
        plt.close(fig)


def main():
    global ROOT_FOLDER, DATA_FILE, CACHE_FILE, BASE_SAVE_DIR
    global MAX_EPOCHS, PATIENCE, PREPROCESS_MAX_WORKERS

    args = parse_ablation_args()
    project_dir = args.project_dir.resolve()
    os.chdir(project_dir)
    DATA_FILE = str((project_dir / args.csv).resolve()) if not Path(args.csv).is_absolute() else str(Path(args.csv))
    ROOT_FOLDER = str((project_dir / args.geometry_root).resolve()) if not Path(args.geometry_root).is_absolute() else str(Path(args.geometry_root))
    CACHE_FILE = str((project_dir / args.cache).resolve()) if not Path(args.cache).is_absolute() else str(Path(args.cache))
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (project_dir / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    BASE_SAVE_DIR = str(output_dir)
    MAX_EPOCHS = int(args.max_epochs)
    PATIENCE = int(args.patience)
    PREPROCESS_MAX_WORKERS = int(args.preprocess_workers)

    available_configs = {}
    if "module" in args.families:
        available_configs.update(MODULE_CONFIGS)
    if "input" in args.families:
        for name, config in INPUT_CONFIGS.items():
            if name in available_configs and available_configs[name] != config:
                raise RuntimeError(f"Conflicting definitions for configuration {name}.")
            available_configs[name] = config
    if args.configs is None:
        selected_configs = list(available_configs)
    else:
        unknown = sorted(set(args.configs) - set(available_configs))
        if unknown:
            raise ValueError(f"Unknown configurations for selected families: {unknown}")
        selected_configs = list(dict.fromkeys(args.configs))

    protocols = tuple(dict.fromkeys(str(value) for value in args.protocols))
    seeds = tuple(dict.fromkeys(int(value) for value in args.seeds))
    data, dataset_fingerprint = load_or_build_validated_data(
        rebuild_cache=bool(args.rebuild_cache), gap_unit=args.gap_unit
    )
    content_fingerprint = validated_dataset_fingerprint(data)
    if len(data) != 7035:
        raise RuntimeError(f"Expected 7035 validated samples, found {len(data)}.")
    provenance = {
        "source_csv": DATA_FILE,
        "geometry_root": ROOT_FOLDER,
        "cache": CACHE_FILE,
        "source_gap_unit": args.gap_unit,
        "sample_count": len(data),
        "training_target_unit": TARGET_UNIT,
        "dataset_fingerprint": dataset_fingerprint,
        "content_fingerprint": content_fingerprint,
        "families": list(args.families),
        "configs": selected_configs,
        "protocols": list(protocols),
        "seeds": list(seeds),
        "expected_split_counts": EXPECTED_SPLIT_COUNTS,
        "postprocessing_enabled": False,
        "primary_metrics": "raw predictions",
        "secondary_metrics": "none",
    }
    (output_dir / "dataset_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(
        f"Validated samples: {len(data)}; unit: {TARGET_UNIT}\n"
        f"Dataset fingerprint: {dataset_fingerprint}\n"
        f"Configurations: {', '.join(selected_configs)}"
    )

    # Save one protocol/seed manifest independently of the ablation configuration.
    for protocol in protocols:
        for seed in seeds:
            indices = make_split_indices(data, protocol, seed)
            manifest_path = output_dir / "manifests" / protocol / f"seed_{seed}.csv"
            write_ablation_manifest(
                data, indices, manifest_path, protocol, seed
            )

    planned_runs = len(selected_configs) * len(protocols) * len(seeds)
    print(
        f"Protocols: {', '.join(protocols)}; seeds: {', '.join(map(str, seeds))}\n"
        f"Planned configuration/protocol/seed runs: {planned_runs}"
    )
    if args.check_only:
        print("CHECK ONLY: dataset, fingerprints, manifests and split counts are valid.")
        return

    for config_name in selected_configs:
        overrides = available_configs[config_name]
        for protocol in protocols:
            for seed in seeds:
                run_one_ablation(
                    data,
                    config_name,
                    overrides,
                    protocol,
                    int(seed),
                    output_dir,
                    dataset_fingerprint,
                    args.resume,
                )

    test_summary = aggregate_ablation_metrics(
        output_dir,
        tuple(args.families),
        selected_configs,
        protocols,
        seeds,
    )
    plot_ablation_summaries(test_summary, output_dir)
    print("\nPrimary five-seed test summary (raw predictions)")
    print(test_summary[[
        "family", "protocol", "config", "n_seeds",
        "r2_raw_mean_pm_std", "mae_raw_mean_pm_std", "rmse_raw_mean_pm_std",
    ]].to_string(index=False))
    print(f"\nCompleted: {output_dir}")


if __name__ == "__main__":
    main()
