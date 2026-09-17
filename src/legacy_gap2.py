import os
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
from sklearn.linear_model import LinearRegression
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# =========================
# 🔧 Global Configuration
# =========================
# ⚠️ Ensure these match your server paths
ROOT_FOLDER = "Organized_Fullerenes"
DATA_FILE = "sorted_fullerene_data.csv"
# A clean cache must never reuse samples created by the legacy path matcher.
CACHE_SCHEMA_VERSION = 4
CACHE_FILE = "processed_data_gjf_clean_v4.pt"

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
# Validation-only affine calibration is reported as a secondary sensitivity
# analysis.  Checkpoint selection and primary paper metrics always use the raw
# model predictions; test labels are never used to fit calibration parameters.
ENABLE_POSTPROCESSING = True
POSTPROCESSING_MODE = "validation_affine"
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
BASE_SAVE_DIR = "checkpoints_final_gjf_fixed_restore_diag_shift"


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
    return "test"


def build_baseline_features_from_item(item) -> np.ndarray:
    return build_baseline_features(int(item["N"]))


def build_structural_global_feature(item, cluster_label: int) -> np.ndarray:
    """Build only descriptors available from the validated molecular graph/geometry.

    The KMeans label is assigned by a model fitted on the training split only.
    No DFT electronic properties (for example HOMO, LUMO, or dipole moment) are used.
    """
    atom_count = float(item["N"])
    base = np.concatenate([
        np.array([
            1.0 / atom_count,
            1.0 / np.sqrt(atom_count),
            float(np.log1p(atom_count) / atom_count),
        ], dtype=np.float32),
        np.array([float(item["ipr_penalty"])], dtype=np.float32),
        np.asarray(item["phys_feat"], dtype=np.float32),
    ])
    cluster_one_hot = np.zeros(NUM_CLUSTERS, dtype=np.float32)
    cluster_one_hot[int(cluster_label)] = 1.0
    return np.concatenate([base, cluster_one_hot]).astype(np.float32)


def build_augmented_global_feature(item, base_y: float, cluster_label: int) -> np.ndarray:
    raw_g = build_structural_global_feature(item, cluster_label)
    atom_count = float(item["N"])
    extra = np.array([
        base_y,
        base_y * np.log1p(atom_count),
    ], dtype=np.float32)
    return np.concatenate([raw_g, extra], axis=0)


def restore_target_value(target_norm, atom_count, base_model, global_mu_res, global_sigma_res):
    base_y = float(base_model.predict(build_baseline_features(atom_count).reshape(1, -1))[0])
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
    base_y = float(base_model.predict(build_baseline_features(atom_count).reshape(1, -1))[0])
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


def fit_affine_calibrator(preds, trues):
    preds = np.asarray(preds, dtype=np.float32)
    trues = np.asarray(trues, dtype=np.float32)
    if len(preds) < 8:
        return 1.0, 0.0

    reg = LinearRegression()
    reg.fit(preds.reshape(-1, 1), trues)
    slope = float(np.clip(reg.coef_[0], CALIBRATION_SLOPE_CLAMP[0], CALIBRATION_SLOPE_CLAMP[1]))
    intercept = float(np.clip(reg.intercept_, -CALIBRATION_INTERCEPT_CLAMP, CALIBRATION_INTERCEPT_CLAMP))
    return slope, intercept


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
        Ns.append(item["N"])
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
                 dropout=0.1):
        super().__init__()
        self.hidden = hidden_dim
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
        e_feat = torch.cat([data["edge_ring"], rbf_e], dim=1)
        m = self.edge_emb(e_feat)

        for ln in self.layers:
            agg = torch.zeros_like(h)
            agg.index_add_(0, edge_dst, m)
            h = ln(h + self.drop(F.silu(agg)))

            hi, hj = h[edge_src], h[edge_dst]
            if data["triplet_ea"].numel() > 0:
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
            else:
                test_idx.append(i)

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

    all_train_N = np.array([all_data[i]["N"] for i in train_idx])
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

    manifest = pd.DataFrame([
        {
            "sample_id": int(item["sample_id"]),
            "intended_N": int(item["intended_N"]),
            "geometry_N": int(item["N"]),
            "split": split_for_atom_count(int(item["intended_N"])),
            "gap_unit": item["unit"],
            "source_path": item["source_path"],
        }
        for item in all_data
    ])
    manifest_file = os.path.join(EXP_DIR, "dataset_manifest.csv")
    manifest.to_csv(manifest_file, index=False)

    from sklearn.linear_model import Ridge

    all_train_N = np.array([all_data[i]["N"] for i in train_idx])
    all_train_y = np.array([all_data[i]["y_target"] for i in train_idx])

    X_base_train = np.stack([build_baseline_features_from_item(all_data[i]) for i in train_idx], axis=0)
    base_model = Ridge(alpha=1.0).fit(X_base_train, all_train_y)

    train_residuals = all_train_y - base_model.predict(X_base_train)
    global_mu_res = float(np.mean(train_residuals))
    global_sigma_res = float(max(np.std(train_residuals), BASELINE_MIN_RES_STD))

    # Fit every unsupervised transformation on the training split only.
    train_phys = np.asarray([all_data[i]["phys_feat"] for i in train_idx], dtype=np.float32)
    all_phys = np.asarray([item["phys_feat"] for item in all_data], dtype=np.float32)
    phys_scaler = StandardScaler().fit(train_phys)
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=SEED, n_init=20)
    kmeans.fit(phys_scaler.transform(train_phys))
    cluster_labels = kmeans.predict(phys_scaler.transform(all_phys))

    g_vals = []
    for i in train_idx:
        base_feat = build_baseline_features_from_item(all_data[i])
        base_y = float(base_model.predict(base_feat.reshape(1, -1))[0])
        g_vals.append(build_augmented_global_feature(all_data[i], base_y, int(cluster_labels[i])))
    scaler_g = StandardScaler().fit(np.array(g_vals))
    train_count_map = pd.Series(all_train_N).value_counts().to_dict()
    mean_train_count = float(np.mean(list(train_count_map.values())))

    def prepare_dataset(indices, is_train):
        ds = []
        for i in indices:
            it = dict(all_data[i])
            atom_count = int(it["N"])
            base_feat = build_baseline_features_from_item(all_data[i])
            base_y = float(base_model.predict(base_feat.reshape(1, -1))[0])
            it["y_target"] = float((it["y_target"] - base_y - global_mu_res) / global_sigma_res)
            g_numpy = build_augmented_global_feature(it, base_y, int(cluster_labels[i])).reshape(1, -1)
            it["g"] = torch.tensor(scaler_g.transform(g_numpy)[0], dtype=torch.float32)
            it["base_feat"] = torch.tensor(base_feat, dtype=torch.float32)
            if split_mode == "group_folder" and is_train and USE_SIZE_BALANCED_LOSS:
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

    model = DimeNetLite(node_dim, edge_dim, g_dim, HIDDEN_DIM, RBF_DIM, ANGLE_BASIS_K, NUM_LAYERS, DROPOUT_RATE).to(
        DEVICE
    )
    write_exp_status(
        "MODEL_INITIALIZED",
        f"DEVICE={DEVICE}",
        f"node_dim={node_dim} edge_dim={edge_dim} global_dim={g_dim}",
    )

    if USE_SAM:
        optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, rho=SAM_RHO)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer.base_optimizer if USE_SAM else optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6,
    )
    criterion = nn.MSELoss(reduction="none")

    save_path = os.path.join(EXP_DIR, "best_model.pth")
    last_save_path = os.path.join(EXP_DIR, "last_model.pth")
    early_stop = EarlyStopping(patience=PATIENCE, verbose=False, path=save_path)
    best_cal_slope = 1.0
    best_cal_intercept = 0.0

    def compute_batch_loss(batch):
        pred = model(batch)
        mse_loss = criterion(pred, batch["y"])
        huber_loss = F.smooth_l1_loss(pred, batch["y"], reduction="none", beta=0.5)
        per_sample_loss = (1.0 - HUBER_MIX) * mse_loss + HUBER_MIX * huber_loss
        if split_mode == "group_folder" and USE_SIZE_BALANCED_LOSS:
            base_loss = (per_sample_loss * batch["sample_weight"]).mean()
        else:
            base_loss = per_sample_loss.mean()

        if pred.numel() > 1:
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
                    base_y = float(base_model.predict(batch_base_features[i].reshape(1, -1))[0])
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

    def collect_rows(loader, split_name, cal_slope, cal_intercept):
        trues, base_preds, raw_residuals, atom_sizes = collect_raw_outputs(loader)
        raw_preds = np.asarray(base_preds, dtype=np.float32) + np.asarray(raw_residuals, dtype=np.float32)
        if ENABLE_POSTPROCESSING:
            preds = cal_slope * raw_preds + cal_intercept
        else:
            preds = raw_preds.copy()
        preds = np.asarray(preds, dtype=np.float32)
        rows_local = []
        for i in range(len(preds)):
            raw_pred = float(raw_preds[i])
            calibrated_pred = float(preds[i])
            rows_local.append(
                {
                    "true": float(trues[i]),
                    # ``pred`` is retained as a backward-compatible alias for
                    # the calibrated output.  ``pred_raw`` is the primary
                    # prediction used for the main paper metrics.
                    "pred": calibrated_pred,
                    "pred_calibrated": calibrated_pred,
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

    # Fit calibration exactly once after raw-validation checkpoint selection.
    # Only validation predictions and labels are used here.  The fitted affine
    # map is then frozen before train/validation/test predictions are exported.
    val_trues_cal, val_base_cal, val_res_cal, _ = collect_raw_outputs(val_loader)
    val_preds_raw_cal = np.asarray(val_base_cal, dtype=np.float32) + np.asarray(val_res_cal, dtype=np.float32)
    if ENABLE_POSTPROCESSING:
        best_cal_slope, best_cal_intercept = fit_affine_calibrator(val_preds_raw_cal, val_trues_cal)
    else:
        best_cal_slope, best_cal_intercept = 1.0, 0.0
    write_exp_status(
        f"POSTPROCESSING_MODE={POSTPROCESSING_MODE if ENABLE_POSTPROCESSING else 'disabled'}",
        "CALIBRATION_FIT_SPLIT=validation",
        "TEST_LABELS_USED_FOR_CALIBRATION=False",
        f"CALIBRATION_SLOPE={best_cal_slope:.10f}",
        f"CALIBRATION_INTERCEPT={best_cal_intercept:.10f}",
    )

    rows = []
    rows.extend(collect_rows(
        train_loader,
        "train",
        best_cal_slope,
        best_cal_intercept,
    ))
    rows.extend(collect_rows(
        val_loader,
        "val",
        best_cal_slope,
        best_cal_intercept,
    ))
    rows.extend(collect_rows(
        test_loader,
        "test",
        best_cal_slope,
        best_cal_intercept,
    ))

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
            "r2_calibrated": safe_r2(sub_df["true"], sub_df["pred_calibrated"]),
            "mae_calibrated": float(mean_absolute_error(sub_df["true"], sub_df["pred_calibrated"])),
            "rmse_calibrated": float(np.sqrt(mean_squared_error(sub_df["true"], sub_df["pred_calibrated"]))),
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
        f.write("calibration_fit_split=validation\n")
        f.write("checkpoint_selection_predictions=raw\n")
        f.write("test_labels_used_for_calibration=False\n")
        f.write("primary_metrics=raw_model_predictions\n")
        f.write("secondary_metrics=validation_affine_calibrated_predictions\n")
        f.write(f"best_cal_slope={best_cal_slope}\n")
        f.write(f"best_cal_intercept={best_cal_intercept}\n")
        f.write(f"calibration_slope_clamp={CALIBRATION_SLOPE_CLAMP}\n")
        f.write(f"calibration_intercept_clamp={CALIBRATION_INTERCEPT_CLAMP}\n")
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
            f.write(f"overall_r2_calibrated={safe_r2(sub_df['true'], sub_df['pred_calibrated'])}\n")
            f.write(f"overall_mae_calibrated={mean_absolute_error(sub_df['true'], sub_df['pred_calibrated'])}\n")
            f.write(f"overall_rmse_calibrated={np.sqrt(mean_squared_error(sub_df['true'], sub_df['pred_calibrated']))}\n")
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
                "r2_calibrated": safe_r2(sub_df["true"], sub_df["pred_calibrated"]),
                "mae_calibrated": float(mean_absolute_error(sub_df["true"], sub_df["pred_calibrated"])),
                "rmse_calibrated": float(np.sqrt(mean_squared_error(sub_df["true"], sub_df["pred_calibrated"]))),
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
    final_r2_calibrated = r2_score(test_df["true"], test_df["pred_calibrated"])
    print(f"Test R2 (raw, primary): {final_r2_raw:.4f}")
    print(f"Test R2 (validation-affine calibrated, secondary): {final_r2_calibrated:.4f}")
    write_exp_status(
        f"TEST_R2_RAW_PRIMARY={final_r2_raw:.6f}",
        f"TEST_R2_CALIBRATED_SECONDARY={final_r2_calibrated:.6f}",
        "SUCCESS",
    )
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

def main():
    """Build a traceable dataset and run the predefined size-distribution split.

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
            cache_is_current = (
                cache.get("schema_version") == CACHE_SCHEMA_VERSION
                and cache.get("target_unit") == TARGET_UNIT
                and bool(cached_data)
                and all(
                    {"N", "intended_N", "sample_id", "source_path", "unit"}.issubset(item)
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

    groups = [split_for_atom_count(int(item["intended_N"])) for item in all_data]
    print("----------------------------------------------------------------")
    _, r2_group = run_training_experiment(all_data, groups, split_mode="group_folder")
    print("----------------------------------------------------------------")
    print(f"Final report -- group (physics) R2: {r2_group:.4f}")


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


if __name__ == "__main__":
    main()
