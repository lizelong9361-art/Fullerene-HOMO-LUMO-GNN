#!/usr/bin/env python
"""Extract the ten gap2.py-aligned fullerene physical descriptors.

The source CSV supplies the authoritative (AtomCount, ID) sample keys and DFT
gap labels.  Each key is matched to one Gaussian input geometry, after which
the molecular graph and descriptors are reconstructed using the same
definitions as gap2.py.  The resulting table is suitable as direct input to
plot_physical_gap_analysis.py.
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.linalg
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors


SCRIPT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA = SCRIPT_DIR / "data" / "dataset.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "physical_analysis" / "fullerene_physical_descriptors.csv"
FALLBACK_GEOMETRY_ROOT = SCRIPT_DIR / "data" / "structures"

DESCRIPTOR_COLUMNS = (
    "mean_radius",
    "radial_std",
    "asphericity",
    "volume_per_atom",
    "area_scaled",
    "isoperimetric_deviation",
    "pentagon_adjacency",
    "spectral_lower_sum",
    "spectral_std",
    "central_spectral_gap",
)


def default_geometry_root() -> Path:
    local = SCRIPT_DIR / "data" / "structures"
    return local if local.is_dir() else FALLBACK_GEOMETRY_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the ten fullerene descriptors used in gap2.py."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--geometry-root", type=Path, default=default_geometry_root()
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of extraction processes (default: 1).",
    )
    return parser.parse_args()


def parse_numeric_token(token: str) -> int | None:
    match = re.fullmatch(r"(?:C)?(\d+)", str(token), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def geometry_key(path: Path, geometry_root: Path) -> tuple[int, int]:
    """Parse Organized_Fullerenes/<N>/<split>/<ID>/<file>.gjf."""
    relative = path.relative_to(geometry_root)
    directories = relative.parts[:-1]
    if not directories:
        raise ValueError(f"Geometry is not below the root: {path}")

    atom_count = parse_numeric_token(directories[0])
    if atom_count is None:
        raise ValueError(f"Cannot parse cage size from: {path}")

    split_tokens = {"train", "val", "valid", "validation", "test"}
    identifier_index = 2 if (
        len(directories) > 1 and directories[1].lower() in split_tokens
    ) else 1
    if len(directories) <= identifier_index:
        match = re.fullmatch(rf"C{atom_count}_(\d+)\.gjf", path.name, flags=re.IGNORECASE)
        if len(directories) == 1 and match:
            return atom_count, int(match.group(1))
        raise ValueError(f"Cannot parse isomer ID from: {path}")

    sample_id = parse_numeric_token(directories[identifier_index])
    if sample_id is None:
        raise ValueError(f"Cannot parse isomer ID from: {path}")
    return atom_count, sample_id


def index_geometries(geometry_root: Path) -> dict[tuple[int, int], Path]:
    if not geometry_root.is_dir():
        raise FileNotFoundError(f"Geometry root was not found: {geometry_root}")

    index: dict[tuple[int, int], Path] = {}
    for path in geometry_root.rglob("*.gjf"):
        key = geometry_key(path, geometry_root)
        if key in index:
            raise ValueError(
                f"Duplicate geometry for (AtomCount, ID)={key}: "
                f"{index[key]} and {path}"
            )
        index[key] = path
    return index


def read_gjf_coordinates(path: Path) -> np.ndarray:
    """Read coordinates after the Gaussian charge/multiplicity line."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start_index: int | None = None
    for index, line in enumerate(lines[:-1]):
        tokens = line.split()
        if len(tokens) != 2:
            continue
        try:
            int(tokens[0])
            int(tokens[1])
            next_tokens = lines[index + 1].split()
            if len(next_tokens) >= 4:
                tuple(float(value) for value in next_tokens[-3:])
                start_index = index + 1
                break
        except ValueError:
            continue

    if start_index is None:
        raise ValueError("Gaussian charge/multiplicity line was not found")

    coordinates: list[list[float]] = []
    for line in lines[start_index:]:
        if not line.strip():
            break
        tokens = line.split()
        if len(tokens) < 4:
            continue
        try:
            coordinates.append([float(value) for value in tokens[-3:]])
        except ValueError:
            continue

    result = np.asarray(coordinates, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("No valid Cartesian coordinate block was parsed")
    return result


def reconstruct_graph(coordinates: np.ndarray) -> tuple[np.ndarray, nx.Graph]:
    """Reproduce the three-nearest-neighbour construction in gap2.py."""
    atom_count = len(coordinates)
    neighbours = NearestNeighbors(
        n_neighbors=4, algorithm="ball_tree"
    ).fit(coordinates)
    _, indices = neighbours.kneighbors(coordinates)

    adjacency = np.zeros((atom_count, atom_count), dtype=float)
    graph = nx.Graph()
    graph.add_nodes_from(range(atom_count))
    for atom_index in range(atom_count):
        for neighbour_index in indices[atom_index, 1:]:
            neighbour_index = int(neighbour_index)
            adjacency[atom_index, neighbour_index] = 1.0
            adjacency[neighbour_index, atom_index] = 1.0
            graph.add_edge(atom_index, neighbour_index)
    return adjacency, graph


def fullerene_faces(graph: nx.Graph) -> list[list[int]]:
    """Return planar faces and apply the same fullerene checks as gap2.py."""
    atom_count = graph.number_of_nodes()
    if not nx.is_connected(graph):
        raise ValueError("reconstructed graph is disconnected")
    if any(degree != 3 for _, degree in graph.degree()):
        raise ValueError("reconstructed graph is not three-regular")
    if graph.number_of_edges() != 3 * atom_count // 2:
        raise ValueError("reconstructed graph has an invalid number of edges")

    is_planar, embedding = nx.check_planarity(graph)
    if not is_planar:
        raise ValueError("reconstructed graph is not planar")

    visited_half_edges: set[tuple[int, int]] = set()
    faces: list[list[int]] = []
    for first, second in embedding.edges():
        if (first, second) not in visited_half_edges:
            face = embedding.traverse_face(first, second, visited_half_edges)
            if len(face) >= 3:
                faces.append(face)

    expected_faces = atom_count // 2 + 2
    expected_hexagons = atom_count // 2 - 10
    face_sizes = [len(face) for face in faces]
    if len(faces) != expected_faces:
        raise ValueError(f"expected {expected_faces} faces, found {len(faces)}")
    if any(size not in (5, 6) for size in face_sizes):
        raise ValueError("face set contains a non-5/6-membered ring")
    if face_sizes.count(5) != 12 or face_sizes.count(6) != expected_hexagons:
        raise ValueError("face counts do not satisfy fullerene topology")
    return faces


def count_pentagon_adjacencies(
    graph: nx.Graph, faces: list[list[int]]
) -> float:
    edge_to_face_sizes: dict[tuple[int, int], list[int]] = {}
    for face in faces:
        face_size = len(face)
        for index, first in enumerate(face):
            second = face[(index + 1) % face_size]
            edge = tuple(sorted((first, second)))
            edge_to_face_sizes.setdefault(edge, []).append(face_size)

    count = 0
    for first, second in graph.edges():
        if edge_to_face_sizes.get(tuple(sorted((first, second))), []).count(5) == 2:
            count += 1
    return float(count)


def calculate_descriptors(
    adjacency: np.ndarray,
    graph: nx.Graph,
    coordinates: np.ndarray,
) -> dict[str, float]:
    """Use the exact formulae and ordering of gap2.py."""
    atom_count = len(coordinates)

    eigenvalues = np.sort(scipy.linalg.eigvalsh(adjacency))
    occupied_count = atom_count // 2
    spectral_lower_sum = float(
        np.sum(eigenvalues[:occupied_count]) * 2.0 / atom_count
    )
    spectral_std = float(np.std(eigenvalues))
    central_spectral_gap = float(
        eigenvalues[occupied_count] - eigenvalues[occupied_count - 1]
    )

    hull = ConvexHull(coordinates)
    volume = float(hull.volume)
    area = float(hull.area)
    isoperimetric_deviation = float(
        1.0 - (36.0 * np.pi * volume**2) / (area**3 + 1e-8)
    )

    centred = coordinates - np.mean(coordinates, axis=0)
    radial_distances = np.linalg.norm(centred, axis=1)
    inertia = centred.T @ centred
    inertia_eigenvalues = np.linalg.eigvalsh(inertia)
    asphericity = float(
        (inertia_eigenvalues[-1] - inertia_eigenvalues[0])
        / (inertia_eigenvalues[-1] + 1e-8)
    )

    faces = fullerene_faces(graph)
    pentagon_adjacency = count_pentagon_adjacencies(graph, faces)

    return {
        "mean_radius": float(np.mean(radial_distances)),
        "radial_std": float(np.std(radial_distances)),
        "asphericity": asphericity,
        "volume_per_atom": volume / atom_count,
        "area_scaled": area / (atom_count ** (2.0 / 3.0)),
        "isoperimetric_deviation": isoperimetric_deviation,
        "pentagon_adjacency": pentagon_adjacency,
        "spectral_lower_sum": spectral_lower_sum,
        "spectral_std": spectral_std,
        "central_spectral_gap": central_spectral_gap,
    }


def extract_one(task: tuple[int, int, float, str]) -> dict[str, object]:
    atom_count, sample_id, gap, source_path = task
    try:
        path = Path(source_path)
        coordinates = read_gjf_coordinates(path)
        if len(coordinates) != atom_count:
            raise ValueError(
                f"coordinate count {len(coordinates)} does not match C{atom_count}"
            )
        adjacency, graph = reconstruct_graph(coordinates)
        descriptors = calculate_descriptors(adjacency, graph, coordinates)
        return {
            "ID": sample_id,
            "AtomCount": atom_count,
            "Gap": gap,
            "source_path": str(path),
            **descriptors,
            "error": "",
        }
    except Exception as error:
        return {
            "ID": sample_id,
            "AtomCount": atom_count,
            "Gap": gap,
            "source_path": source_path,
            **{column: np.nan for column in DESCRIPTOR_COLUMNS},
            "error": str(error),
        }


def build_descriptor_table(
    data_path: Path,
    geometry_root: Path,
    output_path: Path,
    workers: int = 1,
) -> Path:
    data_path = Path(data_path).resolve()
    geometry_root = Path(geometry_root).resolve()
    output_path = Path(output_path).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Source CSV was not found: {data_path}")

    source = pd.read_csv(data_path)
    required = {"ID", "AtomCount", "Gap"}
    missing_columns = required.difference(source.columns)
    if missing_columns:
        raise ValueError(
            f"Source CSV is missing required columns: {sorted(missing_columns)}"
        )

    source = source.copy()
    source["ID"] = pd.to_numeric(source["ID"], errors="raise").astype(int)
    source["AtomCount"] = pd.to_numeric(
        source["AtomCount"], errors="raise"
    ).astype(int)
    source["Gap"] = pd.to_numeric(source["Gap"], errors="raise")
    if source.duplicated(["AtomCount", "ID"]).any():
        raise ValueError("Source CSV contains duplicate (AtomCount, ID) keys")

    print(f"Indexing Gaussian geometries under: {geometry_root}")
    file_index = index_geometries(geometry_root)
    source_keys = [
        (int(row.AtomCount), int(row.ID))
        for row in source.itertuples(index=False)
    ]
    missing_files = [key for key in source_keys if key not in file_index]
    if missing_files:
        preview = ", ".join(str(key) for key in missing_files[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_files)} geometries required by the CSV: {preview}"
        )

    tasks = [
        (
            int(row.AtomCount),
            int(row.ID),
            float(row.Gap),
            str(file_index[(int(row.AtomCount), int(row.ID))]),
        )
        for row in source.itertuples(index=False)
    ]
    print(f"Extracting ten descriptors for {len(tasks):,} structures...")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(extract_one, tasks, chunksize=20))
    else:
        rows = [extract_one(task) for task in tasks]

    result = pd.DataFrame(rows)
    failures = result[result["error"].astype(str).str.len().gt(0)]
    if not failures.empty:
        failure_path = output_path.with_name(
            output_path.stem + "_failures.csv"
        )
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failures.to_csv(failure_path, index=False)
        raise RuntimeError(
            f"Descriptor extraction failed for {len(failures)} structures. "
            f"Details: {failure_path}"
        )

    result = result.drop(columns="error")
    if len(result) != len(source) or result[list(DESCRIPTOR_COLUMNS)].isna().any().any():
        raise RuntimeError("The descriptor table is incomplete")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved complete descriptor table: {output_path}")
    print(f"Rows: {len(result):,}; descriptor columns: {len(DESCRIPTOR_COLUMNS)}")
    return output_path


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    build_descriptor_table(
        args.data, args.geometry_root, args.output, workers=args.workers
    )


if __name__ == "__main__":
    main()
