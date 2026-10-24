"""Prepare benchmark inputs in the FedMACD layout.

Continuous-input preparation requires local source checkouts, which this
script does not modify. BIF download is an explicit separate option.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd


MATMCD_FILES = {
    "Auto_MPG": "Auto MPG",
    "DWD_climate": "DWD climate",
    "Sachs": "Sachs",
}

BN_HASHES = {
    "asia": "9f770c96940dc4d860b581602120790ac538983b30b28d38631b8184dc7b5f89",
    "child": "432c22e661cd0e8235c8d95599ac22ccc261002459f5c4e6e7235b057908fe6e",
    "alarm": "701e6c561f71b55669070c29614f0724b761289aa2c4a35bcc97b638ee881fa2",
    "survey": "3698ccc0f1f74b9f935617d91989e4cdb39edb5360b558216af64047fc88b3b0",
    "earthquake": "74dd131e2c6437a39de17f2963d2d16fadd91e70450d1fc20501df05e8dc4f2e",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_graph(source: Path, destination: Path, transpose: bool, newline: str) -> None:
    graph = np.loadtxt(source, delimiter=",")
    if transpose:
        graph = graph.T
    if not np.isin(graph, (0, 1)).all():
        raise ValueError(f"Expected a binary adjacency matrix: {source}")
    with destination.open("w", encoding="utf-8", newline="") as stream:
        np.savetxt(stream, graph, fmt="%d", delimiter=" ", newline=newline)


def prepare(matmcd_root: Path, fedpc_root: Path, notears_root: Path, output_root: Path) -> None:
    matmcd_data = matmcd_root / "benchmark datasets"
    matmcd_graphs = matmcd_root / "ground truth matrix"
    fedpc_data = fedpc_root / "FedPC-code (matlab)" / "dataset"
    sys.path.insert(0, str(notears_root.resolve()))
    from notears_admm import utils

    destinations = [
        output_root / dataset / filename
        for dataset in (*MATMCD_FILES, "10nodes", "ER20")
        for filename in ("data.csv", "graph.txt")
    ]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing prepared inputs: "
            + ", ".join(str(path) for path in existing)
        )

    outputs: list[Path] = []
    for dataset, source_name in MATMCD_FILES.items():
        out = output_root / dataset
        out.mkdir(parents=True, exist_ok=True)
        source_csv = matmcd_data / f"{source_name} data.csv"
        source_graph = matmcd_graphs / f"GTmatrix_{source_name} data.csv"
        data_out = out / "data.csv"
        graph_out = out / "graph.txt"
        pd.read_csv(source_csv).to_csv(data_out, index=False, lineterminator="\r\n")
        write_graph(source_graph, graph_out, transpose=True,
                    newline="\n" if dataset == "Auto_MPG" else "\r\n")
        outputs.extend((data_out, graph_out))

    out = output_root / "10nodes"
    out.mkdir(parents=True, exist_ok=True)
    data_out = out / "data.csv"
    graph_out = out / "graph.txt"
    pd.read_csv(fedpc_data / "10nodes_5000.txt", sep=r"\s+", header=None).to_csv(
        data_out, header=False, index=False, lineterminator="\r\n"
    )
    with graph_out.open("w", encoding="utf-8", newline="") as stream:
        np.savetxt(stream, np.loadtxt(fedpc_data / "10_DAG.txt"), fmt="%d",
                   delimiter=" ", newline="\r\n")
    outputs.extend((data_out, graph_out))

    out = output_root / "ER20"
    out.mkdir(parents=True, exist_ok=True)
    data_out = out / "data.csv"
    graph_out = out / "graph.txt"
    utils.set_random_seed(1)
    graph = utils.simulate_dag(20, 20, "ER")
    utils.simulate_parameter(graph)
    data = utils.simulate_linear_sem(graph, 256, "gauss")
    pd.DataFrame(data).to_csv(data_out, header=False, index=False)
    np.savetxt(graph_out, graph, fmt="%d", delimiter=" ")
    outputs.extend((data_out, graph_out))

    for path in outputs:
        if path.parent.name != "ER20":
            guide = (Path(__file__).resolve().parents[1] / "docs" / "DATASETS.md").read_text(encoding="utf-8")
            match = re.search(
                rf"\| {re.escape(path.parent.name)} \| `{re.escape(path.name)}` \| `([0-9a-f]{{64}})`", guide
            )
            if not match or sha256(path) != match.group(1):
                raise ValueError(f"Prepared input does not match the study checksum: {path}")
        print(f"{path.relative_to(output_root)}  sha256={sha256(path)}")


def prepare_bn(output_root: Path) -> None:
    """Download verified BIFs, restoring historical Asia block order."""
    base = "https://www.bnlearn.com/bnrepository"
    target = output_root / "bayesian_network_structures"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("asia", "child", "alarm", "survey", "earthquake"):
        destination = target / f"{name}.bif"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing BIF: {destination}")
        url = f"{base}/{name}/{name}.bif.gz"
        with urllib.request.urlopen(url, timeout=60) as response:
            content = gzip.decompress(response.read())
        if hashlib.sha256(content).hexdigest() != BN_HASHES[name]:
            raise ValueError(f"The upstream BIF has changed; see docs/DATA_VALIDATION.md: {url}")
        if name == "asia":
            blocks = re.split(r"(?m)(?=^(?:network|variable|probability)\s)", content.decode("utf-8"))
            network = [block for block in blocks if block.startswith("network")]
            variables = sorted(block for block in blocks if block.startswith("variable"))
            probabilities = sorted(
                (block for block in blocks if block.startswith("probability")),
                key=lambda block: re.search(r"\(\s*(\w+)", block).group(1),
            )
            content = "".join(network + variables + probabilities).encode("utf-8")
        destination.write_bytes(content)
        print(f"{destination.relative_to(output_root)}  sha256={sha256(destination)}  source={url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-scd-root", "--matmcd-root", dest="matmcd_root", type=Path,
                        help="Pinned LLM-and-SCD checkout linked by MATMCD")
    parser.add_argument("--fedpc-root", type=Path)
    parser.add_argument("--notears-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--download-bnlearn", action="store_true",
                        help="Download the bnlearn BIF files used in the experiments")
    args = parser.parse_args()
    if args.download_bnlearn:
        prepare_bn(args.output_root)
    if args.matmcd_root or args.fedpc_root or args.notears_root:
        if not (args.matmcd_root and args.fedpc_root and args.notears_root):
            parser.error("continuous-input preparation requires --llm-scd-root, --fedpc-root and --notears-root")
        prepare(args.matmcd_root, args.fedpc_root, args.notears_root, args.output_root)
    elif not args.download_bnlearn:
        parser.error("specify --download-bnlearn and/or all three continuous-input source roots")
