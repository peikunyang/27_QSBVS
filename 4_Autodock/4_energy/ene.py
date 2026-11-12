#!/usr/bin/env python3
import argparse
import heapq
from pathlib import Path

import numpy as np

N = 32
EXPECTED_N = N ** 3
ELEC_TYPE = "e"
EXPECTED_DECOYS = 1023
ROTATION_STATES = 32
TRANSLATION_STATES = 512
DEFAULT_TOP = 100


def read_grid(path):
    """Read exactly 32^3 values and reshape to z,y,x."""
    values = np.fromstring(
        Path(path).read_text(encoding="utf-8", errors="ignore"),
        sep=" ",
        dtype=np.float64,
    )
    if values.size != EXPECTED_N:
        raise ValueError(
            f"{path} has {values.size} values, expected {EXPECTED_N}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite value found in {path}")
    return values.reshape(N, N, N)


def collect_map_pairs(map_root):
    """Collect <PDBID>_0 and <PDBID>_1 receptor MAP folders."""
    pairs = {}

    for folder in sorted(p for p in map_root.iterdir() if p.is_dir()):
        name = folder.name
        if name.endswith("_0"):
            pdbid = name[:-2]
            pose = 0
        elif name.endswith("_1"):
            pdbid = name[:-2]
            pose = 1
        else:
            continue

        if pose in pairs.setdefault(pdbid, {}):
            raise RuntimeError(f"duplicate MAP folder: {name}")
        pairs[pdbid][pose] = folder

    if not pairs:
        raise RuntimeError(
            f"no <PDBID>_0 / <PDBID>_1 folders found under {map_root}"
        )

    for pdbid, pair in sorted(pairs.items()):
        if set(pair) != {0, 1}:
            raise RuntimeError(
                f"MAP PDBID {pdbid} must contain both "
                f"{pdbid}_0 and {pdbid}_1"
            )

    return pairs


def load_folder(folder):
    """Load all atom-type grids from one MAP/OCC folder."""
    data = {}
    for path in sorted(p for p in folder.iterdir() if p.is_file()):
        data[path.name] = read_grid(path)

    if not data:
        raise ValueError(f"no grid files found in {folder}")

    return data


def collect_decoy_dirs(decoy_root):
    """Collect the 1023 numbered decoy OCC folders."""
    dirs = sorted(
        (p for p in decoy_root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )

    if len(dirs) != EXPECTED_DECOYS:
        raise RuntimeError(
            f"{decoy_root} has {len(dirs)} decoy folders; "
            f"expected {EXPECTED_DECOYS}"
        )

    return dirs


def build_translation_shifts():
    """
    Exact 512 translation selectors used by comb.py:
      x,y,z each range -4 ... +3
      cyclic modulo-32 indexing.
    """
    selector = np.arange(TRANSLATION_STATES, dtype=np.int64)

    x0 = (selector >> 0) & 1
    x1 = (selector >> 1) & 1
    x2 = (selector >> 2) & 1

    y0 = (selector >> 3) & 1
    y1 = (selector >> 4) & 1
    y2 = (selector >> 5) & 1

    z0 = (selector >> 6) & 1
    z1 = (selector >> 7) & 1
    z2 = (selector >> 8) & 1

    shift_x = np.where(x2 == 1, 0, -4) + 2 * x1 + x0
    shift_y = np.where(y2 == 1, 0, -4) + 2 * y1 + y0
    shift_z = np.where(z2 == 1, 0, -4) + 2 * z1 + z0

    return (
        shift_z.astype(np.int64),
        shift_y.astype(np.int64),
        shift_x.astype(np.int64),
    )


def apply_rotation_numpy(vector, rotation_selector):
    """
    Apply the same 5-bit / 32-selector OCC rotations as comb.py.
    vector shape: channels,z,y,x
    """
    out = vector

    if (rotation_selector >> 4) & 1:
        out = np.rot90(out, k=1, axes=(2, 3))

    if (rotation_selector >> 3) & 1:
        out = np.rot90(out, k=2, axes=(2, 3))

    if (rotation_selector >> 2) & 1:
        out = np.rot90(out, k=-1, axes=(1, 3))

    if (rotation_selector >> 1) & 1:
        out = np.rot90(out, k=2, axes=(1, 3))

    if rotation_selector & 1:
        out = np.rot90(out, k=1, axes=(1, 2))

    return out


def apply_rotation_torch(vector, rotation_selector, torch):
    """Torch/CUDA equivalent of apply_rotation_numpy()."""
    out = vector

    if (rotation_selector >> 4) & 1:
        out = torch.rot90(out, k=1, dims=(2, 3))

    if (rotation_selector >> 3) & 1:
        out = torch.rot90(out, k=2, dims=(2, 3))

    if (rotation_selector >> 2) & 1:
        out = torch.rot90(out, k=-1, dims=(1, 3))

    if (rotation_selector >> 1) & 1:
        out = torch.rot90(out, k=2, dims=(1, 3))

    if rotation_selector & 1:
        out = torch.rot90(out, k=1, dims=(1, 2))

    return out.contiguous()


def keep_record(heap, counter, top_k, record):
    """Keep only the globally lowest top_k total energies."""
    total = record["total"]
    item = (-total, counter, record)

    if len(heap) < top_k:
        heapq.heappush(heap, item)
    elif total < -heap[0][0]:
        heapq.heapreplace(heap, item)


def prepare_map_numpy(map_pair):
    """FFT receptor MAP _0/_1 once."""
    loaded = {
        0: load_folder(map_pair[0]),
        1: load_folder(map_pair[1]),
    }

    spectra = {}
    for pose in (0, 1):
        spectra[pose] = {
            atom_type: np.fft.rfftn(grid, axes=(0, 1, 2))
            for atom_type, grid in loaded[pose].items()
        }

    return spectra


def score_ligand_numpy(
    source,
    ligand_name,
    ligand_folder,
    map_spectra,
    shift_z,
    shift_y,
    shift_x,
    heap,
    counter,
    top_k,
):
    """Score one ligand over 32 rotations x 512 translations x 2 MAP poses."""
    occ = load_folder(ligand_folder)
    occ_types = sorted(occ)

    stack = np.stack([occ[t] for t in occ_types], axis=0)

    used_by_pose = {
        pose: [t for t in occ_types if t in map_spectra[pose]]
        for pose in (0, 1)
    }

    for pose in (0, 1):
        missing = [t for t in occ_types if t not in map_spectra[pose]]
        if missing:
            print(
                f"[warn] {ligand_name} vs pose {pose}: "
                f"missing map types: {','.join(missing)}"
            )

    for rotation in range(ROTATION_STATES):
        rotated = apply_rotation_numpy(stack, rotation)
        occ_fft = np.fft.rfftn(rotated, axes=(1, 2, 3))

        type_to_index = {t: i for i, t in enumerate(occ_types)}

        for pose in (0, 1):
            used_types = used_by_pose[pose]
            if not used_types:
                continue

            total_spec = None
            elec_spec = None

            for atom_type in used_types:
                i = type_to_index[atom_type]
                product = (
                    map_spectra[pose][atom_type]
                    * np.conj(occ_fft[i])
                )

                if total_spec is None:
                    total_spec = product.copy()
                else:
                    total_spec += product

                if atom_type == ELEC_TYPE:
                    elec_spec = product

            corr_total = np.fft.irfftn(
                total_spec,
                s=(N, N, N),
                axes=(0, 1, 2),
            ).real

            total_values = corr_total[
                shift_z,
                shift_y,
                shift_x,
            ]

            if elec_spec is None:
                elec_values = np.zeros(
                    TRANSLATION_STATES,
                    dtype=np.float64,
                )
            else:
                corr_elec = np.fft.irfftn(
                    elec_spec,
                    s=(N, N, N),
                    axes=(0, 1, 2),
                ).real
                elec_values = corr_elec[
                    shift_z,
                    shift_y,
                    shift_x,
                ]

            local_k = min(top_k, TRANSLATION_STATES)
            idxs = np.argpartition(
                total_values,
                local_k - 1,
            )[:local_k]

            for trans in idxs:
                trans = int(trans)
                total = float(total_values[trans])
                elec = float(elec_values[trans])
                vdw = total - elec

                record = {
                    "source": source,
                    "ligand": ligand_name,
                    "pose": pose,
                    "rot": rotation,
                    "trans": trans,
                    "dx": int(shift_x[trans]),
                    "dy": int(shift_y[trans]),
                    "dz": int(shift_z[trans]),
                    "elec": elec,
                    "vdw": vdw,
                    "total": total,
                    "used_types": ",".join(used_types),
                }

                keep_record(
                    heap,
                    counter[0],
                    top_k,
                    record,
                )
                counter[0] += 1


def prepare_cuda():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "CUDA mode requires PyTorch with CUDA support"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested, but torch.cuda.is_available() is False"
        )

    return torch, torch.device("cuda")


def prepare_map_cuda(map_pair, torch, device):
    """Load and FFT receptor MAP _0/_1 on GPU once."""
    spectra = {}

    for pose in (0, 1):
        data = load_folder(map_pair[pose])
        spectra[pose] = {}

        for atom_type, grid in data.items():
            tensor = torch.from_numpy(grid).to(
                device=device,
                dtype=torch.float64,
            )
            spectra[pose][atom_type] = torch.fft.rfftn(
                tensor,
                dim=(0, 1, 2),
            )

    return spectra


def score_ligand_cuda(
    source,
    ligand_name,
    ligand_folder,
    map_spectra,
    shift_z_np,
    shift_y_np,
    shift_x_np,
    shift_z,
    shift_y,
    shift_x,
    heap,
    counter,
    top_k,
    torch,
    device,
):
    """CUDA version: same 32 x 512 x 2 search."""
    occ = load_folder(ligand_folder)
    occ_types = sorted(occ)

    stack_np = np.stack([occ[t] for t in occ_types], axis=0)
    stack = torch.from_numpy(stack_np).to(
        device=device,
        dtype=torch.float64,
    )

    type_to_index = {t: i for i, t in enumerate(occ_types)}

    used_by_pose = {
        pose: [t for t in occ_types if t in map_spectra[pose]]
        for pose in (0, 1)
    }

    for pose in (0, 1):
        missing = [t for t in occ_types if t not in map_spectra[pose]]
        if missing:
            print(
                f"[warn] {ligand_name} vs pose {pose}: "
                f"missing map types: {','.join(missing)}"
            )

    with torch.no_grad():
        for rotation in range(ROTATION_STATES):
            rotated = apply_rotation_torch(
                stack,
                rotation,
                torch,
            )

            occ_fft = torch.fft.rfftn(
                rotated,
                dim=(1, 2, 3),
            )

            for pose in (0, 1):
                used_types = used_by_pose[pose]
                if not used_types:
                    continue

                total_spec = None
                elec_spec = None

                for atom_type in used_types:
                    i = type_to_index[atom_type]
                    product = (
                        map_spectra[pose][atom_type]
                        * torch.conj(occ_fft[i])
                    )

                    if total_spec is None:
                        total_spec = product.clone()
                    else:
                        total_spec = total_spec + product

                    if atom_type == ELEC_TYPE:
                        elec_spec = product

                corr_total = torch.fft.irfftn(
                    total_spec,
                    s=(N, N, N),
                    dim=(0, 1, 2),
                ).real

                total_values = corr_total[
                    shift_z,
                    shift_y,
                    shift_x,
                ]

                if elec_spec is None:
                    elec_values = torch.zeros(
                        TRANSLATION_STATES,
                        dtype=torch.float64,
                        device=device,
                    )
                else:
                    corr_elec = torch.fft.irfftn(
                        elec_spec,
                        s=(N, N, N),
                        dim=(0, 1, 2),
                    ).real
                    elec_values = corr_elec[
                        shift_z,
                        shift_y,
                        shift_x,
                    ]

                local_k = min(top_k, TRANSLATION_STATES)
                values, indices = torch.topk(
                    total_values,
                    k=local_k,
                    largest=False,
                    sorted=False,
                )

                values_np = values.detach().cpu().numpy()
                indices_np = indices.detach().cpu().numpy()
                elec_np = (
                    elec_values[indices]
                    .detach()
                    .cpu()
                    .numpy()
                )

                for j in range(local_k):
                    trans = int(indices_np[j])
                    total = float(values_np[j])
                    elec = float(elec_np[j])
                    vdw = total - elec

                    record = {
                        "source": source,
                        "ligand": ligand_name,
                        "pose": pose,
                        "rot": rotation,
                        "trans": trans,
                        "dx": int(shift_x_np[trans]),
                        "dy": int(shift_y_np[trans]),
                        "dz": int(shift_z_np[trans]),
                        "elec": elec,
                        "vdw": vdw,
                        "total": total,
                        "used_types": ",".join(used_types),
                    }

                    keep_record(
                        heap,
                        counter[0],
                        top_k,
                        record,
                    )
                    counter[0] += 1


def write_top(out_file, records):
    header = (
        f"{'rank':>5} "
        f"{'source':<7} "
        f"{'ligand':<16} "
        f"{'pose':>4} "
        f"{'rot':>4} "
        f"{'trans':>5} "
        f"{'dx':>3} "
        f"{'dy':>3} "
        f"{'dz':>3} "
        f"{'elec':>14} "
        f"{'vdw':>14} "
        f"{'total':>14}  "
        f"used_types\n"
    )

    with out_file.open("w", encoding="utf-8") as f:
        f.write(header)

        for rank, r in enumerate(records, start=1):
            f.write(
                f"{rank:5d} "
                f"{r['source']:<7} "
                f"{r['ligand']:<16} "
                f"{r['pose']:4d} "
                f"{r['rot']:4d} "
                f"{r['trans']:5d} "
                f"{r['dx']:3d} "
                f"{r['dy']:3d} "
                f"{r['dz']:3d} "
                f"{r['elec']:14.3f} "
                f"{r['vdw']:14.3f} "
                f"{r['total']:14.3f}  "
                f"{r['used_types']}\n"
            )


def process_one_receptor(
    pdbid,
    map_pair,
    occ_root,
    decoy_dirs,
    out_dir,
    top_k,
    device_mode,
):
    native_dir = occ_root / pdbid

    if not native_dir.is_dir():
        raise FileNotFoundError(
            f"same-PDBID native OCC not found: {native_dir}"
        )

    shift_z_np, shift_y_np, shift_x_np = (
        build_translation_shifts()
    )

    heap = []
    counter = [0]

    ligands = [
        ("native", pdbid, native_dir),
    ] + [
        ("decoy", d.name, d)
        for d in decoy_dirs
    ]

    if device_mode == "cuda":
        torch, device = prepare_cuda()
        print(f"[device] CUDA: {torch.cuda.get_device_name(device)}")

        map_spectra = prepare_map_cuda(
            map_pair,
            torch,
            device,
        )

        shift_z = torch.as_tensor(
            shift_z_np,
            dtype=torch.long,
            device=device,
        )
        shift_y = torch.as_tensor(
            shift_y_np,
            dtype=torch.long,
            device=device,
        )
        shift_x = torch.as_tensor(
            shift_x_np,
            dtype=torch.long,
            device=device,
        )

        for i, (source, ligand_name, folder) in enumerate(
            ligands,
            start=1,
        ):
            print(
                f"[{pdbid}] ligand {i}/{len(ligands)} "
                f"{source} {ligand_name}"
            )

            score_ligand_cuda(
                source,
                ligand_name,
                folder,
                map_spectra,
                shift_z_np,
                shift_y_np,
                shift_x_np,
                shift_z,
                shift_y,
                shift_x,
                heap,
                counter,
                top_k,
                torch,
                device,
            )

    else:
        print("[device] NumPy CPU FFT")
        map_spectra = prepare_map_numpy(map_pair)

        for i, (source, ligand_name, folder) in enumerate(
            ligands,
            start=1,
        ):
            print(
                f"[{pdbid}] ligand {i}/{len(ligands)} "
                f"{source} {ligand_name}"
            )

            score_ligand_numpy(
                source,
                ligand_name,
                folder,
                map_spectra,
                shift_z_np,
                shift_y_np,
                shift_x_np,
                heap,
                counter,
                top_k,
            )

    records = [item[2] for item in heap]
    records.sort(
        key=lambda r: (
            r["total"],
            r["source"],
            r["ligand"],
            r["pose"],
            r["rot"],
            r["trans"],
        )
    )

    out_file = out_dir / pdbid
    write_top(out_file, records)

    total_pose_scores = (
        len(ligands)
        * 2
        * ROTATION_STATES
        * TRANSLATION_STATES
    )

    print(
        f"[OK] {pdbid} -> {out_file}\n"
        f"     ligands={len(ligands)} "
        f"poses={total_pose_scores} "
        f"saved_top={len(records)}"
    )


def resolve_device(requested):
    if requested == "cpu":
        return "cpu"

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False

    if requested == "cuda":
        if not cuda_ok:
            raise RuntimeError(
                "--device cuda requested, but CUDA PyTorch is unavailable"
            )
        return "cuda"

    return "cuda" if cuda_ok else "cpu"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Search native + 1023 decoy OCC ligands against receptor "
            "MAP _0/_1 over 32 rotations x 512 translations, "
            "then save the lowest-energy poses."
        )
    )

    parser.add_argument("--map_root", required=True)
    parser.add_argument("--occ_root", required=True)
    parser.add_argument("--decoy_root", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument(
        "--pdbid",
        default=None,
        help=(
            "Process only one receptor MAP PDBID, e.g. --pdbid 1ecq. "
            "If omitted, process all MAP PDBIDs."
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help=f"number of lowest-energy poses to save (default {DEFAULT_TOP})",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto uses CUDA when available; otherwise CPU",
    )

    args = parser.parse_args()

    if args.top < 1:
        raise ValueError("--top must be >= 1")

    map_root = Path(args.map_root).resolve()
    occ_root = Path(args.occ_root).resolve()
    decoy_root = Path(args.decoy_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    for name, path in (
        ("map_root", map_root),
        ("occ_root", occ_root),
        ("decoy_root", decoy_root),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{name} not found: {path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    map_pairs = collect_map_pairs(map_root)
    decoy_dirs = collect_decoy_dirs(decoy_root)
    device_mode = resolve_device(args.device)

    if args.pdbid is not None:
        if args.pdbid not in map_pairs:
            available = " ".join(sorted(map_pairs))
            raise ValueError(
                f"MAP PDBID not found: {args.pdbid}\n"
                f"available: {available}"
            )
        selected = [args.pdbid]
    else:
        selected = sorted(map_pairs)

    print(f"MAP PDBIDs available : {len(map_pairs)}")
    print(f"MAP PDBIDs selected  : {' '.join(selected)}")
    print(f"decoys               : {len(decoy_dirs)}")
    print(f"rotations            : {ROTATION_STATES}")
    print(f"translations         : {TRANSLATION_STATES}")
    print(f"top output           : {args.top}")
    print(f"device               : {device_mode}")
    print()

    for pdbid in selected:
        process_one_receptor(
            pdbid,
            map_pairs[pdbid],
            occ_root,
            decoy_dirs,
            out_dir,
            args.top,
            device_mode,
        )

    print()
    print("Done")


if __name__ == "__main__":
    main()
