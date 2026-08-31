"""Backfill the stock model's visual embedding for already-recorded sessions.

    python -m babble_personal.compute_embeddings --data "%APPDATA%\\ProjectBabble\\PersonalDataset"

Model C trains on the 1280-d embedding instead of raw pixels, so every frame needs one. Sessions
recorded before the runtime learned to capture them (which is all of them, today) get theirs
computed here from the stored JPEGs.

Format is a flat float16 binary sidecar, not JSON. At 1280 values per frame, JSON would add roughly
25 KB per line to labels.jsonl and make a file that is currently readable in a text editor useless
for debugging. float16 halves the size again and costs nothing that matters: the embedding feeds a
LayerNorm, and its useful precision is nowhere near 11 bits of mantissa.

One honest caveat, recorded in the sidecar as ``source``. Backfilled embeddings are computed from
the *decoded JPEG*, while the stock 45-vector stored alongside them came from the pre-JPEG tensor.
The q95 difference is small but it is not zero. Once the runtime captures embeddings live, the same
files are written with ``source: "live"`` and the skew disappears; ``--verify-against-live`` measures
it directly on any session that has both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import dataset as ds
from .derive_embedding import EMBEDDING_DIM, EMBEDDING_OUTPUT_NAME, file_md5

EMBEDDINGS_FILE = "embeddings.bin"
EMBEDDINGS_META = "embeddings.json"

#: float16 on disk. See the module docstring for why this is not a precision problem.
STORAGE_DTYPE = np.float16


def embeddings_path(session_path: Path) -> Path:
    return Path(session_path) / EMBEDDINGS_FILE


def metadata_path(session_path: Path) -> Path:
    return Path(session_path) / EMBEDDINGS_META


def load_embeddings(session_path: Path, expected_frames: int | None = None) -> np.ndarray | None:
    """Read a session's embeddings as float32 ``[N, 1280]``, or None when absent or mismatched.

    A count mismatch returns None rather than raising: the usual cause is a session that grew after
    its embeddings were computed, and the right answer is to recompute, not to crash a training run
    that could still proceed with another model.
    """
    session_path = Path(session_path)
    binary, meta = embeddings_path(session_path), metadata_path(session_path)

    if not (binary.exists() and meta.exists()):
        return None

    try:
        record = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    dim = int(record.get("dim", EMBEDDING_DIM))
    dtype = np.dtype(record.get("dtype", "float16"))

    values = np.fromfile(binary, dtype=dtype)
    if values.size % dim != 0:
        return None

    values = values.reshape(-1, dim).astype(np.float32)

    if expected_frames is not None and len(values) != expected_frames:
        print(f"  warning: {session_path.name} has {len(values)} embeddings for "
              f"{expected_frames} frames - recompute them")
        return None

    return values


def write_embeddings(session_path: Path, values: np.ndarray, *, source: str,
                     model_md5: str | None = None) -> None:
    session_path = Path(session_path)

    values.astype(STORAGE_DTYPE).tofile(embeddings_path(session_path))
    metadata_path(session_path).write_text(json.dumps({
        "dim": int(values.shape[1]),
        "dtype": np.dtype(STORAGE_DTYPE).name,
        "count": int(values.shape[0]),
        "source": source,
        "model_md5": model_md5,
    }, indent=2), encoding="utf-8")


def compute_for_session(session, model_path: Path, *, batch_size: int = 32) -> np.ndarray:
    """Run the derived model over a session's frames and return ``[N, 1280]`` float32."""
    import onnxruntime as ort

    session_ort = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session_ort.get_inputs()[0].name

    output_names = [o.name for o in session_ort.get_outputs()]
    embedding_index = next(
        (i for i, name in enumerate(output_names)
         if name == EMBEDDING_OUTPUT_NAME or name.endswith("Flatten_output_0")),
        None)

    if embedding_index is None:
        raise ValueError(
            f"{model_path} does not expose an embedding output (has {output_names}).\n"
            "Build it with: python -m babble_personal.derive_embedding --stock <faceModel.onnx>")

    rows: list[np.ndarray] = []
    for start in range(0, len(session.frames), batch_size):
        chunk = session.frames[start:start + batch_size]
        for frame in chunk:
            image = frame.load_image()[None, ...]  # [1,1,H,W]
            outputs = session_ort.run(None, {input_name: image.astype(np.float32)})
            rows.append(np.asarray(outputs[embedding_index], dtype=np.float32).reshape(-1))

    if not rows:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    return np.stack(rows)


def verify_against_live(session, model_path: Path) -> dict | None:
    """Compare freshly computed embeddings against ones the runtime captured live.

    Quantifies the JPEG round-trip skew described in the module docstring, on real data, rather than
    leaving it as an assumption. Returns None when the session has no live embeddings to compare to.
    """
    existing_meta = metadata_path(session.path)
    if not existing_meta.exists():
        return None

    try:
        record = json.loads(existing_meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    if record.get("source") != "live":
        return None

    live = load_embeddings(session.path, expected_frames=len(session.frames))
    if live is None:
        return None

    backfilled = compute_for_session(session, model_path)
    if len(backfilled) != len(live):
        return None

    difference = np.abs(backfilled - live)
    scale = float(np.abs(live).mean()) or 1.0

    return {
        "session": session.session_id,
        "frames": int(len(live)),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "relative_mean": float(difference.mean() / scale),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute the stock visual embedding for recorded sessions.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset root")
    parser.add_argument("--model", type=Path, required=True,
                        help="Derived model with an embedding output")
    parser.add_argument("--sessions", nargs="*", default=None, help="Session ids (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute sessions that already have embeddings")
    parser.add_argument("--verify-against-live", action="store_true",
                        help="Measure backfill-vs-live skew instead of writing anything")
    args = parser.parse_args(argv)

    sessions = ds.discover_sessions(args.data)
    if args.sessions:
        wanted = set(args.sessions)
        sessions = [s for s in sessions if s.session_id in wanted]

    if not sessions:
        raise SystemExit("No sessions selected.")

    model_md5 = file_md5(args.model)

    if args.verify_against_live:
        results = [r for s in sessions if (r := verify_against_live(s, args.model))]
        if not results:
            print("No sessions have live-captured embeddings to compare against yet.")
            return 0

        print(f"{'session':<34}{'frames':>8}{'mean |d|':>12}{'max |d|':>12}{'relative':>11}")
        for result in results:
            print(f"{result['session']:<34}{result['frames']:>8}"
                  f"{result['mean_abs_diff']:>12.5f}{result['max_abs_diff']:>12.5f}"
                  f"{result['relative_mean']:>11.4f}")
        return 0

    total = 0
    for session in sessions:
        if not session.frames:
            print(f"  {session.session_id}: no frames, skipped")
            continue

        if not args.force and load_embeddings(session.path, len(session.frames)) is not None:
            print(f"  {session.session_id}: already has embeddings, skipped")
            continue

        values = compute_for_session(session, args.model)
        write_embeddings(session.path, values, source="backfill", model_md5=model_md5)

        megabytes = values.size * np.dtype(STORAGE_DTYPE).itemsize / (1024 * 1024)
        print(f"  {session.session_id}: {len(values)} embeddings ({megabytes:.1f} MB)")
        total += len(values)

    print(f"\n{total} embeddings written.")
    print("Train model C with: python -m babble_personal.train --data <root> --model c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
