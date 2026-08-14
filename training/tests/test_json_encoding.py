"""Regression tests for the UTF-8 BOM that broke the first real training run.

The C# recorder built its labels writer with ``Encoding.UTF8``, whose *preamble is the BOM*, so
every ``labels.jsonl`` produced before the fix starts with ``EF BB BF`` and ``json.loads`` rejected
the first line with "Unexpected UTF-8 BOM". ``session.json`` went through ``File.WriteAllText``,
which is BOM-less, and was fine - but both are now read with ``utf-8-sig`` so neither file can
reintroduce the failure.

These tests cover both files with and without a BOM, and - when the machine has real recordings -
assert that the actual on-disk corpus loads. Existing recordings are only ever read, never written.

    python -m pytest training/tests/test_json_encoding.py -q
    python training/tests/test_json_encoding.py            (no pytest required)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from babble_personal import dataset as ds  # noqa: E402
from babble_personal import schema  # noqa: E402

BOM = "﻿"
BOM_BYTES = b"\xef\xbb\xbf"
N = schema.EXPRESSION_COUNT


def _write_session(
    root: Path,
    session_id: str,
    *,
    frame_count: int = 3,
    metadata_bom: bool = False,
    labels_bom: bool = False,
) -> Path:
    """A minimal session in the recorder's layout, optionally BOM-prefixed like an old recording."""
    path = root / session_id
    (path / "frames").mkdir(parents=True, exist_ok=True)

    metadata = {
        "SchemaVersion": 1,
        "SessionId": session_id,
        "SessionType": "Neutral",
        "StartedUtc": "2026-08-13T00:00:00Z",
        "AppVersion": "test",
        "ExpressionNames": list(schema.EXPRESSION_NAMES),
        "ExpressionSchemaSha256": schema.SCHEMA_SHA256,
        "ImageWidth": schema.IMAGE_SIZE,
        "ImageHeight": schema.IMAGE_SIZE,
        "JpegQuality": 95,
        "FrameCount": frame_count,
        "EffectiveFps": 30.0,
    }

    lines = []
    for index in range(frame_count):
        image = np.full((schema.IMAGE_SIZE, schema.IMAGE_SIZE), (index * 11) % 200, dtype=np.uint8)
        cv2.imwrite(str(path / "frames" / f"{index:06d}.jpg"), image)
        lines.append(json.dumps({"i": index, "t": index * 333333, "stock": [0.0] * N}))

    # Written as bytes so the BOM is exactly the three bytes the recorder emitted, rather than
    # whatever an encoder might decide to do with it.
    metadata_text = (BOM if metadata_bom else "") + json.dumps(metadata, indent=2)
    labels_text = (BOM if labels_bom else "") + "\n".join(lines) + "\n"
    (path / "session.json").write_bytes(metadata_text.encode("utf-8"))
    (path / "labels.jsonl").write_bytes(labels_text.encode("utf-8"))

    return path


def _assert_loads(path: Path, expected_frames: int) -> None:
    session = ds.load_session(path)
    assert len(session) == expected_frames, f"{path.name}: expected {expected_frames}, got {len(session)}"
    assert session.session_type == "Neutral", f"{path.name}: metadata did not parse"
    assert session.frames[0].stock.shape == (N,)


def test_plain_session_json_loads() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_session(Path(tmp), "20260813_000000_neutral")
        assert (path / "session.json").read_bytes()[:3] != BOM_BYTES
        _assert_loads(path, 3)


def test_bom_prefixed_session_json_loads() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_session(Path(tmp), "20260813_000001_neutral", metadata_bom=True)
        assert (path / "session.json").read_bytes()[:3] == BOM_BYTES, "test fixture lost its BOM"
        _assert_loads(path, 3)


def test_plain_labels_jsonl_loads() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_session(Path(tmp), "20260813_000002_neutral")
        assert (path / "labels.jsonl").read_bytes()[:3] != BOM_BYTES
        _assert_loads(path, 3)


def test_bom_prefixed_labels_jsonl_loads() -> None:
    """The exact failure from the first real run: a BOM at byte 0 of labels.jsonl."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_session(Path(tmp), "20260813_000003_neutral", labels_bom=True)
        assert (path / "labels.jsonl").read_bytes()[:3] == BOM_BYTES, "test fixture lost its BOM"
        _assert_loads(path, 3)


def test_both_files_bom_prefixed_load() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_session(
            Path(tmp), "20260813_000004_neutral", metadata_bom=True, labels_bom=True
        )
        _assert_loads(path, 3)


def test_discovery_over_mixed_bom_and_plain_sessions() -> None:
    """Discovery must not care which recorder version wrote which session."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_session(root, "20260813_000010_neutral", labels_bom=True)   # old recorder
        _write_session(root, "20260813_000011_neutral")                    # fixed recorder

        sessions = ds.discover_sessions(root)
        assert len(sessions) == 2, f"expected 2 sessions, got {len(sessions)}"
        assert all(len(s) == 3 for s in sessions)


def test_mid_file_bom_is_still_rejected() -> None:
    """utf-8-sig must not become a licence to swallow corruption anywhere but byte 0."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_session(Path(tmp), "20260813_000020_neutral")
        labels = path / "labels.jsonl"
        lines = labels.read_text(encoding="utf-8").splitlines()
        lines[1] = BOM + lines[1]
        labels.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))

        try:
            ds.load_session(path)
        except json.JSONDecodeError:
            return
        raise AssertionError("a BOM in the middle of the file should still fail loudly")


def _real_dataset_root() -> Path | None:
    """The recordings on this machine, if any. Read-only."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    root = Path(appdata) / "ProjectBabble" / "PersonalDataset"
    return root if root.is_dir() and any(root.iterdir()) else None


def test_existing_recordings_load() -> None:
    """Discovery over the real corpus - the case that actually failed. Skipped when absent."""
    root = _real_dataset_root()
    if root is None:
        print("  (skipped: no recordings on this machine)")
        return

    sessions = ds.discover_sessions(root)
    assert sessions, f"no sessions discovered under {root}"

    bom_sessions = [
        s for s in sessions if (s.path / "labels.jsonl").read_bytes()[:3] == BOM_BYTES
    ]
    assert bom_sessions, (
        "expected at least one pre-fix BOM recording here; if every recording on this machine was "
        "made by the fixed recorder the fabricated tests above still cover the case"
    )

    for session in sessions:
        assert len(session) > 0, f"{session.session_id} loaded zero frames"
        assert session.frames[0].stock.shape == (N,)

    print(f"  loaded {len(sessions)} real sessions ({len(bom_sessions)} BOM-prefixed), "
          f"{sum(len(s) for s in sessions)} frames")


def main() -> int:
    tests = [
        test_plain_session_json_loads,
        test_bom_prefixed_session_json_loads,
        test_plain_labels_jsonl_loads,
        test_bom_prefixed_labels_jsonl_loads,
        test_both_files_bom_prefixed_load,
        test_discovery_over_mixed_bom_and_plain_sessions,
        test_mid_file_bom_is_still_rejected,
        test_existing_recordings_load,
    ]

    failures = 0
    for test in tests:
        print(f"\n--- {test.__name__}")
        try:
            test()
            print("  PASS")
        except Exception as exc:  # noqa: BLE001 - standalone runner reports rather than raises
            failures += 1
            print(f"  FAIL: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
