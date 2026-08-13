"""Loading recorded sessions, and splitting them without leaking.

A session directory is written by ``DatasetRecorderService`` and looks like::

    20260813_193000_neutral/
        session.json
        frames/000000.jpg ...
        labels.jsonl

The single most important thing in this module is :func:`split_sessions`. Adjacent video frames are
near-duplicates, so a random per-frame split puts near-copies of training frames into validation and
reports a validation score that has nothing to do with generalization. Splitting happens at session
granularity, always.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

from . import schema


@dataclass(frozen=True)
class FrameRecord:
    """One recorded frame: the model's input, its raw output, and any commanded target."""

    index: int
    timestamp_ticks: int
    stock: np.ndarray  # float32 [45]
    image_path: Path
    cue: dict | None = None

    def load_image(self) -> np.ndarray:
        """Grayscale float32 [1, H, W] scaled to [0,1] - the same normalization the stock model uses."""
        raw = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"Could not read frame {self.image_path}")
        return (raw.astype(np.float32) / 255.0)[None, :, :]


@dataclass
class Session:
    """A recorded session plus its metadata."""

    session_id: str
    path: Path
    metadata: dict
    frames: list[FrameRecord] = field(default_factory=list)

    @property
    def session_type(self) -> str:
        return str(self.metadata.get("SessionType", "Unknown"))

    @property
    def is_neutral(self) -> bool:
        return self.session_type.lower() == "neutral"

    @property
    def is_guided(self) -> bool:
        return self.session_type.lower() == "guided"

    @property
    def is_speech(self) -> bool:
        return self.session_type.lower() == "speech"

    def __len__(self) -> int:
        return len(self.frames)


def load_session(path: Path, *, verify_schema: bool = True) -> Session:
    """Read one session directory. Frames whose image is missing are skipped with a warning."""
    metadata_path = path / "session.json"
    labels_path = path / "labels.jsonl"

    if not metadata_path.exists():
        raise FileNotFoundError(f"{path} has no session.json")
    if not labels_path.exists():
        raise FileNotFoundError(f"{path} has no labels.jsonl")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if verify_schema:
        recorded = metadata.get("ExpressionSchemaSha256")
        if recorded:
            schema.assert_compatible(recorded, source=str(path.name))

    frames_dir = path / "frames"
    frames: list[FrameRecord] = []
    missing = 0

    for line in labels_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        record = json.loads(line)
        index = int(record["i"])
        image_path = frames_dir / f"{index:06d}.jpg"

        # A truncated session (crash mid-write) can leave a label with no image. Dropping the label
        # is correct: keeping it would silently pair a target with the wrong picture.
        if not image_path.exists():
            missing += 1
            continue

        stock = np.asarray(record["stock"], dtype=np.float32)
        if stock.shape != (schema.EXPRESSION_COUNT,):
            raise ValueError(
                f"{path.name} frame {index}: expected {schema.EXPRESSION_COUNT} stock values, "
                f"got {stock.shape}"
            )

        frames.append(
            FrameRecord(
                index=index,
                timestamp_ticks=int(record.get("t", 0)),
                stock=stock,
                image_path=image_path,
                cue=record.get("cue"),
            )
        )

    if missing:
        print(f"  warning: {path.name} - {missing} labels had no matching image and were skipped")

    return Session(session_id=path.name, path=path, metadata=metadata, frames=frames)


def discover_sessions(root: Path, *, verify_schema: bool = True) -> list[Session]:
    """Load every session directory under ``root``, sorted by name (chronological by construction)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    sessions = [
        load_session(child, verify_schema=verify_schema)
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / "session.json").exists()
    ]

    if not sessions:
        raise FileNotFoundError(
            f"No sessions found under {root}. Record one from the Personalization page first."
        )

    return sessions


def split_sessions(
    sessions: Sequence[Session],
    val_session_ids: Sequence[str] | None = None,
) -> tuple[list[Session], list[Session]]:
    """Split into (train, val) **by session**, never by frame.

    With no explicit ``val_session_ids``, one session of each available type is held out (the last
    of each, chronologically), so validation covers neutral, guided and speech behavior rather than
    whichever type happens to dominate.
    """
    if val_session_ids:
        wanted = set(val_session_ids)
        val = [s for s in sessions if s.session_id in wanted]

        unknown = wanted - {s.session_id for s in val}
        if unknown:
            raise ValueError(f"Unknown validation session ids: {sorted(unknown)}")

        train = [s for s in sessions if s.session_id not in wanted]
    else:
        val_ids: set[str] = set()
        for session_type in {s.session_type for s in sessions}:
            of_type = [s for s in sessions if s.session_type == session_type]
            if len(of_type) > 1:  # never hold out the only session of a type
                val_ids.add(of_type[-1].session_id)

        val = [s for s in sessions if s.session_id in val_ids]
        train = [s for s in sessions if s.session_id not in val_ids]

    if not train:
        raise ValueError("Every session was assigned to validation; nothing left to train on.")

    return train, val


def iter_frames(sessions: Sequence[Session]) -> Iterator[tuple[Session, FrameRecord]]:
    for session in sessions:
        for frame in session.frames:
            yield session, frame


def stack_stock(sessions: Sequence[Session]) -> np.ndarray:
    """All stock vectors as float32 [N, 45]."""
    rows = [frame.stock for _, frame in iter_frames(sessions)]
    if not rows:
        return np.zeros((0, schema.EXPRESSION_COUNT), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


def load_images(sessions: Sequence[Session]) -> np.ndarray:
    """All frames as float32 [N, 1, H, W]. Small personal datasets fit in RAM comfortably.

    At 224x224 that is ~200 KB per frame, so ~10k frames is ~2 GB; use ``--model a`` (which ignores
    images) or downscale first if a corpus ever outgrows memory.
    """
    images = [frame.load_image() for _, frame in iter_frames(sessions)]
    if not images:
        return np.zeros((0, 1, schema.IMAGE_SIZE, schema.IMAGE_SIZE), dtype=np.float32)
    return np.stack(images).astype(np.float32)


def describe(sessions: Sequence[Session]) -> str:
    """One-line-per-session summary, printed by train/evaluate so runs are self-documenting."""
    lines = []
    for s in sessions:
        fps = s.metadata.get("EffectiveFps")
        fps_text = f", {fps:.1f} fps" if isinstance(fps, (int, float)) else ""
        lines.append(f"  {s.session_id:<32} {s.session_type:<8} {len(s):>6} frames{fps_text}")
    total = sum(len(s) for s in sessions)
    lines.append(f"  {'TOTAL':<32} {'':<8} {total:>6} frames")
    return "\n".join(lines)
