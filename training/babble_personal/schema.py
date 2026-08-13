"""Canonical face-expression schema, mirroring the C# side.

The stock faceModel.onnx emits a positional float[45] and carries no output names, so index meaning
is defined purely by convention. This module is the Python half of that contract; the C# half is
``src/Baballonia/Services/Personalization/PersonalizationSchema.cs``.

The hash recipe must match C# byte for byte: names joined by a single ``\n``, UTF-8 encoded, no
trailing newline and no BOM, lowercase hex SHA-256. Exported personal models embed the result, and
the runtime refuses to load a model whose hash disagrees, which is what prevents a reordered schema
from silently repointing every learned correction.
"""

from __future__ import annotations

import hashlib
from typing import Final, Sequence

SCHEMA_VERSION: Final[int] = 1

EXPRESSION_NAMES: Final[tuple[str, ...]] = (
    "CheekPuffLeft",
    "CheekPuffRight",
    "CheekSuckLeft",
    "CheekSuckRight",
    "JawOpen",
    "JawForward",
    "JawLeft",
    "JawRight",
    "NoseSneerLeft",
    "NoseSneerRight",
    "MouthFunnel",
    "MouthPucker",
    "MouthLeft",
    "MouthRight",
    "MouthRollUpper",
    "MouthRollLower",
    "MouthShrugUpper",
    "MouthShrugLower",
    "MouthClose",
    "MouthSmileLeft",
    "MouthSmileRight",
    "MouthFrownLeft",
    "MouthFrownRight",
    "MouthDimpleLeft",
    "MouthDimpleRight",
    "MouthUpperUpLeft",
    "MouthUpperUpRight",
    "MouthLowerDownLeft",
    "MouthLowerDownRight",
    "MouthPressLeft",
    "MouthPressRight",
    "MouthStretchLeft",
    "MouthStretchRight",
    "TongueOut",
    "TongueUp",
    "TongueDown",
    "TongueLeft",
    "TongueRight",
    "TongueRoll",
    "TongueBendDown",
    "TongueCurlUp",
    "TongueSquish",
    "TongueFlat",
    "TongueTwistLeft",
    "TongueTwistRight",
)

EXPRESSION_COUNT: Final[int] = len(EXPRESSION_NAMES)

INDEX_OF: Final[dict[str, int]] = {name: i for i, name in enumerate(EXPRESSION_NAMES)}

#: Frames are recorded at the stock model's native input size.
IMAGE_SIZE: Final[int] = 224

#: The stock model divides by 255 and does nothing else - no mean/std normalization.
INPUT_NORMALIZATION: Final[str] = "gray_div255"


def schema_sha256(names: Sequence[str] = EXPRESSION_NAMES) -> str:
    """Canonical hash of an expression ordering. Must match ``PersonalizationSchema.Sha256``."""
    canonical = "\n".join(names)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


SCHEMA_SHA256: Final[str] = schema_sha256()


def assert_compatible(recorded_hash: str, *, source: str = "dataset") -> None:
    """Fail loudly when data or a model was produced against a different expression ordering.

    Silently training across a schema change would produce a model that maps corrections onto the
    wrong expressions, which is far worse than refusing to run.
    """
    if recorded_hash != SCHEMA_SHA256:
        raise ValueError(
            f"Expression schema mismatch in {source}.\n"
            f"  expected {SCHEMA_SHA256}\n"
            f"  found    {recorded_hash}\n"
            "The 45-value output order changed. Data recorded under a different ordering cannot be "
            "mixed with this one; re-record it or migrate it explicitly."
        )


# Groups used for label masking and reporting. Indices, not names, so they stay cheap at train time.
JAW_DIMS: Final[tuple[int, ...]] = tuple(INDEX_OF[n] for n in EXPRESSION_NAMES if n.startswith("Jaw"))
MOUTH_DIMS: Final[tuple[int, ...]] = tuple(INDEX_OF[n] for n in EXPRESSION_NAMES if n.startswith("Mouth"))
TONGUE_DIMS: Final[tuple[int, ...]] = tuple(INDEX_OF[n] for n in EXPRESSION_NAMES if n.startswith("Tongue"))
CHEEK_DIMS: Final[tuple[int, ...]] = tuple(INDEX_OF[n] for n in EXPRESSION_NAMES if n.startswith("Cheek"))
NOSE_DIMS: Final[tuple[int, ...]] = tuple(INDEX_OF[n] for n in EXPRESSION_NAMES if n.startswith("Nose"))

#: Dimensions that natural speech can plausibly supervise as weak pseudo-labels.
SPEECH_SUPERVISED_DIMS: Final[tuple[int, ...]] = tuple(sorted(set(JAW_DIMS) | set(MOUTH_DIMS)))
