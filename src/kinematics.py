"""Skeleton model shared by the motion library, blending and the Blender rig.

COCO-17 joints are expanded into a 20-node skeleton with a proper tree so that
motion can be represented as *bone directions + bone lengths*.  Direction-space
blending lets the pipeline interpolate poses with spherical interpolation
(Slerp) instead of averaging joint positions, which avoids the shrinking and
foot sliding of naive position cross-fades.

Bone directions come straight from the dataset (no IK), so the mapping from
music to motion never introduces hidden pose reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# COCO-17 joint order used by AIST++
# ---------------------------------------------------------------------------
COCO_JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Node order of the expanded skeleton.  Index 0 must be the root.
NODE_NAMES = [
    "hips", "spine_mid", "chest", "neck", "head", "head_top",
    "shoulder.L", "elbow.L", "wrist.L", "hand.L",
    "shoulder.R", "elbow.R", "wrist.R", "hand.R",
    "hip.L", "knee.L", "ankle.L", "toe.L",
    "hip.R", "knee.R", "ankle.R", "toe.R",
]
NODE_INDEX = {name: index for index, name in enumerate(NODE_NAMES)}

# (bone name, parent node, child node)
BONES: list[tuple[str, str, str]] = [
    ("spine", "hips", "spine_mid"),
    ("spine_up", "spine_mid", "chest"),
    ("neck", "chest", "neck"),
    ("head", "neck", "head"),
    ("head_top", "head", "head_top"),

    ("clavicle.L", "chest", "shoulder.L"),
    ("upper_arm.L", "shoulder.L", "elbow.L"),
    ("forearm.L", "elbow.L", "wrist.L"),
    ("hand.L", "wrist.L", "hand.L"),

    ("clavicle.R", "chest", "shoulder.R"),
    ("upper_arm.R", "shoulder.R", "elbow.R"),
    ("forearm.R", "elbow.R", "wrist.R"),
    ("hand.R", "wrist.R", "hand.R"),

    ("pelvis.L", "hips", "hip.L"),
    ("thigh.L", "hip.L", "knee.L"),
    ("shin.L", "knee.L", "ankle.L"),
    ("foot.L", "ankle.L", "toe.L"),

    ("pelvis.R", "hips", "hip.R"),
    ("thigh.R", "hip.R", "knee.R"),
    ("shin.R", "knee.R", "ankle.R"),
    ("foot.R", "ankle.R", "toe.R"),
]

BONE_NAMES = [bone[0] for bone in BONES]
BONE_INDEX = {name: index for index, name in enumerate(BONE_NAMES)}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def normalize(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(lengths, eps)


def slerp_direction(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation between two unit direction vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))

    if dot > 0.9995:
        mixed = (1.0 - t) * a + t * b
        return normalize(mixed)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    if abs(sin_theta) < 1e-8:
        return a.copy()

    return normalize(
        (np.sin((1.0 - t) * theta) / sin_theta) * a
        + (np.sin(t * theta) / sin_theta) * b
    )


def rotate_axis_angle(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of ``vector`` around a unit ``axis``."""
    axis = normalize(np.asarray(axis, dtype=float))
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    cross = np.cross(axis, vector)
    dot = np.dot(axis, vector)
    return vector * cos_a + cross * sin_a + axis * dot * (1.0 - cos_a)


def scale_direction(
    direction: np.ndarray,
    rest: np.ndarray,
    gain: float,
) -> np.ndarray:
    """Scale the angle between rest and current direction (motion gain).

    ``gain = 1`` keeps the pose, ``gain = 0`` returns the rest pose, and
    ``gain > 1`` exaggerates the movement.  This is how musical energy is
    mapped to "how strong the movement is".  Works for a single direction
    ``(3,)`` or a batch ``(F, 3)``.
    """
    direction = np.asarray(direction, dtype=float)
    rest = np.asarray(rest, dtype=float)
    if abs(gain - 1.0) < 1e-6:
        return direction

    single = direction.ndim == 1
    batch = direction.reshape(-1, 3)
    rest_batch = np.broadcast_to(rest, batch.shape)

    dot = np.clip(np.sum(rest_batch * batch, axis=-1, keepdims=True), -1.0, 1.0)
    angle = np.arccos(dot)

    axis = np.cross(rest_batch, batch)
    axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)

    # Degenerate cases: identical direction (no rotation) and exact flip.
    axis_safe = axis / np.maximum(axis_norm, 1e-9)
    parallel = axis_norm < 1e-8
    if np.any(parallel):
        fallback = np.cross(rest_batch, np.array([0.0, 0.0, 1.0]))
        weak = np.linalg.norm(fallback, axis=-1, keepdims=True) < 1e-8
        fallback = np.where(weak, np.cross(rest_batch, np.array([1.0, 0.0, 0.0])), fallback)
        fallback = fallback / np.maximum(np.linalg.norm(fallback, axis=-1, keepdims=True), 1e-9)
        axis_safe = np.where(parallel, fallback, axis_safe)

    new_angle = angle * max(gain, 0.0)
    cos_a = np.cos(new_angle)
    sin_a = np.sin(new_angle)
    rotated = (
        rest_batch * cos_a
        + np.cross(axis_safe, rest_batch) * sin_a
        + axis_safe * np.sum(axis_safe * rest_batch, axis=-1, keepdims=True) * (1.0 - cos_a)
    )
    rotated = normalize(rotated)
    rotated = np.where(angle < 1e-6, batch, rotated)

    if single:
        return rotated[0]
    return rotated.reshape(direction.shape)


# ---------------------------------------------------------------------------
# COCO-17 -> expanded skeleton
# ---------------------------------------------------------------------------

def nodes_from_joints(joints: np.ndarray) -> np.ndarray:
    """Expand COCO-17 joints into the 21-node skeleton (root first).

    ``joints`` is ``(..., 17, 3)``; the result is ``(..., 21, 3)``.
    """
    joints = np.asarray(joints, dtype=float)
    leading = joints.shape[:-2]
    j = joints.reshape(-1, 17, 3)

    hips = 0.5 * (j[:, 11] + j[:, 12])
    chest = 0.5 * (j[:, 5] + j[:, 6])
    head = j[:, [0, 1, 2, 3, 4]].mean(axis=1)

    up = normalize(head - chest)
    head_top = head + up * 0.20

    lateral = normalize(j[:, 12] - j[:, 11])
    forward = normalize(np.cross(lateral, np.array([0.0, 0.0, 1.0])))
    # Keep the forward vector horizontal.
    forward[:, 2] = 0.0
    forward = normalize(forward)

    wrist_l, elbow_l = j[:, 9], j[:, 7]
    wrist_r, elbow_r = j[:, 10], j[:, 8]
    hand_l = wrist_l + normalize(wrist_l - elbow_l) * 0.09
    hand_r = wrist_r + normalize(wrist_r - elbow_r) * 0.09

    ankle_l, ankle_r = j[:, 15], j[:, 16]
    toe_l = ankle_l + forward * 0.13 - np.array([0.0, 0.0, 0.04])
    toe_r = ankle_r + forward * 0.13 - np.array([0.0, 0.0, 0.04])

    spine_mid = 0.5 * (hips + chest)
    neck = 0.5 * (chest + head)

    nodes = np.stack([
        hips, spine_mid, chest, neck, head, head_top,
        j[:, 5], elbow_l, wrist_l, hand_l,
        j[:, 6], elbow_r, wrist_r, hand_r,
        j[:, 11], j[:, 13], ankle_l, toe_l,
        j[:, 12], j[:, 14], ankle_r, toe_r,
    ], axis=1)

    return nodes.reshape(*leading, len(NODE_NAMES), 3)


def joints_from_nodes(nodes: np.ndarray) -> np.ndarray:
    """Project the expanded skeleton back to COCO-17 positions."""
    nodes = np.asarray(nodes, dtype=float)
    leading = nodes.shape[:-2]
    n = nodes.reshape(-1, len(NODE_NAMES), 3)
    j = np.empty((n.shape[0], 17, 3), dtype=float)
    j[:, 0] = n[:, NODE_INDEX["head"]]
    j[:, 1] = j[:, 0]
    j[:, 2] = j[:, 0]
    j[:, 3] = j[:, 0]
    j[:, 4] = j[:, 0]
    j[:, 5] = n[:, NODE_INDEX["shoulder.L"]]
    j[:, 6] = n[:, NODE_INDEX["shoulder.R"]]
    j[:, 7] = n[:, NODE_INDEX["elbow.L"]]
    j[:, 8] = n[:, NODE_INDEX["elbow.R"]]
    j[:, 9] = n[:, NODE_INDEX["wrist.L"]]
    j[:, 10] = n[:, NODE_INDEX["wrist.R"]]
    j[:, 11] = n[:, NODE_INDEX["hip.L"]]
    j[:, 12] = n[:, NODE_INDEX["hip.R"]]
    j[:, 13] = n[:, NODE_INDEX["knee.L"]]
    j[:, 14] = n[:, NODE_INDEX["knee.R"]]
    j[:, 15] = n[:, NODE_INDEX["ankle.L"]]
    j[:, 16] = n[:, NODE_INDEX["ankle.R"]]
    return j.reshape(*leading, 17, 3)


# ---------------------------------------------------------------------------
# Skeleton description
# ---------------------------------------------------------------------------

@dataclass
class Skeleton:
    """Rest pose, bone graph and per-bone rest geometry."""

    rest_nodes: np.ndarray                    # (21, 3)
    rest_directions: np.ndarray = field(init=False)   # (21, 3)
    rest_lengths: np.ndarray = field(init=False)      # (21,)
    parents: np.ndarray = field(init=False)           # (21,)
    bone_order: np.ndarray = field(init=False)        # (21,) evaluation order

    def __post_init__(self) -> None:
        rest = np.asarray(self.rest_nodes, dtype=float)
        self.rest_nodes = rest
        self.parents = np.zeros(len(NODE_NAMES), dtype=int)
        for _, parent, child in BONES:
            self.parents[NODE_INDEX[child]] = NODE_INDEX[parent]
        self.parents[0] = -1

        # Node depth via relaxation over the bone list.
        depth = np.full(len(NODE_NAMES), -1, dtype=int)
        depth[0] = 0
        for _ in range(len(NODE_NAMES)):
            for _, parent, child in BONES:
                parent_index = NODE_INDEX[parent]
                if depth[parent_index] >= 0:
                    depth[NODE_INDEX[child]] = depth[parent_index] + 1
        depth[depth < 0] = 0

        self.bone_order = np.argsort(
            [depth[NODE_INDEX[parent]] for _, parent, _ in BONES],
            kind="stable",
        )

        self.rest_directions = np.stack([
            normalize(rest[NODE_INDEX[child]] - rest[NODE_INDEX[parent]])
            for _, parent, child in BONES
        ])
        self.rest_lengths = np.array([
            np.linalg.norm(rest[NODE_INDEX[child]] - rest[NODE_INDEX[parent]])
            for _, parent, child in BONES
        ], dtype=float)

    def directions_from_nodes(self, nodes: np.ndarray) -> np.ndarray:
        """Bone directions for a sequence of node positions ``(F, 21, 3)``."""
        nodes = np.asarray(nodes, dtype=float)
        directions = np.stack([
            normalize(nodes[:, NODE_INDEX[child]] - nodes[:, NODE_INDEX[parent]])
            for _, parent, child in BONES
        ], axis=1)
        return directions

    def fk(
        self,
        root: np.ndarray,
        directions: np.ndarray,
        lengths: np.ndarray | None = None,
    ) -> np.ndarray:
        """Forward kinematics: root + bone directions -> node positions.

        ``root`` is ``(3,)`` or ``(F, 3)``; ``directions`` is ``(F, 21, 3)``;
        ``lengths`` optionally overrides the rest bone lengths.
        """
        directions = np.asarray(directions, dtype=float)
        frames = directions.shape[0]
        root = np.asarray(root, dtype=float)
        if root.ndim == 1:
            root = np.repeat(root[None, :], frames, axis=0)

        if lengths is None:
            lengths = self.rest_lengths
        lengths = np.asarray(lengths, dtype=float)

        nodes = np.empty((frames, len(NODE_NAMES), 3), dtype=float)
        nodes[:, 0] = root

        for index in self.bone_order:
            _, parent_name, child_name = BONES[index]
            parent = NODE_INDEX[parent_name]
            child = NODE_INDEX[child_name]
            nodes[:, child] = (
                nodes[:, parent] + directions[:, index] * lengths[index]
            )

        return nodes

    def to_dict(self) -> dict:
        return {
            "node_names": NODE_NAMES,
            "bone_names": BONE_NAMES,
            "bones": [list(bone) for bone in BONES],
            "rest_nodes": self.rest_nodes.tolist(),
            "rest_directions": self.rest_directions.tolist(),
            "rest_lengths": self.rest_lengths.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Skeleton":
        return cls(np.asarray(payload["rest_nodes"], dtype=float))
