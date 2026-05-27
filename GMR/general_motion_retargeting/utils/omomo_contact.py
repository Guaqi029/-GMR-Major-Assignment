import numpy as np


CONTACT_BODIES = (
    "left_wrist",
    "right_wrist",
    "left_elbow",
    "right_elbow",
)


def _closing(labels, min_frames):
    if min_frames <= 1 or len(labels) == 0:
        return labels
    labels = labels.astype(bool, copy=True)
    gap = 0
    start = None
    for idx, flag in enumerate(labels):
        if flag:
            if gap > 0 and start is not None and gap < min_frames:
                labels[start:idx] = True
            gap = 0
            start = idx + 1
        else:
            if gap == 0:
                start = idx
            gap += 1
    return labels


def _remove_short_runs(labels, min_frames):
    if min_frames <= 1 or len(labels) == 0:
        return labels
    labels = labels.astype(bool, copy=True)
    start = None
    for idx, flag in enumerate(np.append(labels, False)):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start < min_frames:
                labels[start:idx] = False
            start = None
    return labels


def detect_contact_phase(
    human_frames,
    object_positions,
    threshold=0.35,
    min_frames=3,
):
    if object_positions is None:
        return np.zeros(len(human_frames), dtype=bool), np.full(len(human_frames), np.nan)

    frame_count = min(len(human_frames), len(object_positions))
    contact_labels = np.zeros(frame_count, dtype=bool)
    min_distances = np.full(frame_count, np.nan, dtype=float)

    for idx in range(frame_count):
        frame = human_frames[idx]
        obj_pos = object_positions[idx]
        distances = []
        for body_name in CONTACT_BODIES:
            body = frame.get(body_name)
            if body is None:
                continue
            distances.append(np.linalg.norm(np.asarray(body[0], dtype=float) - obj_pos))
        if not distances:
            continue
        min_distance = float(np.min(distances))
        min_distances[idx] = min_distance
        contact_labels[idx] = min_distance <= threshold

    contact_labels = _closing(contact_labels, min_frames=min_frames)
    contact_labels = _remove_short_runs(contact_labels, min_frames=min_frames)
    return contact_labels, min_distances
