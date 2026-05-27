import numpy as np


def moving_average_smooth(values, window_size=5):
    if window_size <= 1 or len(values) == 0:
        return values.copy()
    if window_size % 2 == 0:
        window_size += 1
    pad = window_size // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window_size, dtype=float) / window_size
    smoothed = np.empty_like(values, dtype=float)
    for col in range(values.shape[1]):
        smoothed[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
    return smoothed


def smooth_selected_dofs(dof_pos, dof_indices, window_size=5):
    if len(dof_indices) == 0 or window_size <= 1:
        return dof_pos
    smoothed = dof_pos.copy()
    selected = smoothed[:, dof_indices]
    smoothed[:, dof_indices] = moving_average_smooth(selected, window_size=window_size)
    return smoothed
