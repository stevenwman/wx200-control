#!/usr/bin/env python3
"""
Interactive trajectory viewer (Dash + Plotly)
--------------------------------------------
- Lists `trajectory_*.npz` in data/ (auto-refreshes, excludes *_quarantined).
- Prev/Next navigation + dropdown; quarantine button renames file with _quarantined suffix.
- Views: 3D EE/Object poses (ArUco), marker visibility heatmap, actions over time.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import Dash, dcc, html, Input, Output, State


# --------------------------------------------------------------------------- #
# Constants / styles
# --------------------------------------------------------------------------- #
DATA_DIR = Path(__file__).parent / "data/"

DARK_BG = "#111111"
DARK_FG = "#EEEEEE"
CARD_BG = "#333333"
BORDER = "#555555"

SELECT_DARK_CSS = """
.Select-control {
  background-color: #333333 !important;
  color: #EEEEEE !important;
  border-color: #555555 !important;
}
.Select-value-label { color: #EEEEEE !important; }
.Select-input > input { color: #EEEEEE !important; }
.Select-menu-outer {
  background-color: #333333 !important;
  border-color: #555555 !important;
}
.Select-option {
  background-color: #333333 !important;
  color: #EEEEEE !important;
}
.Select-option.is-focused {
  background-color: #555555 !important;
  color: #FFFFFF !important;
}
.Select-option.is-selected {
  background-color: #444444 !important;
  color: #FFFFFF !important;
}
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def list_trajectory_files():
    """Sorted trajectory_*.npz files, excluding quarantined (prefix)."""
    return [
        f for f in sorted(DATA_DIR.glob("trajectory_*.npz"))
        if not f.name.startswith("quarantined_")
    ]


def load_trajectory(npz_path: Path):
    """Load one trajectory file and return a dict of arrays/metadata."""
    data = np.load(npz_path, allow_pickle=True)

    timestamps = data["timestamps"]
    states = data["states"]
    actions = data["actions"]
    metadata = data["metadata"].item() if "metadata" in data else {}

    # EE pose from IK / Mink (target/commanded) and from encoders (FK ground truth)
    if "ee_poses_target" in data:
        ee_poses_mink = data["ee_poses_target"]
    elif "ee_poses_debug" in data:
        # Backwards compatibility with older files
        ee_poses_mink = data["ee_poses_debug"]
    else:
        ee_poses_mink = None

    ee_poses_encoder = data["ee_pose_encoder"] if "ee_pose_encoder" in data else None

    aruco_keys = [
        "aruco_ee_in_world",
        "aruco_object_in_world",
        "aruco_ee_in_object",
        "aruco_object_in_ee",
        "aruco_visibility",
    ]
    aruco_data = {k: data[k] for k in aruco_keys if k in data}
    
    # Also load smoothed versions if available
    smoothed_keys = [f"smoothed_{k}" for k in aruco_keys if k != "aruco_visibility"]
    for k in smoothed_keys:
        if k in data:
            aruco_data[k] = data[k]

    return dict(
        timestamps=timestamps,
        states=states,
        actions=actions,
        metadata=metadata,
        ee_poses_mink=ee_poses_mink,
        ee_poses_encoder=ee_poses_encoder,
        aruco_data=aruco_data,
    )


def _visibility_masks(aruco_data, n):
    vis = aruco_data.get("aruco_visibility", np.ones((n, 3)))
    valid_ee = (vis[:, 0] > 0.5) & (vis[:, 2] > 0.5)
    valid_obj = (vis[:, 0] > 0.5) & (vis[:, 1] > 0.5)
    lost_ee = ~valid_ee
    lost_obj = ~valid_obj
    return vis, valid_ee, valid_obj, lost_ee, lost_obj


def make_3d_figure(traj, title_suffix: str):
    """3D Plot of EE + object poses; marks lost tracking with 'X'."""
    timestamps = traj["timestamps"]
    aruco = traj["aruco_data"]
    if "aruco_ee_in_world" not in aruco or "aruco_object_in_world" not in aruco:
        return go.Figure(layout={"title": "World-frame ArUco data missing"})

    ee_w = aruco["aruco_ee_in_world"]
    obj_w = aruco["aruco_object_in_world"]
    vis, valid_ee, valid_obj, lost_ee, lost_obj = _visibility_masks(aruco, len(timestamps))

    fig = go.Figure()

    if np.any(valid_ee):
        fig.add_trace(go.Scatter3d(
            x=ee_w[valid_ee, 0], y=ee_w[valid_ee, 1], z=ee_w[valid_ee, 2],
            mode="lines", name="EE path",
            line=dict(color="blue", width=4), opacity=0.8,
        ))
    if np.any(lost_ee):
        fig.add_trace(go.Scatter3d(
            x=ee_w[lost_ee, 0], y=ee_w[lost_ee, 1], z=ee_w[lost_ee, 2],
            mode="markers", name="EE lost track",
            marker=dict(size=4, color="blue", symbol="x"), opacity=0.9,
        ))
    if np.any(valid_obj):
        fig.add_trace(go.Scatter3d(
            x=obj_w[valid_obj, 0], y=obj_w[valid_obj, 1], z=obj_w[valid_obj, 2],
            mode="lines+markers", name="Object",
            line=dict(color="red", width=3),
            marker=dict(size=3, color="red", opacity=0.7),
        ))
    if np.any(lost_obj):
        fig.add_trace(go.Scatter3d(
            x=obj_w[lost_obj, 0], y=obj_w[lost_obj, 1], z=obj_w[lost_obj, 2],
            mode="markers", name="Object lost track",
            marker=dict(size=4, color="red", symbol="x"), opacity=0.9,
        ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers", name="World origin",
        marker=dict(size=8, color="green", symbol="x"),
    ))

    fig.update_layout(
        title=f"3D EE & Object Trajectory – {title_suffix}",
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing="constant"),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
    )
    return fig


def _draw_rgb_axes(fig, pos, quat_wxyz, axis_length=0.008, name_prefix="", opacity=0.6, showlegend=False):
    """Draw RGB coordinate axes (X=red, Y=green, Z=blue) at a pose."""
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rot = R.from_quat(quat_xyzw)
    R_matrix = rot.as_matrix()
    
    axes_local = np.array([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
    axes_world = (R_matrix @ axes_local.T).T + pos
    
    colors = ['red', 'green', 'blue']
    labels = ['X', 'Y', 'Z']
    
    for i in range(3):
        fig.add_trace(go.Scatter3d(
            x=[pos[0], axes_world[i, 0]],
            y=[pos[1], axes_world[i, 1]],
            z=[pos[2], axes_world[i, 2]],
            mode='lines',
            name=f'{name_prefix} {labels[i]}',
            line=dict(color=colors[i], width=6),  # Much thicker frame lines
            showlegend=showlegend and (i == 0),
            legendgroup=name_prefix,
            opacity=opacity,
        ))


def make_smoothed_3d_figure(traj, title_suffix: str):
    """3D plot of smoothed trajectories with green connectors and coordinate frames at every step."""
    timestamps = traj["timestamps"]
    aruco = traj["aruco_data"]
    
    # Check for smoothed data
    ee_smooth_key = "smoothed_aruco_ee_in_world"
    obj_smooth_key = "smoothed_aruco_object_in_world"
    
    if ee_smooth_key not in aruco or obj_smooth_key not in aruco:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="Smoothed trajectory data not available.<br>Run: python smooth_aruco_trajectory.py [trajectory_file.npz]",
            showarrow=False,
            font=dict(size=14, color="#CCCCCC"),
        )
        fig.update_layout(
            title=f"Smoothed Trajectory – {title_suffix}",
            template="plotly_dark",
            paper_bgcolor=DARK_BG,
            height=400,
        )
        return fig
    
    ee_smooth = aruco[ee_smooth_key]
    obj_smooth = aruco[obj_smooth_key]
    
    fig = go.Figure()
    
    # Smoothed trajectories with markers (thicker lines, bigger dots)
    fig.add_trace(go.Scatter3d(
        x=ee_smooth[:, 0], y=ee_smooth[:, 1], z=ee_smooth[:, 2],
        mode="lines+markers", name="EE smoothed",
        line=dict(color="blue", width=5), 
        marker=dict(size=4, color="blue", opacity=0.8),
        opacity=0.7,
    ))
    fig.add_trace(go.Scatter3d(
        x=obj_smooth[:, 0], y=obj_smooth[:, 1], z=obj_smooth[:, 2],
        mode="lines+markers", name="Object smoothed",
        line=dict(color="red", width=5),
        marker=dict(size=4, color="red", opacity=0.8),
        opacity=0.7,
    ))
    
    # Yellow connectors at every timestep (thicker)
    xs, ys, zs = [], [], []
    for i in range(len(timestamps)):
        if np.all(np.isfinite(ee_smooth[i, :3])) and np.all(np.isfinite(obj_smooth[i, :3])):
            xs.extend([ee_smooth[i, 0], obj_smooth[i, 0], np.nan])
            ys.extend([ee_smooth[i, 1], obj_smooth[i, 1], np.nan])
            zs.extend([ee_smooth[i, 2], obj_smooth[i, 2], np.nan])
    
    if xs:
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines", name="EE↔Object connectors",
            line=dict(color="yellow", width=2), opacity=0.5,
        ))
    
    # Coordinate frames at EVERY timestep
    N = len(timestamps)
    frame_indices = list(range(N))
    
    # Adjust opacity and size based on number of frames (smaller frames)
    if N > 200:
        axis_length = 0.004
        opacity = 0.3
    elif N > 100:
        axis_length = 0.005
        opacity = 0.4
    else:
        axis_length = 0.006
        opacity = 0.5
    
    # Draw frames for EE at every step (smaller)
    for idx in frame_indices:
        if np.all(np.isfinite(ee_smooth[idx, :3])) and np.all(np.isfinite(ee_smooth[idx, 3:7])):
            pos = ee_smooth[idx, :3]
            quat = ee_smooth[idx, 3:7]
            _draw_rgb_axes(fig, pos, quat, axis_length=axis_length, name_prefix=f"EE_{idx}", opacity=opacity, showlegend=(idx == 0))
    
    # Draw frames for Object at every step (smaller)
    for idx in frame_indices:
        if np.all(np.isfinite(obj_smooth[idx, :3])) and np.all(np.isfinite(obj_smooth[idx, 3:7])):
            pos = obj_smooth[idx, :3]
            quat = obj_smooth[idx, 3:7]
            _draw_rgb_axes(fig, pos, quat, axis_length=axis_length, name_prefix=f"Obj_{idx}", opacity=opacity, showlegend=(idx == 0))
    
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers", name="World origin",
        marker=dict(size=8, color="black", symbol="x"),
    ))
    
    fig.update_layout(
        title=f"Smoothed Trajectory with Frames (every step) – {title_suffix}",
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing="constant"),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
    )
    return fig


def make_3d_connections_figure(traj, title_suffix: str):
    """3D plot showing EE/Object paths plus green connectors at each timestep.

    - Blue/red lines: full EE/Object trajectories (using visibility masks, like main 3D plot).
    - Green segments: connect EE and Object poses at every timestep where both poses exist,
      even if ArUco visibility flags them as not visible. This makes it easy to see cases
      where multiple EE points map to the same Object point in space.
    """
    timestamps = traj["timestamps"]
    aruco = traj["aruco_data"]
    if "aruco_ee_in_world" not in aruco or "aruco_object_in_world" not in aruco:
        return go.Figure(layout={"title": "World-frame ArUco data missing"})

    ee_w = aruco["aruco_ee_in_world"]
    obj_w = aruco["aruco_object_in_world"]
    vis, valid_ee, valid_obj, _, _ = _visibility_masks(aruco, len(timestamps))

    # For connectors, ignore visibility and just require finite positions for both
    has_pose = (
        np.all(np.isfinite(ee_w[:, :3]), axis=1)
        & np.all(np.isfinite(obj_w[:, :3]), axis=1)
    )
    if not np.any(has_pose):
        return go.Figure(layout={"title": "No timesteps with valid EE & Object poses"})

    # Build one long line trace with NaN separators between segments
    xs, ys, zs = [], [], []
    for i in np.where(has_pose)[0]:
        xs.extend([ee_w[i, 0], obj_w[i, 0], np.nan])
        ys.extend([ee_w[i, 1], obj_w[i, 1], np.nan])
        zs.extend([ee_w[i, 2], obj_w[i, 2], np.nan])

    fig = go.Figure()

    # EE / Object trajectories (same style as main plot)
    if np.any(valid_ee):
        fig.add_trace(go.Scatter3d(
            x=ee_w[valid_ee, 0], y=ee_w[valid_ee, 1], z=ee_w[valid_ee, 2],
            mode="lines",
            name="EE path",
            line=dict(color="blue", width=3),
            opacity=0.8,
        ))
    if np.any(valid_obj):
        fig.add_trace(go.Scatter3d(
            x=obj_w[valid_obj, 0], y=obj_w[valid_obj, 1], z=obj_w[valid_obj, 2],
            mode="lines",
            name="Object path",
            line=dict(color="red", width=3),
            opacity=0.8,
        ))

    # EE / Object points (visible timesteps only)
    fig.add_trace(go.Scatter3d(
        x=ee_w[valid_ee, 0], y=ee_w[valid_ee, 1], z=ee_w[valid_ee, 2],
        mode="markers", name="EE (visible)",
        marker=dict(size=3, color="blue", opacity=0.8),
    ))
    fig.add_trace(go.Scatter3d(
        x=obj_w[valid_obj, 0], y=obj_w[valid_obj, 1], z=obj_w[valid_obj, 2],
        mode="markers", name="Object (visible)",
        marker=dict(size=3, color="red", opacity=0.8),
    ))

    # Green connectors
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        name="EE–Object connector",
        line=dict(color="green", width=3),
        opacity=0.7,
    ))

    fig.update_layout(
        title=f"EE↔Object Connectors (Green) – {title_suffix}",
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing="constant"),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
    )
    return fig


def make_actions_figure(traj, title_suffix: str):
    """Build a multi-subplot figure for actions over time, highlighting gripper."""
    timestamps = traj["timestamps"]
    actions = traj["actions"]
    metadata = traj["metadata"]

    action_labels = metadata.get(
        "action_labels", ["vx", "vy", "vz", "wx", "wy", "wz", "gripper"]
    )
    num_actions = actions.shape[1]

    # Identify gripper dimension
    if "gripper" in action_labels:
        gripper_idx = action_labels.index("gripper")
    elif "gripper_target" in action_labels:
        gripper_idx = action_labels.index("gripper_target")
    else:
        gripper_idx = num_actions - 1

    non_gripper_indices = [i for i in range(num_actions) if i != gripper_idx]
    n_non_gripper = len(non_gripper_indices)
    nrows = n_non_gripper + 1

    fig = make_subplots(
        rows=nrows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[action_labels[i] for i in non_gripper_indices] + ["gripper"],
    )

    row = 1
    for i in non_gripper_indices:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=actions[:, i],
                mode="lines",
                name=action_labels[i],
            ),
            row=row,
            col=1,
        )
        row += 1

    # Gripper subplot
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=actions[:, gripper_idx],
            mode="lines",
            name=action_labels[gripper_idx],
            line=dict(color="red"),
        ),
        row=nrows,
        col=1,
    )

    fig.update_xaxes(title_text="Time (s)", row=nrows, col=1)
    fig.update_yaxes(title_text="Value")

    fig.update_layout(
        height=300 + 150 * nrows,
        title=f"Actions Over Time – {title_suffix}",
        showlegend=False,
        margin=dict(l=60, r=10, b=40, t=40),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
    )
    return fig


def make_mink_encoder_figure(traj, title_suffix: str):
    """
    3D plot comparing Mink/IK target EE pose vs encoder-based EE pose.

    - Single 3D plot
    - Two colored lines (target vs encoder)
    - Coordinate frames (RGB axes) drawn along each trajectory
    """
    timestamps = traj["timestamps"]
    mink = traj.get("ee_poses_mink", None)
    encoder = traj.get("ee_poses_encoder", None)

    if mink is None and encoder is None:
        return go.Figure(layout={"title": "No Mink/encoder EE pose data (ee_poses_mink / ee_pose_encoder) in this trajectory"})

    fig = go.Figure()

    # Ensure matching length where both are present
    if mink is not None:
        N_mink = mink.shape[0]
    else:
        N_mink = 0
    if encoder is not None:
        N_enc = encoder.shape[0]
    else:
        N_enc = 0

    N = max(N_mink, N_enc)
    if N == 0:
        return go.Figure(layout={"title": "Mink/encoder EE pose arrays are empty"})

    # Truncate to smallest available length across timestamps / poses
    max_len = min(len(timestamps), N_mink if N_mink > 0 else len(timestamps), N_enc if N_enc > 0 else len(timestamps))
    ts = timestamps[:max_len]

    if mink is not None:
        mink = mink[:max_len]
    if encoder is not None:
        encoder = encoder[:max_len]

    # Plot position trajectories
    if mink is not None:
        pos_mink = mink[:, :3]
        fig.add_trace(
            go.Scatter3d(
                x=pos_mink[:, 0],
                y=pos_mink[:, 1],
                z=pos_mink[:, 2],
                mode="lines",
                name="Target EE (Mink / IK)",
                line=dict(color="cyan", width=4),
                opacity=0.9,
            )
        )

    if encoder is not None:
        pos_enc = encoder[:, :3]
        fig.add_trace(
            go.Scatter3d(
                x=pos_enc[:, 0],
                y=pos_enc[:, 1],
                z=pos_enc[:, 2],
                mode="lines",
                name="Encoder EE (FK)",
                line=dict(color="magenta", width=4),
                opacity=0.9,
            )
        )

    # Draw RGB frames along each trajectory (sampled if long)
    if mink is not None or encoder is not None:
        num_frames = max_len
        if num_frames > 300:
            step = 10
        elif num_frames > 150:
            step = 5
        else:
            step = 2

        # Axis size / opacity based on length
        if num_frames > 200:
            axis_length = 0.006
            opacity = 0.35
        elif num_frames > 100:
            axis_length = 0.007
            opacity = 0.4
        else:
            axis_length = 0.008
            opacity = 0.5

        for idx in range(0, num_frames, step):
            showlegend = idx == 0
            if mink is not None and np.all(np.isfinite(mink[idx, :7])):
                pos = mink[idx, :3]
                quat = mink[idx, 3:7]
                _draw_rgb_axes(
                    fig,
                    pos,
                    quat,
                    axis_length=axis_length,
                    name_prefix="Target",
                    opacity=opacity,
                    showlegend=showlegend,
                )
            if encoder is not None and np.all(np.isfinite(encoder[idx, :7])):
                pos = encoder[idx, :3]
                quat = encoder[idx, 3:7]
                _draw_rgb_axes(
                    fig,
                    pos,
                    quat,
                    axis_length=axis_length,
                    name_prefix="Encoder",
                    opacity=opacity,
                    showlegend=showlegend,
                )

    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            name="World origin",
            marker=dict(size=6, color="white", symbol="cross"),
        )
    )

    fig.update_layout(
        title=f"Mink vs Encoder EE Trajectory – {title_suffix}",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing="constant"),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
    )
    return fig


def make_visibility_figure(traj, title_suffix: str):
    """Build a heatmap showing when each ArUco marker is NOT visible."""
    timestamps = traj["timestamps"]
    aruco_data = traj["aruco_data"]

    if "aruco_visibility" not in aruco_data:
        fig = go.Figure()
        fig.update_layout(
            title="No aruco_visibility data in this trajectory",
        )
        return fig

    vis = aruco_data["aruco_visibility"]  # shape (N, 3), 1 = visible, 0 = not
    # Emphasize NOT visible (1 = not visible, 0 = visible)
    not_vis = 1.0 - vis.astype(float)  # shape (N, 3)
    z = not_vis.T  # (3, N) so each row is a marker over time

    marker_labels = ["World (0)", "Object (2)", "Gripper (3)"]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=timestamps,
            y=marker_labels,
            colorscale=[
                [0.0, "white"],   # visible
                [1.0, "red"],     # not visible
            ],
            colorbar=dict(
                title="Not visible",
                tickvals=[0, 1],
                ticktext=["visible", "not visible"],
            ),
            showscale=True,
        )
    )

    fig.update_layout(
        title=f"Marker Visibility Over Time – {title_suffix}",
        xaxis_title="Time (s)",
        yaxis_title="Marker",
        height=250,
        margin=dict(l=60, r=20, b=40, t=40),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
    )
    return fig


def create_app() -> Dash:
    external_stylesheets = ["https://cdnjs.cloudflare.com/ajax/libs/semantic-ui/2.4.1/semantic.min.css"]
    app = Dash(__name__, external_stylesheets=external_stylesheets)

    def file_options():
        return [{"label": f.name, "value": str(f)} for f in list_trajectory_files()]

    options = file_options()
    initial_value = options[-1]["value"] if options else None

    app.layout = html.Div(
        style={
            "fontFamily": "sans-serif",
            "padding": "1rem",
            "backgroundColor": DARK_BG,
            "color": DARK_FG,
            "minHeight": "100vh",
        },
        children=[
            # Note: for extra theming, consider adding a CSS file under assets/ instead of inline Style
            html.H2("Trajectory Viewer", style={"color": "#FFFFFF"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Trajectory file:", style={"color": "#DDDDDD"}),
                            html.Span(
                                f"({len(options)} valid)",
                                id="traj-count",
                                style={
                                    "color": "#AAAAAA",
                                    "fontSize": "0.85rem",
                                    "marginLeft": "0.5rem",
                                },
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    dcc.Dropdown(
                        id="traj-file-dropdown",
                        options=options,
                        value=initial_value,
                        clearable=False,
                        # Make the text clearly readable (black on light dropdown background)
                        style={"width": "400px", "color": "#000000"},
                    ),
                    html.Div(
                        [
                            html.Button("← Prev", id="traj-prev-btn", n_clicks=0, style={"marginRight": "0.5rem"}),
                            html.Button("Next →", id="traj-next-btn", n_clicks=0, style={"marginRight": "0.5rem"}),
                            html.Button(
                                "↻ Refresh files",
                                id="traj-refresh-btn",
                                n_clicks=0,
                                style={"marginRight": "0.5rem"},
                            ),
                            html.Button(
                                "🚫 Quarantine",
                                id="traj-quarantine-btn",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "#8B0000",
                                    "color": "white",
                                    "border": "none",
                                    "padding": "0.5rem 1rem",
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                        style={"marginTop": "0.5rem", "display": "flex", "alignItems": "center", "gap": "0.5rem"},
                    ),
                    html.Div(
                        id="traj-summary",
                        style={"marginTop": "0.5rem", "fontSize": "0.9rem", "color": "#CCCCCC"},
                    ),
                ]
            ),
            html.Hr(),
            html.Details(
                [
                    html.Summary("3D EE & Object Motion (World Frame)"),
                    dcc.Graph(id="traj-3d-graph"),
                ],
                open=True,
            ),
            html.Details(
                [
                    html.Summary("EE–Object Spatial Relationship (Connectors)"),
                    dcc.Graph(id="traj-3d-connect-graph"),
                ],
                open=True,
            ),
            html.Details(
                [
                    html.Summary("Mink vs Encoder EE Pose (Target vs Actual)"),
                    dcc.Graph(id="traj-mink-encoder-graph"),
                ],
                open=True,
            ),
            html.Details(
                [
                    html.Summary("Smoothed Trajectory with Coordinate Frames"),
                    dcc.Graph(id="traj-smoothed-3d-graph"),
                ],
                open=True,
            ),
            html.H3("Marker Visibility Over Time"),
            dcc.Graph(id="traj-vis-graph"),
            html.H3("Actions Over Time"),
            dcc.Graph(id="traj-actions-graph"),
        ],
    )

    # Refresh file list & handle quarantine (manual via buttons)
    @app.callback(
        Output("traj-file-dropdown", "options"),
        Output("traj-file-dropdown", "value"),
        Output("traj-count", "children"),
        Input("traj-refresh-btn", "n_clicks"),
        Input("traj-quarantine-btn", "n_clicks"),
        State("traj-file-dropdown", "value"),
        State("traj-file-dropdown", "options"),
        prevent_initial_call=True,
    )
    def _update_file_list(_refresh_clicks, _quarantine_clicks, current_value, old_options):
        ctx = dash.callback_context
        triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        # Handle quarantine
        if triggered == "traj-quarantine-btn" and current_value:
            npz_path = Path(current_value)
            if npz_path.exists() and not npz_path.name.startswith("quarantined_"):
                new_name = "quarantined_" + npz_path.name
                try:
                    npz_path.rename(npz_path.parent / new_name)
                    print(f"Quarantined: {npz_path.name} -> {new_name}")
                except Exception as e:
                    print(f"Error quarantining file: {e}")

        new_options = file_options()
        new_values = [o["value"] for o in new_options]
        count_text = f"({len(new_options)} valid)"

        # Keep current if still valid; else pick last available
        if current_value in new_values:
            new_value = current_value
        else:
            new_value = new_values[-1] if new_values else None

        return new_options, new_value, count_text

    # Prev/Next navigation
    @app.callback(
        Output("traj-file-dropdown", "value", allow_duplicate=True),
        Input("traj-prev-btn", "n_clicks"),
        Input("traj-next-btn", "n_clicks"),
        State("traj-file-dropdown", "value"),
        State("traj-file-dropdown", "options"),
        prevent_initial_call=True,
    )
    def _navigate_files(prev_clicks, next_clicks, current_value, options):
        if not options:
            return current_value
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_value
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        values = [opt["value"] for opt in options]
        idx = values.index(current_value) if current_value in values else 0
        if triggered_id == "traj-prev-btn":
            idx = max(0, idx - 1)
        elif triggered_id == "traj-next-btn":
            idx = min(len(values) - 1, idx + 1)
        return values[idx]

    # Main plot updater
    @app.callback(
        Output("traj-summary", "children"),
        Output("traj-3d-graph", "figure"),
        Output("traj-3d-connect-graph", "figure"),
        Output("traj-mink-encoder-graph", "figure"),
        Output("traj-smoothed-3d-graph", "figure"),
        Output("traj-vis-graph", "figure"),
        Output("traj-actions-graph", "figure"),
        Input("traj-file-dropdown", "value"),
    )
    def update_plots(npz_path_str):
        if not npz_path_str:
            empty = go.Figure(layout={"title": "No trajectory selected"})
            return "No trajectory selected.", empty, empty, empty, empty, empty, empty

        npz_path = Path(npz_path_str)
        try:
            traj = load_trajectory(npz_path)
        except Exception as e:
            msg = f"Error loading {npz_path.name}: {e}"
            err = go.Figure(layout={"title": msg})
            return msg, err, err, err, err, err, err

        ts = traj["timestamps"]
        metadata = traj["metadata"]
        duration = float(ts[-1]) if len(ts) else 0.0
        freq = metadata.get("control_frequency", None)

        summary_lines = [
            f"File: {npz_path.name}",
            f"Samples: {len(ts)}",
            f"Duration: {duration:.2f} s",
        ]
        if freq is not None:
            summary_lines.append(f"Control frequency: {freq:.1f} Hz")

        # ArUco drop stats
        aruco = traj["aruco_data"]
        if "aruco_visibility" in aruco:
            vis = aruco["aruco_visibility"]
            total = vis.shape[0]
            drops = [int(np.sum(vis[:, i] < 0.5)) for i in range(vis.shape[1])]
            any_lost = np.any(vis < 0.5, axis=1)
            pct_any_lost = 100.0 * any_lost.mean() if total > 0 else 0.0
            summary_lines.append(
                f"ArUco frame drops (not visible / total): "
                f"World {drops[0]}/{total}, Object {drops[1]}/{total}, Gripper {drops[2]}/{total}"
            )
            summary_lines.append(
                f"Percent of trajectory with ANY marker lost: {pct_any_lost:.1f}%"
            )

        summary = html.Div([html.Div(line) for line in summary_lines])
        suffix = npz_path.name
        return (
            summary,
            make_3d_figure(traj, suffix),
            make_3d_connections_figure(traj, suffix),
            make_mink_encoder_figure(traj, suffix),
            make_smoothed_3d_figure(traj, suffix),
            make_visibility_figure(traj, suffix),
            make_actions_figure(traj, suffix),
        )

    return app


def main():
    traj_files = list_trajectory_files()
    if not traj_files:
        print(f"No trajectory_*.npz files found in {DATA_DIR}.")
        return

    print("Available trajectories:")
    for f in traj_files:
        print("  -", f)

    app = create_app()
    # Dash 3.x: use app.run instead of deprecated app.run_server
    app.run(debug=False)


if __name__ == "__main__":
    main()


