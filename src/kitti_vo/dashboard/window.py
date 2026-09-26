"""Dashboard window: camera, LiDAR, depth, trajectory and metrics."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6 import QtCore, QtWidgets

from ..stereo import DISPARITY_FLOOR
from .sources import FrameView, LiveSource, ReplaySource

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

EST_COLOR = (80, 220, 120)
GT_COLOR = (150, 150, 150)
MAX_LIDAR_POINTS = 40000  # keep the GL view responsive
# Fixed depth colour range, so a colour means the same distance in every frame
# and playback does not flicker. Roughly the useful range of both sensors.
DEPTH_DISPLAY_MAX = 80.0


class Dashboard(QtWidgets.QMainWindow):
    """Five synchronised panels over one frame source."""

    def __init__(self, source: ReplaySource | LiveSource, title: str = "KITTI Visual Odometry"):
        super().__init__()
        self.source = source
        self.live = isinstance(source, LiveSource)
        self.setWindowTitle(title)
        self.resize(1600, 950)

        self.ground_truth = source.ground_truth
        # Depth the matcher assigns to pixels it could not match. Hidden rather
        # than clipped, or "no data" paints as the brightest colour on the map.
        calib = source.seq.calib
        self.unmatched_depth = 0.99 * calib.focal_length * calib.baseline / DISPARITY_FLOOR
        self.poses: list[np.ndarray] = []
        self.track_x: list[float] = []
        self.track_z: list[float] = []
        self.errors: list[float] = []
        self.inlier_rates: list[float] = []
        self.index = 0
        self.playing = False

        self._build_ui()

        if self.live:
            source.frame_ready.connect(self._on_live_frame)
            source.finished_run.connect(lambda: self.statusBar().showMessage("Run finished"))
            source.start()
        else:
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._advance)
            self.show_frame(0)

    # ---------------- layout ----------------

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(self._camera_panel(), 0, 0)
        grid.addWidget(self._lidar_panel(), 0, 1)
        grid.addWidget(self._depth_panel(), 1, 0)
        grid.addWidget(self._trajectory_panel(), 1, 1)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        layout.addLayout(grid, stretch=5)
        layout.addWidget(self._metrics_panel(), stretch=2)
        layout.addLayout(self._controls())

    def _titled(self, widget: QtWidgets.QWidget, title: str) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox(title)
        inner = QtWidgets.QVBoxLayout(box)
        inner.setContentsMargins(4, 12, 4, 4)
        inner.addWidget(widget)
        return box

    def _camera_panel(self) -> QtWidgets.QWidget:
        self.camera_view = pg.GraphicsLayoutWidget()
        vb = self.camera_view.addViewBox(lockAspect=True, invertY=True, enableMouse=True)
        self.camera_image = pg.ImageItem()
        vb.addItem(self.camera_image)
        # Match lines drawn as disconnected segments, keypoints on top.
        self.match_lines = pg.PlotDataItem(pen=pg.mkPen((255, 190, 60, 130), width=1), connect="pairs")
        self.keypoints = pg.ScatterPlotItem(size=4, brush=pg.mkBrush(80, 220, 120, 200), pen=None)
        vb.addItem(self.match_lines)
        vb.addItem(self.keypoints)
        self.camera_vb = vb
        return self._titled(self.camera_view, "Left camera — feature tracking")

    def _lidar_panel(self) -> QtWidgets.QWidget:
        self.lidar_view = gl.GLViewWidget()
        self.lidar_view.setCameraPosition(distance=45, elevation=22, azimuth=180)
        grid_item = gl.GLGridItem()
        grid_item.setSize(100, 100)
        grid_item.setSpacing(5, 5)
        self.lidar_view.addItem(grid_item)
        self.lidar_scatter = gl.GLScatterPlotItem(size=1.6, pxMode=True)
        self.lidar_view.addItem(self.lidar_scatter)
        return self._titled(self.lidar_view, "LiDAR scan")

    def _depth_panel(self) -> QtWidgets.QWidget:
        self.depth_view = pg.GraphicsLayoutWidget()
        vb = self.depth_view.addViewBox(lockAspect=True, invertY=True)
        self.depth_image = pg.ImageItem()
        vb.addItem(self.depth_image)
        self.depth_bar = pg.ColorBarItem(
            colorMap=pg.colormap.get("plasma"), interactive=False, values=(0.0, DEPTH_DISPLAY_MAX)
        )
        self.depth_bar.setImageItem(self.depth_image)
        self.depth_view.addItem(self.depth_bar, row=0, col=1)
        return self._titled(self.depth_view, "Depth (m)")

    def _trajectory_panel(self) -> QtWidgets.QWidget:
        self.traj_view = pg.GraphicsLayoutWidget()
        plot = self.traj_view.addPlot()
        plot.setAspectLocked(True)
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setLabel("bottom", "x", units="m")
        plot.setLabel("left", "z", units="m")
        plot.addLegend(offset=(-10, 10))

        if len(self.ground_truth):
            # Dashed, so it stays readable where the estimate sits on top of it.
            gt = self.ground_truth
            pen = pg.mkPen(GT_COLOR, width=2, style=QtCore.Qt.PenStyle.DashLine)
            plot.plot(gt[:, 0, 3], gt[:, 2, 3], pen=pen, name="ground truth")
        self.traj_curve = plot.plot([], [], pen=pg.mkPen(EST_COLOR, width=2), name="estimated")
        self.traj_head = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(*EST_COLOR), pen=pg.mkPen("k"))
        plot.addItem(self.traj_head)
        return self._titled(self.traj_view, "Trajectory (top down)")

    def _metrics_panel(self) -> QtWidgets.QWidget:
        self.metrics_view = pg.GraphicsLayoutWidget()

        p1 = self.metrics_view.addPlot(title="Position error (m)")
        p1.showGrid(x=True, y=True, alpha=0.2)
        p1.setLabel("bottom", "frame")
        self.error_curve = p1.plot([], [], pen=pg.mkPen((255, 110, 110), width=2))
        self.error_plot = p1

        p2 = self.metrics_view.addPlot(title="PnP inliers (%)")
        p2.showGrid(x=True, y=True, alpha=0.2)
        p2.setLabel("bottom", "frame")
        p2.setYRange(0, 100)
        self.inlier_curve = p2.plot([], [], pen=pg.mkPen((110, 180, 255), width=2))
        self.inlier_plot = p2
        return self._titled(self.metrics_view, "Metrics over time")

    def _controls(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        row.addWidget(self.play_button)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self._on_slider)
        row.addWidget(self.slider, stretch=1)

        self.frame_label = QtWidgets.QLabel("frame 0")
        self.frame_label.setMinimumWidth(150)
        row.addWidget(self.frame_label)

        row.addWidget(QtWidgets.QLabel("speed"))
        self.speed = QtWidgets.QComboBox()
        self.speed.addItems(["0.5x", "1x", "2x", "4x", "max"])
        self.speed.setCurrentText("1x")
        self.speed.currentTextChanged.connect(self._on_speed)
        row.addWidget(self.speed)

        if self.live:
            # Frames arrive in order from the worker; seeking and playback do not apply.
            self.play_button.setEnabled(False)
            self.slider.setEnabled(False)
            self.speed.setEnabled(False)
        else:
            self.slider.setMaximum(max(len(self.source) - 1, 0))
        return row

    # ---------------- playback ----------------

    def _interval_ms(self) -> int:
        text = self.speed.currentText()
        if text == "max":
            return 0
        return int(100 / float(text.rstrip("x")))  # KITTI runs at ~10 Hz

    def toggle_play(self) -> None:
        self._set_playing(not self.playing)

    def _set_playing(self, playing: bool) -> None:
        self.playing = playing
        self.play_button.setText("Pause" if playing else "Play")
        if playing:
            self.timer.start(self._interval_ms())
        else:
            self.timer.stop()

    def _on_speed(self) -> None:
        if self.playing:
            self.timer.start(self._interval_ms())

    def _advance(self) -> None:
        if self.index + 1 >= len(self.source):
            self._set_playing(False)  # stop at the end, never restart
            return
        self.slider.setValue(self.index + 1)

    def _on_slider(self, value: int) -> None:
        if value != self.index:
            self.show_frame(value)

    def show_frame(self, i: int) -> None:
        """Draw frame ``i`` from a replay source."""
        self.index = i
        view = self.source[i]
        self._draw(view)

        # Replay knows the whole trajectory up front.
        traj = self.source.trajectory[: i + 1]
        self.traj_curve.setData(traj[:, 0, 3], traj[:, 2, 3])
        self.traj_head.setData([traj[-1, 0, 3]], [traj[-1, 2, 3]])

        frames = self.source.metrics[: i + 1, 0]
        rates = self.source.metrics[: i + 1, 2] / np.maximum(self.source.metrics[: i + 1, 1], 1) * 100
        self.inlier_curve.setData(frames, rates)
        if self.source.metrics.shape[1] > 5:
            self.error_curve.setData(frames, self.source.metrics[: i + 1, 5])

        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)

    def _on_live_frame(self, view: FrameView) -> None:
        """Append a frame produced by the live worker."""
        self.poses.append(view.pose)
        self._draw(view)

        # Keep the two plotted coordinates directly, rather than rebuilding the
        # whole pose history into an array on every frame.
        self.track_x.append(float(view.pose[0, 3]))
        self.track_z.append(float(view.pose[2, 3]))
        self.traj_curve.setData(self.track_x, self.track_z)
        self.traj_head.setData([self.track_x[-1]], [self.track_z[-1]])

        self.inlier_rates.append(view.n_inliers / max(view.n_points, 1) * 100)
        self.inlier_curve.setData(np.arange(len(self.inlier_rates)), self.inlier_rates)

        if len(self.ground_truth) > len(self.poses) - 1:
            gt = self.ground_truth[len(self.poses) - 1]
            self.errors.append(float(np.linalg.norm(view.pose[:3, 3] - gt[:3, 3])))
            self.error_curve.setData(np.arange(len(self.errors)), self.errors)

    # ---------------- drawing ----------------

    def _draw(self, view: FrameView) -> None:
        self.camera_image.setImage(view.image, autoLevels=False, levels=(0, 255))

        if len(view.points_prev):
            self.keypoints.setData(view.points_prev[:, 0], view.points_prev[:, 1])
            # Interleave start and end points so 'pairs' draws one segment per match.
            xs = np.empty(len(view.points_prev) * 2)
            ys = np.empty_like(xs)
            xs[0::2], xs[1::2] = view.points_prev[:, 0], view.points_curr[:, 0]
            ys[0::2], ys[1::2] = view.points_prev[:, 1], view.points_curr[:, 1]
            self.match_lines.setData(xs, ys)
        else:
            self.keypoints.clear()
            self.match_lines.clear()

        if view.depth is not None:
            d = view.depth
            matched = d < self.unmatched_depth
            shown = np.where(matched, d, np.nan)  # NaN draws as background
            # Levels must be passed explicitly: setImage would otherwise rescale
            # to each frame's own range and the colours would shift as it plays.
            self.depth_image.setImage(shown, autoLevels=False, levels=(0.0, DEPTH_DISPLAY_MAX))

        if view.pointcloud is not None:
            self._draw_lidar(view.pointcloud)

        rate = view.n_inliers / max(view.n_points, 1) * 100
        self.frame_label.setText(f"frame {view.index}  |  {rate:.0f}% inliers")
        self.statusBar().showMessage(
            f"{view.n_inliers}/{view.n_points} inliers  |  {view.elapsed*1000:.0f} ms/frame"
        )

    def _draw_lidar(self, pointcloud: np.ndarray) -> None:
        xyz = pointcloud[:, :3]
        if len(xyz) > MAX_LIDAR_POINTS:
            xyz = xyz[:: len(xyz) // MAX_LIDAR_POINTS + 1]

        # Colour by height, which reads better than a flat cloud.
        z = xyz[:, 2]
        t = np.clip((z - np.percentile(z, 2)) / (np.ptp(z) + 1e-6), 0, 1)
        colors = np.column_stack([0.25 + 0.75 * t, 0.55 * np.ones_like(t), 1.0 - 0.6 * t,
                                  0.85 * np.ones_like(t)]).astype(np.float32)
        # GL view uses x right, y forward, z up; LiDAR is x forward, y left, z up.
        gl_xyz = np.column_stack([-xyz[:, 1], xyz[:, 0], xyz[:, 2]]).astype(np.float32)
        self.lidar_scatter.setData(pos=gl_xyz, color=colors)

    def closeEvent(self, event) -> None:
        if self.live:
            self.source.stop()
        event.accept()