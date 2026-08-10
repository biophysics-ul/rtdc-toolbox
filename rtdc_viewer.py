# -*- coding: utf-8 -*-
"""Simple PyQt5 viewer for RT-DC files."""

import os
import sys

import dclab
import numpy as np
import rtdc_toolbox

from PyQt5.QtCore import Qt, QRect, pyqtSignal
from PyQt5.QtGui import (
    QColor, QFont, QImage, QKeySequence, QPainter, QPixmap, QTextCursor
)
from PyQt5.QtWidgets import (
    QAction, QApplication, QComboBox, QDialog, QFileDialog, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QShortcut,
    QSizePolicy, QSlider, QStyle, QToolBar, QVBoxLayout, QWidget
)


N_COLS = 5
N_ROWS = 10
N_PER_PAGE = N_COLS * N_ROWS
LEGACY_CLASS_FEATURE = "userdef1"
LEGACY_CONFIDENCE_FEATURE = "userdef2"

# ["all"] = show all scalar features available through dclab.
# Otherwise, list only the feature names you want to show.
SHOW_FEATURES = ["frame","index","nevents",
                 "area_um","deform","area_ratio","volume",
                 "pos_x","pos_y","size_x","size_y"
    ]


def qimage_from_array(arr):
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        h, w = arr.shape
        image = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, _ = arr.shape
        image = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    return image.copy()


def draw_text(painter, rect, text, alignment):
    painter.setPen(QColor("black"))
    painter.drawText(rect.translated(1, 1), alignment, text)
    painter.setPen(QColor("white"))
    painter.drawText(rect, alignment, text)


def get_rtdc_index(ds, event_idx):
    """Return the RTDC 'index' feature; fall back to the Python event position."""
    try:
        return int(ds["index"][event_idx])
    except Exception:
        return int(event_idx)


def make_event_image(
    ds, event_idx, show_contour=False,
    class_feature="", confidence_feature="", class_names=None
):
    img = np.asarray(ds["image"][event_idx]).copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    rgb = np.repeat(img[:, :, None], 3, axis=2)

    if show_contour:
        try:
            contour = np.asarray(ds["contour"][event_idx], dtype=int)
        except Exception:
            contour = None
        if contour is not None and contour.ndim == 2 and contour.shape[1] >= 2:
            x, y = contour[:, 0], contour[:, 1]
            ok = (
                (x >= 0) & (x < rgb.shape[1]) &
                (y >= 0) & (y < rgb.shape[0])
            )
            rgb[y[ok], x[ok]] = (255, 0, 0)

    image = qimage_from_array(rgb)
    painter = QPainter(image)
    painter.setFont(QFont("Arial", 8))
    w, h = image.width(), image.height()

    text = str(get_rtdc_index(ds, event_idx))
    if class_feature:
        try:
            class_id = int(ds[class_feature][event_idx])
            text += "-" + (class_names or {}).get(class_id, str(class_id))
        except Exception:
            pass

    draw_text(
        painter, QRect(4, 2, w - 8, 18), text,
        Qt.AlignLeft | Qt.AlignTop
    )

    try:
        event_time = float(ds["time"][event_idx])
    except Exception:
        event_time = None
    if event_time is not None:
        draw_text(
            painter, QRect(4, 2, w - 8, 18),
            str(round(event_time, 1)),
            Qt.AlignRight | Qt.AlignTop
        )

    if confidence_feature:
        try:
            confidence = f"{float(ds[confidence_feature][event_idx]):.3f}"
        except Exception:
            confidence = None
        if confidence is not None:
            draw_text(
                painter, QRect(4, h - 20, w - 8, 18), confidence,
                Qt.AlignLeft | Qt.AlignBottom
            )

    painter.end()
    return image



class EventTile(QLabel):
    clicked = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.event_idx = None
        self.source_pixmap = None
        self.setAlignment(Qt.AlignCenter)
        # Ignore the pixmap's size hint so one occupied tile cannot enlarge its
        # row/column when the selected class contains only a few events.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(1, 1)
        self.setStyleSheet("background:black; border:1px solid #555;")

    def set_event(self, event_idx, image):
        self.event_idx = int(event_idx)
        self.source_pixmap = QPixmap.fromImage(image)
        self.scale_pixmap()

    def clear_event(self):
        self.event_idx = None
        self.source_pixmap = None
        self.clear()

    def scale_pixmap(self):
        if self.source_pixmap is not None:
            self.setPixmap(self.source_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.scale_pixmap()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.event_idx is not None:
            self.clicked.emit(self.event_idx)
        super().mousePressEvent(event)


class RTDCViewerQt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ds = None
        self.rtdc_path = ""
        self.classifiers = []
        self.class_feature = ""
        self.confidence_feature = ""
        self.class_names = {}
        self.class_list = np.array([], dtype=int)
        self.active_class = None
        self.img_table = []
        self.show_contours = False
        self.inspect_dialog = None

        self.setWindowTitle("RTDC Viewer Qt")
        self.resize(1400, 900)
        self.build_ui()
        self.build_shortcuts()

    def build_ui(self):
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction(
            self.style().standardIcon(QStyle.SP_DialogOpenButton),
            "Open", self
        )
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Classifier: "))
        self.classifier_combo = QComboBox()
        self.classifier_combo.setMinimumWidth(220)
        self.classifier_combo.currentIndexChanged.connect(self.classifier_changed)
        toolbar.addWidget(self.classifier_combo)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Class: "))
        self.class_combo = QComboBox()
        self.class_combo.setMinimumWidth(160)
        self.class_combo.currentIndexChanged.connect(self.class_changed)
        toolbar.addWidget(self.class_combo)

        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        grid = QGridLayout()
        grid.setSpacing(2)
        self.tiles = []
        for i in range(N_PER_PAGE):
            tile = EventTile()
            tile.clicked.connect(self.inspect_event)
            self.tiles.append(tile)
            grid.addWidget(tile, i // N_COLS, i % N_COLS)

        # Keep all 5 x 10 cells exactly equal, independent of how many tiles
        # currently contain images.
        for col in range(N_COLS):
            grid.setColumnStretch(col, 1)
        for row in range(N_ROWS):
            grid.setRowStretch(row, 1)

        main_layout.addLayout(grid, 1)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Image slider:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setTracking(True)  # Refresh continuously while dragging.
        self.slider.valueChanged.connect(self.update_display)
        slider_layout.addWidget(self.slider, 1)
        self.page_label = QLabel("0 / 0")
        slider_layout.addWidget(self.page_label)
        main_layout.addLayout(slider_layout)

        self.setCentralWidget(central)

    def build_shortcuts(self):
        self.shortcuts = []

        def add(key, callback):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

        add("Space", self.toggle_contours)
        add("Backspace", self.select_all)
        add("Delete", self.select_all)
        add("Escape", self.close)
        for digit in range(10):
            add(str(digit), lambda d=digit: self.select_class_by_key(d))

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select RTDC File", self.rtdc_path, "RTDC Files (*.rtdc)"
        )
        if not path:
            return

        try:
            ds = dclab.new_dataset(path)
            if "image" not in ds:
                raise KeyError("No image feature in RTDC file")
            _ = ds["image"][0]
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        self.ds = ds
        self.rtdc_path = path
        self.show_contours = False

        self.load_classifiers()
        print(f"Opened RTDC: {path}")
        print(f"Number of images in RTDC: {len(ds):,}")

    def load_classifiers(self):
        self.classifier_combo.blockSignals(True)
        self.classifier_combo.clear()

        try:
            info = rtdc_toolbox.extract_rtdc_classifier_metadata(self.rtdc_path)
            self.classifiers = list(info.get("classifiers", []))
        except Exception:
            self.classifiers = []

        if not self.classifiers:
            self.classifiers = [{
                "classifier_name": "Legacy userdef1",
                "class_feature": LEGACY_CLASS_FEATURE,
                "probability_feature": (
                    LEGACY_CONFIDENCE_FEATURE
                    if LEGACY_CONFIDENCE_FEATURE in self.ds else None
                ),
                "class_map": {},
            }]

        for entry in self.classifiers:
            self.classifier_combo.addItem(entry["classifier_name"])

        self.classifier_combo.blockSignals(False)
        self.classifier_combo.setCurrentIndex(0)
        self.classifier_changed(0)

    def classifier_changed(self, index):
        if self.ds is None or index < 0:
            return

        entry = self.classifiers[index]
        self.class_feature = str(entry.get("class_feature") or "")
        self.confidence_feature = str(entry.get("probability_feature") or "")
        self.class_names = {
            int(k): str(v) for k, v in entry.get("class_map", {}).items()
        }

        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItem("All", None)

        if self.class_feature and self.class_feature in self.ds:
            self.class_list = np.asarray(
                np.unique(self.ds[self.class_feature][:]), dtype=int
            )
            for class_id in self.class_list:
                class_id = int(class_id)
                self.class_combo.addItem(
                    self.class_names.get(class_id, str(class_id)), class_id
                )
        else:
            self.class_list = np.array([], dtype=int)

        self.class_combo.setCurrentIndex(0)
        self.class_combo.blockSignals(False)
        self.active_class = None
        self.apply_filter()
        self.update_title()

    def class_changed(self, index):
        if self.ds is None or index < 0:
            return
        value = self.class_combo.itemData(index)
        self.active_class = None if value is None else int(value)
        self.apply_filter()
        self.update_title()

    def apply_filter(self):
        if self.active_class is None or self.class_feature not in self.ds:
            self.img_table = list(range(len(self.ds)))
        else:
            values = np.asarray(self.ds[self.class_feature][:])
            self.img_table = np.where(values == self.active_class)[0].tolist()

        max_page = max(0, (len(self.img_table) - 1) // N_PER_PAGE)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max_page)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.update_display(0)

    def update_display(self, page):
        if self.ds is None:
            return

        start = page * N_PER_PAGE
        for i, tile in enumerate(self.tiles):
            table_idx = start + i
            if table_idx >= len(self.img_table):
                tile.clear_event()
                continue

            event_idx = self.img_table[table_idx]
            image = make_event_image(
                self.ds,
                event_idx,
                show_contour=self.show_contours,
                class_feature=self.class_feature,
                confidence_feature=self.confidence_feature,
                class_names=self.class_names,
            )
            tile.set_event(event_idx, image)

        self.page_label.setText(f"{page} / {self.slider.maximum()}")

    def toggle_contours(self):
        if self.ds is not None and "contour" in self.ds:
            self.show_contours = not self.show_contours
            self.update_display(self.slider.value())
            self.update_title()

    def select_all(self):
        if self.ds is not None:
            self.class_combo.setCurrentIndex(0)

    def select_class_by_key(self, digit):
        if self.class_list.size == 0:
            return

        if 0 in self.class_list:
            if digit >= len(self.class_list):
                return
            class_id = int(self.class_list[digit])
        else:
            if digit < 1 or digit > len(self.class_list):
                return
            class_id = int(self.class_list[digit - 1])

        index = self.class_combo.findData(class_id)
        if index >= 0:
            self.class_combo.setCurrentIndex(index)

    def inspect_event(self, event_idx):
        """Show image/background and stream scalar feature values as they load."""
        shown_idx = get_rtdc_index(self.ds, event_idx)

        if self.inspect_dialog is not None:
            try:
                self.inspect_dialog.close()
            except RuntimeError:
                pass

        dialog = QDialog(self)
        self.inspect_dialog = dialog
        dialog.setWindowTitle(f"RTDC event {shown_idx}")
        dialog.resize(1200, 820)

        layout = QHBoxLayout(dialog)
        image_layout = QVBoxLayout()

        image_label = QLabel("Loading image...")
        image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(image_label)

        bg_label = QLabel("Loading background...")
        bg_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(bg_label)

        layout.addLayout(image_layout, 1)

        feature_text = QPlainTextEdit()
        feature_text.setReadOnly(True)
        feature_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        feature_text.setFont(QFont("Consolas", 9))
        feature_text.setPlainText(
            f"RTDC file: {self.rtdc_path}\n"
            f"Number of events: {len(self.ds):,}\n"
            f"Event: {shown_idx}\n\n"
        )
        layout.addWidget(feature_text, 2)

        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        QApplication.processEvents()

        def append_text(text):
            feature_text.moveCursor(QTextCursor.End)
            feature_text.insertPlainText(text)
            feature_text.ensureCursorVisible()
            QApplication.processEvents()

        # Load image.
        append_text("Loading image... ")
        try:
            image = qimage_from_array(self.ds["image"][event_idx])
            image_label.setPixmap(QPixmap.fromImage(image).scaled(
                image.width() * 1,
                image.height() * 1,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            ))
            append_text("done\n")
        except Exception as exc:
            image_label.setText(f"Could not load image:\n{exc}")
            append_text(f"ERROR ({exc})\n")

        # Load background image.
        append_text("Loading image_bg... ")
        try:
            bg = qimage_from_array(self.ds["image_bg"][event_idx])
            bg_label.setPixmap(QPixmap.fromImage(bg).scaled(
                bg.width() * 1,
                bg.height() * 1,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            ))
            append_text("done\n")
        except Exception as exc:
            bg_label.setText(f"Could not load image_bg:\n{exc}")
            append_text(f"ERROR ({exc})\n")

        # Choose which features to show.
        if len(SHOW_FEATURES) == 1 and str(SHOW_FEATURES[0]).lower() == "all":
            append_text("\nGetting ds.features_scalar... ")
            try:
                features_to_show = sorted(self.ds.features_scalar)
                append_text(f"{len(features_to_show)} features\n\n")
            except Exception as exc:
                append_text(f"ERROR ({exc})\n")
                return
        else:
            features_to_show = list(SHOW_FEATURES)
            append_text(
                f"\nUsing SHOW_FEATURES: {len(features_to_show)} features\n\n"
            )

        append_text("FEATURES\n" + "=" * 80 + "\n")

        for feature in features_to_show:
            # Print the name before reading the value. If a feature is slow,
            # the user can see exactly which read is currently in progress.
            append_text(f"{feature:30s} : ")

            try:
                value = np.asarray(self.ds[feature][event_idx])
                if value.ndim == 0:
                    value = value.item()
                append_text(f"{value}\n")
            except Exception as exc:
                append_text(f"ERROR ({exc})\n")

    def update_title(self):
        if not self.rtdc_path:
            return
        class_name = "All" if self.active_class is None else self.class_names.get(
            self.active_class, str(self.active_class)
        )
        self.setWindowTitle(
            f"RTDC Viewer Qt | {os.path.basename(self.rtdc_path)} | "
            f"Classifier: {self.classifier_combo.currentText()} | "
            f"Class: {class_name} | "
            f"Contours: {'ON' if self.show_contours else 'OFF'}"
        )


def main():
    app = QApplication(sys.argv)
    viewer = RTDCViewerQt()
    viewer.showMaximized()
    viewer.open_file()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()