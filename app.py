import os
import sys
import torch
import torchvision
import pydicom
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import threading
import time
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout,
    QPushButton, QLabel, QSlider, QHBoxLayout, QScrollArea, QProgressBar,
    QFrame, QSplitter, QTextEdit, QGroupBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QTabWidget, QStatusBar, QMessageBox, QGridLayout,
    QListWidget, QListWidgetItem, QToolButton, QButtonGroup, QSizePolicy
)
from PySide6.QtCore import (
    Qt, QPropertyAnimation, QRect, QTimer, QThread, 
    QEasingCurve, QParallelAnimationGroup, QSequentialAnimationGroup,
    QPoint, QSize, QObject, Signal
)
from PySide6.QtGui import (
    QPalette, QColor, QPixmap, QFont, QIcon, QPainter, QBrush, 
    QGradient, QLinearGradient, QPen, QMovie
)
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QGraphicsOpacityEffect
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# Configuration
MODEL_PATH = "tumor_epoch10.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESH = 0.5

class ModelLoader(QThread):
    """Threaded model loading"""
    model_loaded = Signal(object)
    progress_updated = Signal(int)
    
    def run(self):
        self.progress_updated.emit(20)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        self.progress_updated.emit(40)
        num_classes = 2
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        self.progress_updated.emit(70)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE).eval()
        self.progress_updated.emit(100)
        self.model_loaded.emit(model)

class ImageProcessor(QThread):
    """Threaded image processing"""
    image_processed = Signal(object, int, int)
    progress_updated = Signal(int)
    
    def __init__(self, files, model):
        super().__init__()
        self.files = files
        self.model = model
        
    def run(self):
        images = []
        total = len(self.files)
        
        for i, (path, typ) in enumerate(self.files):
            try:
                if typ == "dicom":
                    ds = pydicom.dcmread(path)
                    img = self.dicom_to_image(ds)
                else:
                    img = Image.open(path).convert("RGB")
                
                # Run inference
                img = self.run_inference(img)
                images.append((img, path))
                
                progress = int((i + 1) / total * 100)
                self.progress_updated.emit(progress)
                
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
                
        self.image_processed.emit(images, total, len(images))
    
    def dicom_to_image(self, ds):
        arr = ds.pixel_array.astype(np.float32)
        # Window/Level adjustment
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            center = float(ds.WindowCenter[0]) if isinstance(ds.WindowCenter, list) else float(ds.WindowCenter)
            width = float(ds.WindowWidth[0]) if isinstance(ds.WindowWidth, list) else float(ds.WindowWidth)
            
            lower = center - width / 2
            upper = center + width / 2
            arr = np.clip(arr, lower, upper)
            arr = (arr - lower) / (upper - lower) * 255
        else:
            arr -= np.min(arr)
            arr = arr / np.max(arr) * 255 if np.max(arr) > 0 else arr
            
        arr = arr.astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
        return img
    
    def run_inference(self, image):
        transform = torchvision.transforms.ToTensor()
        img_tensor = transform(image).to(DEVICE)
        
        with torch.no_grad():
            pred = self.model([img_tensor])[0]
            
        boxes = pred["boxes"].cpu()
        scores = pred["scores"].cpu()
        labels = pred["labels"].cpu()
        
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            
        detection_count = 0
        for box, score, label in zip(boxes, scores, labels):
            if score < CONF_THRESH:
                continue
                
            detection_count += 1
            x1, y1, x2, y2 = box
            
            # Futuristic glow effect
            for i in range(3):
                draw.rectangle([x1-i, y1-i, x2+i, y2+i], 
                             outline=f"rgba(0, 255, 255, {255-i*50})", width=2)
            
            draw.rectangle([x1, y1, x2, y2], outline="#00FFFF", width=2)
            draw.text((x1, y1 - 20), f"Tumor: {score:.2f}", fill="#00FFFF", font=font)
            
        return image

class FuturisticImageViewer(FigureCanvas):
    """Advanced matplotlib image viewer with animations"""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 8), facecolor='#0a0a0a')
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111, facecolor='#0a0a0a')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self.images = []
        self.index = 0
        self.brightness = 1.0
        self.contrast = 1.0
        
        # Animation setup
        self.mpl_connect('scroll_event', self.on_scroll)
        self.mpl_connect('button_press_event', self.on_click)
        self.mpl_connect('motion_notify_event', self.on_drag)
        
        self.dragging = False
        self.last_pos = None
        
    def set_images(self, images):
        self.images = images
        self.index = 0
        if images:
            self.update_view()
    
    def update_view(self):
        if not self.images:
            return
            
        self.ax.clear()
        self.ax.set_facecolor('#0a0a0a')
        
        img_array = np.array(self.images[self.index][0])
        
        # Apply brightness/contrast
        img_array = img_array.astype(np.float32)
        img_array = img_array * self.contrast + (self.brightness - 1) * 127
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        self.im = self.ax.imshow(img_array, cmap='gray')
        self.ax.set_title(f"Slice {self.index+1}/{len(self.images)} - {os.path.basename(self.images[self.index][1])}", 
                         color='#00FFFF', fontsize=12, fontweight='bold')
        
        # Add crosshairs for medical viewing
        height, width = img_array.shape[:2]
        self.ax.axhline(y=height//2, color='#FF00FF', alpha=0.3, linewidth=1)
        self.ax.axvline(x=width//2, color='#FF00FF', alpha=0.3, linewidth=1)
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()
        # self.draw()
        QTimer.singleShot(0, self.draw_idle)
    
    def on_scroll(self, event):
        if not self.images:
            return
            
        change = 1 if event.button == 'up' else -1
        self.index = max(0, min(self.index + change, len(self.images) - 1))
        self.update_view()
    
    def on_click(self, event):
        if event.button == 1:  # Left click
            self.dragging = True
            self.last_pos = (event.xdata, event.ydata)
    
    def on_drag(self, event):
        if self.dragging and self.last_pos and event.xdata and event.ydata:
            dx = event.xdata - self.last_pos[0]
            dy = event.ydata - self.last_pos[1]
            
            # Window/Level adjustment
            self.brightness += dy * 0.001
            self.contrast += dx * 0.001
            
            self.brightness = max(0.1, min(3.0, self.brightness))
            self.contrast = max(0.1, min(3.0, self.contrast))
            
            self.update_view()
            self.last_pos = (event.xdata, event.ydata)
    
    def set_slice(self, index):
        if 0 <= index < len(self.images):
            self.index = index
            self.update_view()
from PySide6.QtWidgets import (
    QPushButton, QGraphicsDropShadowEffect, QFrame, QVBoxLayout, QGroupBox, QLabel,
    QSlider, QDoubleSpinBox, QGridLayout, QTextEdit
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect
from PySide6.QtGui import QColor


class AnimatedButton(QPushButton):
    """Custom animated button with cyan glow and smooth hover animation"""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setup_style()
        self.setup_glow()
        self.setup_animation()
        self.setMinimumHeight(45)
        self.setCursor(Qt.PointingHandCursor)

    def setup_style(self):
        self.setStyleSheet("""
            QPushButton {
                background-color: #13293D;
                border: 2px solid #00FFFF;
                border-radius: 12px;
                color: #00FFFF;
                font-weight: bold;
                font-size: 14px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #1A3A55;
                border: 2px solid #00FFAA;
                color: #00FFAA;
            }
            QPushButton:pressed {
                background-color: #0A1F30;
                border: 2px solid #0088FF;
            }
        """)

    def setup_glow(self):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(0, 255, 255, 160))
        shadow.setOffset(0, 0)
        self.setGraphicsEffect(shadow)

    def setup_animation(self):
        self.hover_in = QPropertyAnimation(self, b"geometry")
        self.hover_in.setDuration(160)
        self.hover_in.setEasingCurve(QEasingCurve.OutCubic)

        self.hover_out = QPropertyAnimation(self, b"geometry")
        self.hover_out.setDuration(160)
        self.hover_out.setEasingCurve(QEasingCurve.InCubic)

    def enterEvent(self, event):
        rect = self.geometry()
        self.hover_in.setStartValue(rect)
        self.hover_in.setEndValue(QRect(rect.x() - 2, rect.y() - 2,
                                        rect.width() + 4, rect.height() + 4))
        self.hover_in.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        rect = self.geometry()
        self.hover_out.setStartValue(rect)
        self.hover_out.setEndValue(QRect(rect.x() + 2, rect.y() + 2,
                                         rect.width() - 4, rect.height() - 4))
        self.hover_out.start()
        super().leaveEvent(event)


class FuturisticProgressBar(QProgressBar):
    """Animated progress bar with glow effects"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_style()
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self.pulse_effect)
        
    def setup_style(self):
        self.setStyleSheet("""
            QProgressBar {
                border: 2px solid #00FFFF;
                border-radius: 10px;
                background-color: #1a1a1a;
                text-align: center;
                color: #00FFFF;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #00FFFF, stop:0.5 #0099FF, stop:1 #00FFFF);
                border-radius: 8px;
                margin: 2px;
            }
        """)
        
        # Add glow effect
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(15)
        self.shadow.setColor(QColor(0, 255, 255, 150))
        self.shadow.setOffset(0, 0)
        self.setGraphicsEffect(self.shadow)
    
    def pulse_effect(self):
        if self.isVisible():
            self.shadow.setBlurRadius(15 + np.sin(time.time() * 5) * 5)

class AetherByteMainWindow(QMainWindow):
    """Main application window with advanced features"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.images = []
        self.current_folder = None
        
         # Status tracking
        self.session_stats = {
            'images_processed': 0,
            'detections_found': 0,
            'session_start': datetime.now()
        }
    
        
        self.setup_ui()
        self.setup_style()
        self.setup_animations()
        self.load_model()
        
       
    def setup_ui(self):
        self.setWindowTitle("Tumor Detection System")
        self.setGeometry(100, 50, 1400, 900)
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls
        self.setup_control_panel(splitter)
        
        # Right panel - Viewer
        self.setup_viewer_panel(splitter)
        
        # Status bar
        self.setup_status_bar()
        
        # Set splitter proportions
        splitter.setSizes([350, 1050])
    
    def setup_control_panel(self, parent):
        """Setup the left control panel"""
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.StyledPanel)
        control_frame.setMaximumWidth(400)
        control_frame.setMinimumWidth(300)
        
        layout = QVBoxLayout(control_frame)
        
        # Title with animation
        title = QLabel("Tumora")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: #00FFFF;
                font-size: 24px;
                font-weight: bold;
                padding: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #1a1a2e, stop:1 #16213e);
                border-radius: 15px;
                margin: 10px;
            }
        """)
        layout.addWidget(title)
        
        # File operations group
        file_group = QGroupBox("📁 File Operations")
        file_group.setStyleSheet(self.get_group_style())
        file_layout = QVBoxLayout(file_group)
        
        self.btn_load = QPushButton("🔍 Browse Folder")
        # self.btn_load = AnimatedButton("🔍 Browse Folder")
        self.btn_load.clicked.connect(self.load_folder)
        file_layout.addWidget(self.btn_load)
        
        self.btn_close = QPushButton("❌ Close Files")
        # self.btn_close = AnimatedButton("❌ Close Files")
        self.btn_close.clicked.connect(self.close_files)
        self.btn_close.setEnabled(False)
        file_layout.addWidget(self.btn_close)
        
        self.btn_export = QPushButton("💾 Export Results")
        # self.btn_export = AnimatedButton("💾 Export Results")
        self.btn_export.clicked.connect(self.export_results)
        self.btn_export.setEnabled(False)
        file_layout.addWidget(self.btn_export)
        
        layout.addWidget(file_group)
        
        # Model settings group
        model_group = QGroupBox("🧠 AI Model Settings")
        model_group.setStyleSheet(self.get_group_style())
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("Confidence Threshold:"), 0, 0)
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.1, 1.0)
        self.conf_spinbox.setSingleStep(0.05)
        self.conf_spinbox.setValue(CONF_THRESH)
        self.conf_spinbox.valueChanged.connect(self.update_confidence)
        model_layout.addWidget(self.conf_spinbox, 0, 1)
        
        model_layout.addWidget(QLabel("Device:"), 1, 0)
        device_label = QLabel(f"{'🔥 CUDA' if DEVICE.type == 'cuda' else '💻 CPU'}")
        device_label.setStyleSheet("color: #00FF00; font-weight: bold;")
        model_layout.addWidget(device_label, 1, 1)
        
        layout.addWidget(model_group)
        
        # Image controls group
        img_group = QGroupBox("🖼️ Image Controls")
        img_group.setStyleSheet(self.get_group_style())
        img_layout = QGridLayout(img_group)
        
        # Slice navigation
        img_layout.addWidget(QLabel("Slice:"), 0, 0)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.valueChanged.connect(self.change_slice)
        img_layout.addWidget(self.slice_slider, 0, 1)
        
        self.slice_label = QLabel("0/0")
        self.slice_label.setStyleSheet("color: #00FFFF; font-weight: bold;")
        img_layout.addWidget(self.slice_label, 0, 2)
        
        # Brightness control
        img_layout.addWidget(QLabel("Brightness:"), 1, 0)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(10, 300)
        self.brightness_slider.setValue(100)
        self.brightness_slider.valueChanged.connect(self.adjust_brightness)
        img_layout.addWidget(self.brightness_slider, 1, 1)
        
        # Contrast control
        img_layout.addWidget(QLabel("Contrast:"), 2, 0)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.adjust_contrast)
        img_layout.addWidget(self.contrast_slider, 2, 1)
        
        layout.addWidget(img_group)
        
        # Statistics group
        stats_group = QGroupBox("📊 Statistics")
        stats_group.setStyleSheet(self.get_group_style())
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setStyleSheet("""
            QTextEdit {
                background-color: #0a0a0a;
                color: #00FFFF;
                border: 1px solid #00FFFF;
                border-radius: 5px;
                font-family: 'Courier New';
                font-size: 11px;
            }
        """)
        self.update_statistics()
        stats_layout.addWidget(self.stats_text)
        
        layout.addWidget(stats_group)
        
        # Progress bar
        self.progress_bar = FuturisticProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        parent.addWidget(control_frame)
    
    def setup_viewer_panel(self, parent):
        """Setup the right viewer panel"""
        viewer_frame = QFrame()
        viewer_frame.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(viewer_frame)
        
        # Viewer area
        self.viewer = FuturisticImageViewer()
        self.viewer.setStyleSheet("""
            QWidget {
                background-color: #0a0a0a;
                border: 2px solid #00FFFF;
                border-radius: 10px;
            }
        """)
        layout.addWidget(self.viewer)
        
        parent.addWidget(viewer_frame)
    
    def setup_status_bar(self):
        """Setup status bar with live information"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #1a1a2e;
                color: #00FFFF;
                border-top: 1px solid #00FFFF;
                font-weight: bold;
            }
        """)
        
        # Update status regularly
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # Update every second
    
    def get_group_style(self):
        """Get consistent group box styling"""
        return """
            QGroupBox {
                font-weight: bold;
                border: 2px solid #00FFFF;
                border-radius: 10px;
                margin-top: 10px;
                color: #00FFFF;
                font-size: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background-color: #1a1a2e;
            }
            QLabel {
                color: #FFFFFF;
                font-weight: normal;
            }
            QSlider::groove:horizontal {
                border: 1px solid #00FFFF;
                height: 8px;
                background: #1a1a1a;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00FFFF;
                border: 1px solid #00FFFF;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #1a1a1a;
                border: 1px solid #00FFFF;
                border-radius: 5px;
                color: #FFFFFF;
                padding: 5px;
            }
        """
    
    def setup_style(self):
        """Setup the main application styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f0f23;
                color: #FFFFFF;
            }
            QFrame {
                background-color: #1a1a2e;
                border-radius: 10px;
            }
        """)
        
        # Apply dark theme globally
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(15, 15, 35))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(26, 26, 46))
        palette.setColor(QPalette.AlternateBase, QColor(35, 35, 55))
        palette.setColor(QPalette.Button, QColor(45, 74, 102))
        palette.setColor(QPalette.ButtonText, QColor(0, 255, 255))
        self.setPalette(palette)
    
    def setup_animations(self):
        """Setup UI animations"""
        # Window entrance animation
        self.entrance_anim = QPropertyAnimation(self, b"geometry")
        self.entrance_anim.setDuration(800)
        self.entrance_anim.setEasingCurve(QEasingCurve.OutBounce)
        
        # Fade in animation
        self.opacity_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)
        
        self.fade_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_anim.setDuration(1000)
        self.fade_anim.setStartValue(0.0)
        self.fade_anim.setEndValue(1.0)
        self.fade_anim.start()
    
    def load_model(self):
        """Load the AI model in a separate thread"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.pulse_timer.start(50)
        
        self.model_loader = ModelLoader()
        self.model_loader.model_loaded.connect(self.on_model_loaded)
        self.model_loader.progress_updated.connect(self.progress_bar.setValue)
        self.model_loader.start()
    
    def on_model_loaded(self, model):
        """Handle model loading completion"""
        self.model = model
        self.progress_bar.setVisible(False)
        self.progress_bar.pulse_timer.stop()
        self.status_bar.showMessage("✅ AI Model loaded successfully!", 3000)
    
    def load_folder(self):
        """Load images from a folder"""
        if not self.model:
            QMessageBox.warning(self, "Warning", "Please wait for model to load first!")
            return
            
        folder = QFileDialog.getExistingDirectory(self, "Select Image/DICOM Folder")
        if not folder:
            return
            
        self.current_folder = folder
        
        # Find supported files
        files = []
        for filename in os.listdir(folder):
            ext = os.path.splitext(filename)[1].lower()
            full_path = os.path.join(folder, filename)
            
            if ext in [".dcm", ".dicom"]:
                files.append((full_path, "dicom"))
            elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                files.append((full_path, "image"))
        
        if not files:
            QMessageBox.information(self, "Info", "No supported image files found!")
            return
        
        files.sort()
        
        # Process images in separate thread
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.pulse_timer.start(50)
        
        self.image_processor = ImageProcessor(files, self.model)
        self.image_processor.image_processed.connect(self.on_images_processed)
        self.image_processor.progress_updated.connect(self.progress_bar.setValue)
        self.image_processor.start()
        
        self.btn_load.setEnabled(False)
        self.status_bar.showMessage(f"Processing {len(files)} files...")
    
    def on_images_processed(self, images, total_files, processed_files):
        """Handle image processing completion"""
        self.images = images
        self.progress_bar.setVisible(False)
        self.progress_bar.pulse_timer.stop()
        
        if images:
            self.viewer.set_images(images)
            self.slice_slider.setMaximum(len(images) - 1)
            self.slice_slider.setValue(0)
            self.update_slice_label()
            
            self.btn_close.setEnabled(True)
            self.btn_export.setEnabled(True)
            
            # Update statistics
            self.session_stats['images_processed'] = processed_files
            self.update_statistics()
            
            self.status_bar.showMessage(f"✅ Loaded {processed_files}/{total_files} images", 5000)
        else:
            QMessageBox.warning(self, "Warning", "No images could be processed!")
            
        self.btn_load.setEnabled(True)
    
    def close_files(self):
        """Close all loaded files"""
        self.images = []
        self.viewer.set_images([])
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.update_slice_label()
        
        self.btn_close.setEnabled(False)
        self.btn_export.setEnabled(False)
        
        self.status_bar.showMessage("Files closed", 2000)
    
    def export_results(self):
        """Export processing results"""
        if not self.images:
            return
            
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder:
            return
            
        # Export images with detections
        for i, (img, original_path) in enumerate(self.images):
            filename = f"detected_{i:03d}_{os.path.basename(original_path)}.png"
            export_path = os.path.join(folder, filename)
            img.save(export_path)
        
        # Export session report
        report = {
            'session_start': self.session_stats['session_start'].isoformat(),
            'images_processed': len(self.images),
            'folder_path': self.current_folder,
            'model_confidence': self.conf_spinbox.value(),
            'device_used': str(DEVICE)
        }
        
        report_path = os.path.join(folder, "detection_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        QMessageBox.information(self, "Export Complete", 
                               f"Results exported to:\n{folder}")
    
    def change_slice(self, index):
        """Change the displayed slice"""
        if self.images:
            self.viewer.set_slice(index)
            self.update_slice_label()
    
    def update_slice_label(self):
        """Update slice counter label"""
        current = self.slice_slider.value() + 1 if self.images else 0
        total = len(self.images)
        self.slice_label.setText(f"{current}/{total}")
    
    def adjust_brightness(self, value):
        """Adjust image brightness"""
        self.viewer.brightness = value / 100.0
        self.viewer.update_view()
    
    def adjust_contrast(self, value):
        """Adjust image contrast"""
        self.viewer.contrast = value / 100.0
        self.viewer.update_view()
    
    def update_confidence(self, value):
        """Update confidence threshold"""
        global CONF_THRESH
        CONF_THRESH = value
        # Re-process current images if any
        if self.images and self.model:
            # This would require re-running inference
            pass
    
    def update_statistics(self):
        """Update statistics display"""
        uptime = datetime.now() - self.session_stats['session_start']
        
        stats_text = f"""
═══════════════════════════════════
🚀 SESSION STATS
═══════════════════════════════════
⏱️  Session Time: {str(uptime).split('.')[0]}
📁  Images Loaded: {len(self.images)}
🎯  Current Slice: {self.slice_slider.value() + 1 if self.images else 0}
💻  Device: {DEVICE.type.upper()}
🔍  Confidence: {self.conf_spinbox.value():.2f}
═══════════════════════════════════
        """.strip()
        
        self.stats_text.setPlainText(stats_text)
    
    def update_status(self):
        """Update status bar with live information"""
        if self.images:
            current_slice = self.slice_slider.value() + 1
            total_slices = len(self.images)
            brightness = self.brightness_slider.value()
            contrast = self.contrast_slider.value()
            
            status = f"Slice: {current_slice}/{total_slices} | Brightness: {brightness}% | Contrast: {contrast}% | Device: {DEVICE.type.upper()}"
        else:
            status = f"Ready | Device: {DEVICE.type.upper()} | Tumora 0.0.1"
            
        self.status_bar.showMessage(status)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_Left and self.images:
            new_index = max(0, self.slice_slider.value() - 1)
            self.slice_slider.setValue(new_index)
        elif event.key() == Qt.Key_Right and self.images:
            new_index = min(len(self.images) - 1, self.slice_slider.value() + 1)
            self.slice_slider.setValue(new_index)
        elif event.key() == Qt.Key_Space and self.btn_load.isEnabled():
            self.load_folder()
        elif event.key() == Qt.Key_Escape:
            if self.images:
                self.close_files()
        
        super().keyPressEvent(event)

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("AetherByte Tumor Detection")
    app.setApplicationVersion("2.0")
    
    # Set application icon (if available)
    try:
        app.setWindowIcon(QIcon("icon.png"))
    except:
        pass
    
    # Apply global dark theme
    app.setStyle("Fusion")
    
    window = AetherByteMainWindow()
    window.showMaximized()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()