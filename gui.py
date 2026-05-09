import sys
import os
import threading
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFileDialog, QComboBox, QTextEdit, QProgressBar,
                             QMessageBox, QGroupBox)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from qwen3_asr import Qwen3ASR

class ASRWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)
    status = Signal(str)

    def __init__(self, asr, model_path, audio_path, device_id, params):
        super().__init__()
        self.asr = asr
        self.model_path = model_path
        self.audio_path = audio_path
        self.device_id = device_id
        self.params = params
        self.task = "transcribe" # "load" or "transcribe"

    def run(self):
        try:
            if self.task == "load":
                self.status.emit(f"Loading model on device {self.device_id}...")
                self.asr.set_gpu_device(self.device_id)
                if self.asr.load_model(self.model_path):
                    self.finished.emit({"task": "load", "success": True})
                else:
                    self.error.emit("Failed to load model")
            else:
                self.status.emit("Transcribing...")
                result = self.asr.transcribe(self.audio_path, self.params)
                self.finished.emit({"task": "transcribe", "result": result})
        except Exception as e:
            self.error.emit(str(e))

class Qwen3ASRGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qwen3 ASR Desktop")
        self.resize(800, 600)
        
        # Initialize ASR
        try:
            self.asr = Qwen3ASR()
        except Exception as e:
            QMessageBox.critical(self, "Library Error", f"Could not load shared library: {e}")
            sys.exit(1)

        self.setup_ui()
        self.refresh_devices()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Model and Device Section ---
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)
        
        # Model path
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model Path:"))
        self.model_edit = QLineEdit("models/qwen3-asr-0.6b-f16.gguf")
        model_layout.addWidget(self.model_edit)
        model_browse_btn = QPushButton("Browse")
        model_browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(model_browse_btn)
        config_layout.addLayout(model_layout)

        # Device selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Compute Device:"))
        self.device_combo = QComboBox()
        device_layout.addWidget(self.device_combo, 1)
        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self.load_model)
        device_layout.addWidget(self.load_btn)
        config_layout.addLayout(device_layout)
        
        main_layout.addWidget(config_group)

        # --- Transcription Section ---
        trans_group = QGroupBox("Transcription")
        trans_layout = QVBoxLayout(trans_group)

        # Audio file
        audio_layout = QHBoxLayout()
        audio_layout.addWidget(QLabel("Audio File:"))
        self.audio_edit = QLineEdit("sample.wav")
        audio_layout.addWidget(self.audio_edit)
        audio_browse_btn = QPushButton("Browse")
        audio_browse_btn.clicked.connect(self.browse_audio)
        audio_layout.addWidget(audio_browse_btn)
        trans_layout.addLayout(audio_layout)

        # Transcription button
        self.transcribe_btn = QPushButton("Start Transcription")
        self.transcribe_btn.setEnabled(False)
        self.transcribe_btn.setFixedHeight(40)
        self.transcribe_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        self.transcribe_btn.clicked.connect(self.start_transcription)
        trans_layout.addWidget(self.transcribe_btn)

        main_layout.addWidget(trans_group)

        # --- Results Section ---
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("Transcription result will appear here...")
        results_layout.addWidget(self.result_text)
        
        # Stats
        self.stats_label = QLabel("Ready")
        results_layout.addWidget(self.stats_label)
        
        main_layout.addWidget(results_group)

        # Status Bar
        self.statusBar().showMessage("Ready")

    def refresh_devices(self):
        devices = self.asr.get_devices()
        for dev in devices:
            self.device_combo.addItem(dev['description'], dev['index'])

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select GGUF Model", "models", "GGUF Files (*.gguf)")
        if path:
            self.model_edit.setText(path)

    def browse_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", ".", "Audio Files (*.wav *.mp3 *.flac)")
        if path:
            self.audio_edit.setText(path)

    def load_model(self):
        model_path = self.model_edit.text()
        device_id = self.device_combo.currentData()
        
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "File Not Found", f"Model file not found: {model_path}")
            return

        self.load_btn.setEnabled(False)
        self.statusBar().showMessage("Loading model...")
        
        self.worker = ASRWorker(self.asr, model_path, None, device_id, None)
        self.worker.task = "load"
        self.worker.finished.connect(self.on_model_loaded)
        self.worker.error.connect(self.on_error)
        self.worker.status.connect(lambda s: self.statusBar().showMessage(s))
        self.worker.start()

    def on_model_loaded(self, data):
        self.load_btn.setEnabled(True)
        self.transcribe_btn.setEnabled(True)
        self.statusBar().showMessage("Model loaded successfully")
        QMessageBox.information(self, "Success", "Model loaded successfully!")

    def start_transcription(self):
        audio_path = self.audio_edit.text()
        if not os.path.exists(audio_path):
            QMessageBox.warning(self, "File Not Found", f"Audio file not found: {audio_path}")
            return

        self.transcribe_btn.setEnabled(False)
        self.result_text.clear()
        
        params = self.asr.get_default_params()
        # You could add UI for these params later
        
        self.worker = ASRWorker(self.asr, None, audio_path, None, params)
        self.worker.task = "transcribe"
        self.worker.finished.connect(self.on_transcription_finished)
        self.worker.error.connect(self.on_error)
        self.worker.status.connect(lambda s: self.statusBar().showMessage(s))
        self.worker.start()

    def on_transcription_finished(self, data):
        self.transcribe_btn.setEnabled(True)
        result = data["result"]
        
        if result["success"]:
            self.result_text.setPlainText(result["text"])
            self.stats_label.setText(f"Done in {result['time_ms']}ms | Tokens: {len(result['tokens'])}")
            self.statusBar().showMessage("Transcription completed")
        else:
            self.on_error(result["error"])

    def on_error(self, message):
        self.load_btn.setEnabled(True)
        self.transcribe_btn.setEnabled(True)
        self.statusBar().showMessage("Error occurred")
        QMessageBox.critical(self, "Error", f"An error occurred: {message}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Qwen3ASRGui()
    window.show()
    sys.exit(app.exec())
