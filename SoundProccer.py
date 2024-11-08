import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QWidget, QFileDialog, QMessageBox, QSlider, QLabel,
                             QHBoxLayout, QProgressBar, QFrame, QSplitter)
from PySide6.QtCore import QTimer
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import wave
import struct
import sys
from scipy.io import wavfile
import sounddevice as sd
import time
import threading
from PySide6.QtGui import QPalette, QColor, QFont, QIcon
from PySide6.QtCore import Qt, QSize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Agregar esta función antes de la clase MainWindow
def filtrar_ruido(y, sr, umbral=5000):
    if y is None:
        return None
        
    # Aplicar FFT para obtener la representación en frecuencia
    fft_y = fft(y)
    frecuencias = fftfreq(len(y), 1/sr)
    
    # Filtrar frecuencias altas (ruido)
    mascara = np.abs(frecuencias) > umbral
    fft_y[mascara] = 0
    
    # Aplicar la inversa de Fourier
    y_filtrado = ifft(fft_y).real
    return y_filtrado

def sintetizar_sonido(frecuencia, duracion=2, sr=44100):
    t = np.linspace(0, duracion, int(sr * duracion), endpoint=False)
    y_sintetizado = 0.5 * np.sin(2 * np.pi * frecuencia * t)
    return y_sintetizado, sr

def comprimir_audio(y, porcentaje=50):
    if y is None:
        return None
        
    fft_y = fft(y)
    umbral = int(len(fft_y) * porcentaje / 100)
    fft_y[umbral:] = 0  # Mantener solo el porcentaje seleccionado de componentes
    y_comprimido = ifft(fft_y).real
    return y_comprimido

class AudioProcessor:
    def __init__(self):
        self.y = None
        self.sr = None

    def cargar_audio(self):
        try:
            archivo_audio, _ = QFileDialog.getOpenFileName(
                None, 
                "Seleccionar archivo de audio",
                "",
                "Audio Files (*.wav)"
            )
            if archivo_audio:
                try:
                    self.sr, data = wavfile.read(archivo_audio)
                    # Convertir a float y normalizar
                    self.y = data.astype(float)
                    if data.dtype == np.int16:
                        self.y /= 32768.0
                    elif data.dtype == np.int32:
                        self.y /= 2147483648.0
                    
                    # Si es estéreo, convertir a mono
                    if len(self.y.shape) > 1:
                        self.y = np.mean(self.y, axis=1)
                    
                    QMessageBox.information(None, "Cargar Audio", f"Audio cargado con éxito: {archivo_audio}")
                except Exception as e:
                    QMessageBox.critical(None, "Error", f"Error al cargar el archivo: {str(e)}")
            return self.y, self.sr
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error inesperado: {str(e)}")
            return None, None

# ... (mantener las funciones de procesamiento de audio sin cambios) ...

class AudioPlayer:
    def __init__(self):
        self.playing = False
        self.current_position = 0
        self.audio_data = None
        self.sr = None
        self.stream = None
        
    def play(self, data, sr):
        try:
            # Detener reproducción anterior si existe
            if self.stream is not None:
                self.stop()
            
            self.audio_data = data
            self.sr = sr
            self.playing = True
            self.current_position = 0
            self.stream = sd.OutputStream(samplerate=sr, channels=1, callback=self.callback)
            self.stream.start()
        except Exception as e:
            print(f"Error al iniciar reproducción: {e}")
            self.stop()
            
    def stop(self):
        self.playing = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error al detener stream: {e}")
            finally:
                self.stream = None
        self.current_position = 0
            
    def pause(self):
        self.playing = False
        if self.stream:
            try:
                self.stream.stop()
            except Exception as e:
                print(f"Error al pausar: {e}")
            
    def resume(self):
        if self.audio_data is not None and self.stream:
            try:
                self.playing = True
                self.stream.start()
            except Exception as e:
                print(f"Error al resumir: {e}")
                self.stop()

    def callback(self, outdata, frames, time, status):
        if self.playing and self.current_position < len(self.audio_data):
            chunk = self.audio_data[self.current_position:self.current_position + frames]
            if len(chunk) < frames:
                outdata[:len(chunk), 0] = chunk
                outdata[len(chunk):, 0] = 0
                self.playing = False
            else:
                outdata[:, 0] = chunk
            self.current_position += frames
        else:
            outdata.fill(0)
            self.playing = False
            
    def seek(self, position):
        self.current_position = int(position * len(self.audio_data))
        
    def get_progress(self):
        if self.audio_data is None:
            return 0
        return self.current_position / len(self.audio_data) * 100

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = AudioProcessor()
        self.audio_player = AudioPlayer()
        self.y_filtrado = None
        self.y_comprimido = None
        self.currently_playing = None

        self.setWindowTitle("Audio Processor")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FFFFFF;
            }
            QWidget {
                background-color: #FFFFFF;
                color: #333333;
            }
            QPushButton {
                background-color: #FFFFFF;
                border: 2px solid #333333;
                color: #333333;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 13px;
                font-weight: bold;
                margin: 5px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #333333;
                color: #FFFFFF;
            }
            QPushButton:pressed {
                background-color: #1a1a1a;
            }
            QProgressBar {
                border: 1px solid #333333;
                border-radius: 2px;
                text-align: center;
                background-color: #FFFFFF;
            }
            QProgressBar::chunk {
                background-color: #333333;
            }
            QSlider::groove:horizontal {
                border: 1px solid #333333;
                height: 4px;
                background: #FFFFFF;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #333333;
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QLabel {
                color: #333333;
                font-size: 13px;
            }
        """)

        # Widget central principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Panel izquierdo (controles)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(15)

        # Título
        title_label = QLabel("AUDIO PROCESSOR")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #333333;
            padding: 10px;
            border-bottom: 2px solid #333333;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title_label)

        # Controles de reproducción
        self.control_widget = QWidget()
        control_layout = QVBoxLayout(self.control_widget)
        
        # Botones de control
        playback_buttons = QHBoxLayout()
        self.play_pause_button = QPushButton()
        self.play_pause_button.setIcon(QIcon("icons/play.png"))
        self.play_pause_button.setFixedSize(40, 40)
        self.play_pause_button.setStyleSheet("""
            QPushButton {
                border-radius: 20px;
                padding: 5px;
            }
        """)
        playback_buttons.addWidget(self.play_pause_button)
        playback_buttons.addStretch()
        
        # Agregar barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #333333;
                border-radius: 2px;
                text-align: center;
                height: 8px;
                margin: 0px 5px;
            }
            QProgressBar::chunk {
                background-color: #333333;
            }
        """)
        self.progress_bar.setTextVisible(False)
        
        # Slider y tiempo
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 100)
        self.seek_slider.sliderPressed.connect(self.on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self.on_slider_released)
        self.seek_slider.sliderMoved.connect(self.on_slider_moved)
        self.slider_pressed = False
        
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setAlignment(Qt.AlignRight)
        
        control_layout.addLayout(playback_buttons)
        control_layout.addWidget(self.seek_slider)
        control_layout.addWidget(self.progress_bar)  # Agregar progress_bar
        control_layout.addWidget(self.time_label)
        
        left_layout.addWidget(self.control_widget)

        # Botones de funciones
        boton_cargar = QPushButton(" Cargar Audio")
        boton_cargar.setIcon(QIcon("icons/load.png"))
        boton_filtrado = QPushButton(" Filtrar Ruido")
        boton_filtrado.setIcon(QIcon("icons/filter.png"))
        boton_sintesis = QPushButton(" Sintetizar")
        boton_sintesis.setIcon(QIcon("icons/synth.png"))
        boton_compresion = QPushButton(" Comprimir")
        boton_compresion.setIcon(QIcon("icons/compress.png"))

        left_layout.addWidget(boton_cargar)
        left_layout.addWidget(boton_filtrado)
        left_layout.addWidget(boton_sintesis)
        left_layout.addWidget(boton_compresion)
        left_layout.addStretch()

        # Panel derecho (gráfica)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Crear figura de matplotlib
        self.figure = Figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        
        # Botón de guardar (inicialmente oculto)
        self.save_button = QPushButton(" Guardar")
        self.save_button.setIcon(QIcon("icons/save.png"))
        self.save_button.hide()
        right_layout.addWidget(self.save_button)

        # Agregar paneles al layout principal
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)

        # Conectar señales
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        boton_cargar.clicked.connect(self.cargar_audio)
        boton_filtrado.clicked.connect(self.aplicar_filtro)
        boton_sintesis.clicked.connect(self.generar_sintesis)
        boton_compresion.clicked.connect(self.aplicar_compresion)
        self.save_button.clicked.connect(self.guardar_audio_actual)

        # Timer para actualizar la barra de progreso
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(100)

    def plot_audio(self, data, title):
        """Método para graficar en el panel derecho"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(data, color='#333333')
        ax.set_title(title)
        ax.set_xlabel('Muestras')
        ax.set_ylabel('Amplitud')
        ax.grid(True, linestyle='--', alpha=0.7)
        self.canvas.draw()
        
        # Mostrar botón de guardar con el título correspondiente
        self.save_button.setText(f" Guardar {title}")
        self.save_button.show()

    def guardar_audio_actual(self):
        """Método para guardar el audio actual"""
        if hasattr(self, 'ultimo_audio_procesado'):
            self.preguntar_guardar(self.ultimo_audio_procesado, 
                                 self.save_button.text().replace(" Guardar ", ""))

    def toggle_play_pause(self):
        """Alternar entre reproducir y pausar"""
        if self.audio_player.playing:
            self.audio_player.pause()
            self.play_pause_button.setIcon(QIcon("icons/play.png"))
        else:
            self.audio_player.resume()
            self.play_pause_button.setIcon(QIcon("icons/pause.png"))

    def seek_audio(self, position):
        """Buscar en el audio"""
        self.audio_player.seek(position / 100.0)

    def on_slider_pressed(self):
        """Cuando el usuario presiona el slider"""
        self.slider_pressed = True
        if self.audio_player.playing:
            self.audio_player.pause()

    def on_slider_released(self):
        """Cuando el usuario suelta el slider"""
        self.slider_pressed = False
        position = self.seek_slider.value() / 100.0
        self.audio_player.seek(position)
        if self.audio_player.playing:
            self.audio_player.resume()

    def on_slider_moved(self, position):
        """Mientras el usuario mueve el slider"""
        if self.audio_player.audio_data is not None:
            total_time = len(self.audio_player.audio_data) / self.audio_player.sr
            current_time = position * total_time / 100
            self.time_label.setText(f"{int(current_time//60)}:{int(current_time%60):02d} / "
                                  f"{int(total_time//60)}:{int(total_time%60):02d}")

    def update_progress(self):
        """Actualizar la barra de progreso y el tiempo"""
        if self.audio_player.audio_data is not None and not self.slider_pressed:
            progress = self.audio_player.get_progress()
            self.progress_bar.setValue(int(progress))
            self.seek_slider.setValue(int(progress))
            
            current_time = self.audio_player.current_position / self.audio_player.sr
            total_time = len(self.audio_player.audio_data) / self.audio_player.sr
            self.time_label.setText(f"{int(current_time//60)}:{int(current_time%60):02d} / "
                                  f"{int(total_time//60)}:{int(total_time%60):02d}")

    def guardar_audio(self):
        if not hasattr(self, 'ultimo_audio_procesado') or self.ultimo_audio_procesado is None:
            QMessageBox.warning(self, "Advertencia", "No hay audio procesado para guardar")
            return
            
        archivo_guardar, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar Audio",
            "",
            "WAV Files (*.wav)"
        )
        
        if archivo_guardar:
            try:
                # Normalizar y convertir a int16
                audio_normalizado = np.int16(self.ultimo_audio_procesado * 32767)
                wavfile.write(archivo_guardar, self.processor.sr, audio_normalizado)
                QMessageBox.information(self, "Éxito", "Audio guardado correctamente")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al guardar el archivo: {str(e)}")

    def reproducir_audio(self, datos, sr):
        try:
            if datos is not None and sr is not None:
                # Detener reproducción actual si existe
                if self.audio_player.playing:
                    self.audio_player.stop()
                    self.play_pause_button.setText("Play")
                
                self.audio_player.play(datos, sr)
                self.play_pause_button.setText("Pause")
                self.currently_playing = datos
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al reproducir: {str(e)}")

    def reproducir_original(self):
        if self.processor.y is None:
            QMessageBox.critical(self, "Error", "Por favor, carga un archivo de audio primero")
            return
        self.reproducir_audio(self.processor.y, self.processor.sr)

    def reproducir_filtrado(self):
        if self.y_filtrado is None:
            QMessageBox.critical(self, "Error", "Por favor, aplica un filtro primero")
            return
        self.reproducir_audio(self.y_filtrado, self.processor.sr)

    def preguntar_guardar(self, audio_modificado, titulo):
        """Función auxiliar para preguntar si se quiere guardar el audio"""
        respuesta = QMessageBox.question(
            self,
            "Guardar Audio",
            "¿Deseas guardar el audio modificado?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if respuesta == QMessageBox.Yes:
            archivo_guardar, _ = QFileDialog.getSaveFileName(
                self,
                f"Guardar {titulo}",
                "",
                "WAV Files (*.wav)"
            )
            
            if archivo_guardar:
                try:
                    # Normalizar y convertir a int16
                    audio_normalizado = np.int16(audio_modificado * 32767)
                    wavfile.write(archivo_guardar, self.processor.sr, audio_normalizado)
                    QMessageBox.information(self, "Éxito", f"{titulo} guardado correctamente")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error al guardar el archivo: {str(e)}")

    def aplicar_filtro(self):
        try:
            if self.processor.y is None:
                QMessageBox.critical(self, "Error", "Por favor, carga un archivo de audio primero")
                return
            
            if self.audio_player.playing:
                self.audio_player.stop()
                self.play_pause_button.setIcon(QIcon("icons/play.png"))
                
            self.y_filtrado = filtrar_ruido(self.processor.y, self.processor.sr)
            if self.y_filtrado is not None:
                self.plot_audio(self.y_filtrado, "Audio Filtrado")
                self.reproducir_audio(self.y_filtrado, self.processor.sr)
                self.ultimo_audio_procesado = self.y_filtrado
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al filtrar: {str(e)}")

    def generar_sintesis(self):
        try:
            if self.audio_player.playing:
                self.audio_player.stop()
                self.play_pause_button.setText("Play")
                
            y_sintetizado, sr_sintetizado = sintetizar_sonido(frecuencia=440)
            plt.plot(y_sintetizado)
            plt.title("Sonido Sintetizado (440Hz)")
            plt.show()
            
            # Reproducir el sonido sintetizado
            self.reproducir_audio(y_sintetizado, sr_sintetizado)
            
            # Preguntar si desea guardar
            self.preguntar_guardar(y_sintetizado, "Sonido Sintetizado")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en síntesis: {str(e)}")

    def aplicar_compresion(self):
        try:
            if self.processor.y is None:
                QMessageBox.critical(self, "Error", "Por favor, carga un archivo de audio primero")
                return
            if self.audio_player.playing:
                self.audio_player.stop()
                self.play_pause_button.setText("Play")
                
            # Calcular la FFT y aplicar compresión
            fft_data = fft(self.processor.y)
            frecuencias = fftfreq(len(self.processor.y), 1/self.processor.sr)
            
            # Crear una figura con dos subplots
            plt.figure(figsize=(12, 8))
            
            # Graficar espectro original
            plt.subplot(2, 2, 1)
            plt.plot(frecuencias[:len(frecuencias)//2], np.abs(fft_data)[:len(frecuencias)//2])
            plt.title("Espectro Original")
            plt.xlabel("Frecuencia (Hz)")
            plt.ylabel("Magnitud")
            
            # Aplicar compresión en el dominio de la frecuencia
            umbral = int(len(fft_data) * 50 / 100)  # 50% de compresión
            fft_comprimido = fft_data.copy()
            fft_comprimido[umbral:-umbral] = 0
            
            # Graficar espectro comprimido
            plt.subplot(2, 2, 2)
            plt.plot(frecuencias[:len(frecuencias)//2], np.abs(fft_comprimido)[:len(frecuencias)//2])
            plt.title("Espectro Comprimido")
            plt.xlabel("Frecuencia (Hz)")
            plt.ylabel("Magnitud")
            
            # Convertir de vuelta al dominio del tiempo
            y_comprimido = ifft(fft_comprimido).real
            
            # Graficar señal original
            plt.subplot(2, 2, 3)
            plt.plot(self.processor.y)
            plt.title("Señal Original")
            plt.xlabel("Muestras")
            plt.ylabel("Amplitud")
            
            # Graficar señal comprimida
            plt.subplot(2, 2, 4)
            plt.plot(y_comprimido)
            plt.title("Señal Comprimida")
            plt.xlabel("Muestras")
            plt.ylabel("Amplitud")
            
            plt.tight_layout()
            plt.show()
            
            # Reproducir audio comprimido
            self.reproducir_audio(y_comprimido, self.processor.sr)
            
            # Preguntar si desea guardar
            self.preguntar_guardar(y_comprimido, "Audio Comprimido")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en compresión: {str(e)}")

    def closeEvent(self, event):
        """Manejo limpio del cierre de la aplicación"""
        try:
            if self.audio_player:
                self.audio_player.stop()
            if hasattr(self, 'timer'):
                self.timer.stop()
        except Exception as e:
            print(f"Error al cerrar: {e}")
        event.accept()

    def cargar_audio(self):
        """Método para manejar la carga de audio"""
        y, sr = self.processor.cargar_audio()
        if y is not None and sr is not None:
            self.plot_audio(y, "Audio Original")
            self.ultimo_audio_procesado = y

# Ejecutar la aplicación
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ventana = MainWindow()
    ventana.show()
    sys.exit(app.exec())  # Nota: en PySide6 es exec() en lugar de exec_()