import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QWidget, QFileDialog, QMessageBox, QSlider, QLabel,
                             QHBoxLayout, QProgressBar)
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
        self.setWindowTitle("Procesamiento de Audio con Series de Fourier")
        self.setGeometry(100, 100, 600, 400)
        
        self.processor = AudioProcessor()
        self.audio_player = AudioPlayer()
        self.y_filtrado = None
        self.y_comprimido = None
        self.currently_playing = None  # Para rastrear qué audio se está reproduciendo
        
        # Crear widget central y layout
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout = QVBoxLayout(widget_central)
        
        # Crear widget para controles de reproducción
        self.control_widget = QWidget()
        control_layout = QHBoxLayout(self.control_widget)
        
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        control_layout.addWidget(self.play_pause_button)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        control_layout.addWidget(self.progress_bar)
        
        self.time_label = QLabel("0:00 / 0:00")
        control_layout.addWidget(self.time_label)
        
        # Agregar slider para buscar en el audio
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 100)
        self.seek_slider.sliderMoved.connect(self.seek_audio)
        control_layout.addWidget(self.seek_slider)
        
        # Ocultar controles inicialmente
        self.control_widget.hide()
        
        # Agregar botón para mostrar/ocultar línea de tiempo
        self.boton_timeline = QPushButton("Mostrar Línea de Tiempo")
        self.boton_timeline.clicked.connect(self.toggle_timeline)
        layout.addWidget(self.boton_timeline)
        
        # Agregar el widget de controles al layout principal
        layout.addWidget(self.control_widget)
        
        # Botones de funcionalidad
        boton_cargar = QPushButton("Cargar Audio")
        boton_cargar.clicked.connect(self.processor.cargar_audio)
        layout.addWidget(boton_cargar)
        
        boton_filtrado = QPushButton("Aplicar Filtrado de Ruido")
        boton_filtrado.clicked.connect(self.aplicar_filtro)
        layout.addWidget(boton_filtrado)
        
        boton_sintesis = QPushButton("Generar Síntesis de Sonido (440Hz)")
        boton_sintesis.clicked.connect(self.generar_sintesis)
        layout.addWidget(boton_sintesis)
        
        boton_compresion = QPushButton("Aplicar Compresión de Audio")
        boton_compresion.clicked.connect(self.aplicar_compresion)
        layout.addWidget(boton_compresion)
        
        # Timer para actualizar la barra de progreso
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(100)  # Actualizar cada 100ms

    def toggle_timeline(self):
        if self.control_widget.isHidden():
            self.control_widget.show()
            self.boton_timeline.setText("Ocultar Línea de Tiempo")
        else:
            self.control_widget.hide()
            self.boton_timeline.setText("Mostrar Línea de Tiempo")

    def toggle_play_pause(self):
        try:
            if self.currently_playing is None:
                QMessageBox.warning(self, "Aviso", "No hay audio para reproducir")
                return
                
            if self.audio_player.playing:
                self.audio_player.pause()
                self.play_pause_button.setText("Play")
            else:
                self.audio_player.resume()
                self.play_pause_button.setText("Pause")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en reproducción: {str(e)}")

    def seek_audio(self, position):
        self.audio_player.seek(position / 100.0)

    def update_progress(self):
        if self.audio_player.audio_data is not None:
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
                self.play_pause_button.setText("Play")
                
            self.y_filtrado = filtrar_ruido(self.processor.y, self.processor.sr)
            if self.y_filtrado is not None:
                plt.figure(figsize=(10, 4))
                plt.plot(self.y_filtrado)
                plt.title("Audio Filtrado")
                plt.xlabel("Muestras")
                plt.ylabel("Amplitud")
                plt.grid(True)
                plt.show()
                
                # Reproducir el audio filtrado
                self.reproducir_audio(self.y_filtrado, self.processor.sr)
                
                # Preguntar si desea guardar
                self.preguntar_guardar(self.y_filtrado, "Audio Filtrado")
                
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

# Ejecutar la aplicación
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ventana = MainWindow()
    ventana.show()
    sys.exit(app.exec())  # Nota: en PySide6 es exec() en lugar de exec_()