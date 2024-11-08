import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QWidget, QFileDialog, QMessageBox, QSlider, QLabel,
                             QHBoxLayout, QProgressBar, QFrame, QSplitter, QSizePolicy)
from PySide6.QtCore import QTimer
from PySide6.QtCore import Qt
from scipy.fft import fft, ifft, fftfreq
import sys
from scipy.io import wavfile
import sounddevice as sd
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Función para filtrar ruido de una señal de audio usando FFT
def filtrar_ruido(y, sr, umbral=5000):
    """
    Esta función se encarga de limpiar el ruido de una señal de audio.
    Usa la transformada de Fourier para convertir la señal al dominio de frecuencia,
    elimina las frecuencias por encima del umbral (que normalmente es ruido),
    y vuelve a convertir la señal al dominio del tiempo.
    
    Parámetros:
    - y: señal de audio
    - sr: frecuencia de muestreo
    - umbral: frecuencia de corte para el filtro (por defecto 5000 Hz)
    """
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

# Función para sintetizar un sonido de una frecuencia específica
def sintetizar_sonido(frecuencia, duracion=2, sr=44100):
    """
    Genera un tono puro usando una onda sinusoidal.
    Útil para pruebas y para entender cómo se genera el sonido digitalmente.
    
    Parámetros:
    - frecuencia: frecuencia del tono en Hz
    - duracion: duración en segundos
    - sr: frecuencia de muestreo (calidad del audio)
    """
    t = np.linspace(0, duracion, int(sr * duracion), endpoint=False)
    y_sintetizado = 0.5 * np.sin(2 * np.pi * frecuencia * t)
    return y_sintetizado, sr

# Función para comprimir audio eliminando componentes de frecuencia
def comprimir_audio(y, porcentaje=50):
    if y is None:
        return None
        
    fft_y = fft(y)
    umbral = int(len(fft_y) * porcentaje / 100)
    fft_y[umbral:] = 0  # Mantener solo el porcentaje seleccionado de componentes
    y_comprimido = ifft(fft_y).real
    return y_comprimido

# Clase para procesar archivos de audio
class AudioProcessor:
    """
    Clase principal para el procesamiento de archivos de audio.
    Maneja la carga y procesamiento básico de archivos WAV.
    """
    
    def __init__(self):
        """
        Inicializa las variables necesarias para el procesador de audio.
        y: datos de audio
        sr: frecuencia de muestreo
        """
        self.y = None
        self.sr = None

    # Método para cargar un archivo de audio WAV
    def cargar_audio(self):
        """
        Abre un diálogo para seleccionar y cargar un archivo WAV.
        Normaliza el audio y lo convierte a mono si es necesario.
        Muestra mensajes de éxito o error según corresponda.
        """
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

# Clase para reproducir audio
class AudioPlayer:
    """
    Maneja la reproducción del audio usando sounddevice.
    Implementa controles básicos como play, pause, stop y seek.
    """
    
    def __init__(self):
        """
        Inicializa el reproductor con valores por defecto y 
        prepara el sistema de streaming de audio.
        """
        self.playing = False
        self.current_position = 0
        self.audio_data = None
        self.sr = None
        self.stream = None
        
    # Método para iniciar la reproducción
    def play(self, data, sr):
        """
        Inicia la reproducción del audio.
        Detiene cualquier reproducción anterior si existe.
        
        Parámetros:
        - data: datos de audio a reproducir
        - sr: frecuencia de muestreo
        """
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
            
    # Método para detener la reproducción
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
            
    # Método para pausar la reproducción
    def pause(self):
        self.playing = False
        if self.stream:
            try:
                self.stream.stop()
            except Exception as e:
                print(f"Error al pausar: {e}")
            
    # Método para reanudar la reproducción
    def resume(self):
        if self.audio_data is not None and self.stream:
            try:
                self.playing = True
                self.stream.start()
            except Exception as e:
                print(f"Error al resumir: {e}")
                self.stop()

    # Callback para el stream de audio
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
            
    # Método para buscar una posición en el audio
    def seek(self, position):
        self.current_position = int(position * len(self.audio_data))
        
    # Método para obtener el progreso actual
    def get_progress(self):
        if self.audio_data is None:
            return 0
        return self.current_position / len(self.audio_data) * 100

# Clase principal de la interfaz gráfica
class MainWindow(QMainWindow):
    """
    Ventana principal de la aplicación.
    Integra todos los componentes y maneja la interfaz gráfica.
    """
    
    def __init__(self):
        """
        Configura la interfaz gráfica principal.
        Inicializa todos los componentes, estilos y conexiones.
        """
        super().__init__()
        
        # Definir símbolos como constantes de clase
        self.PLAY_SYMBOL = "▶"
        self.PAUSE_SYMBOL = "❚❚"
        
        # Inicializar processor y audio_player
        self.processor = AudioProcessor()
        self.audio_player = AudioPlayer()
        self.y_filtrado = None
        self.y_comprimido = None
        self.currently_playing = None

        # Obtener el tamaño de la pantalla
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.8)  # 80% del ancho de la pantalla
        height = int(screen.height() * 0.8)  # 80% del alto de la pantalla
        
        # Configurar el tamaño inicial de la ventana
        self.setGeometry(
            (screen.width() - width) // 2,  # Centrar horizontalmente
            (screen.height() - height) // 2,  # Centrar verticalmente
            width,
            height
        )
        
        # Configurar tamaño mínimo para evitar problemas de visualización
        self.setMinimumSize(800, 600)

        # Estilos CSS para la interfaz
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

        # Configuración del widget central y layouts
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Panel izquierdo para controles
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(15)

        # Título de la aplicación
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

        # Widget de controles de reproducción
        self.control_widget = QWidget()
        control_layout = QVBoxLayout(self.control_widget)
        
        # Botones de control de reproducción
        playback_buttons = QHBoxLayout()
        self.play_pause_button = QPushButton("▶")
        self.play_pause_button.setFixedSize(40, 40)
        self.play_pause_button.setStyleSheet("""
            QPushButton {
                border-radius: 20px;
                padding: 5px;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        playback_buttons.addWidget(self.play_pause_button)
        playback_buttons.addStretch()
        
        # Barra de progreso
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
        
        # Slider y etiqueta de tiempo
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 100)
        self.seek_slider.sliderPressed.connect(self.on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self.on_slider_released)
        self.seek_slider.sliderMoved.connect(self.on_slider_moved)
        self.slider_pressed = False
        
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setAlignment(Qt.AlignRight)
        
        # Agregar widgets al layout de control
        control_layout.addLayout(playback_buttons)
        control_layout.addWidget(self.seek_slider)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.time_label)
        
        left_layout.addWidget(self.control_widget)

        # Botones de funciones principales
        boton_cargar = QPushButton("Cargar Audio")
        boton_filtrado = QPushButton("Filtrar Ruido")
        boton_sintesis = QPushButton("Sintetizar")
        boton_compresion = QPushButton("Comprimir")

        # Estilo adicional para botones
        botones_estilo = """
            QPushButton {
                padding: 12px 20px;
                font-size: 14px;
                font-weight: bold;
                text-align: center;
            }
        """
        for boton in [boton_cargar, boton_filtrado, boton_sintesis, boton_compresion]:
            boton.setStyleSheet(botones_estilo)

        # Agregar botones al panel izquierdo
        left_layout.addWidget(boton_cargar)
        left_layout.addWidget(boton_filtrado)
        left_layout.addWidget(boton_sintesis)
        left_layout.addWidget(boton_compresion)
        left_layout.addStretch()

        # Panel derecho para gráficas
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Configuración de matplotlib
        self.figure = Figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        
        # Botón de guardar
        self.save_button = QPushButton(" Guardar")
        self.save_button.setIcon(QIcon("icons/save.png"))
        self.save_button.hide()
        right_layout.addWidget(self.save_button)

        # Configuración del splitter y layout principal
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)  # Panel izquierdo
        splitter.setStretchFactor(1, 3)  # Panel derecho (más espacio para gráficas)
        main_layout.addWidget(splitter)

        # Ajustar tamaños mínimos de los paneles
        left_panel.setMinimumWidth(int(width * 0.25))  # 25% del ancho mínimo
        right_panel.setMinimumWidth(int(width * 0.5))  # 50% del ancho mínimo
        
        # Hacer que los botones se ajusten al ancho del panel
        for boton in [boton_cargar, boton_filtrado, boton_sintesis, boton_compresion]:
            boton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            
        # Ajustar el tamaño del botón de reproducción
        button_size = int(min(width, height) * 0.05)  # 5% del menor lado
        self.play_pause_button.setFixedSize(button_size, button_size)
        
        # Ajustar el tamaño de la figura de matplotlib
        self.figure.set_size_inches(width/100, height/100)
        
        # Hacer que el canvas se ajuste al contenedor
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Conexión de señales
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        boton_cargar.clicked.connect(self.cargar_audio)
        boton_filtrado.clicked.connect(self.aplicar_filtro)
        boton_sintesis.clicked.connect(self.generar_sintesis)
        boton_compresion.clicked.connect(self.aplicar_compresion)
        self.save_button.clicked.connect(self.guardar_audio_actual)

        # Timer para actualizar progreso
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(100)

    # Método para graficar audio
    def plot_audio(self, data, title):
        """
        Grafica los datos de audio en el panel derecho.
        Actualiza el botón de guardar según el tipo de audio.
        
        Parámetros:
        - data: datos a graficar
        - title: título de la gráfica
        """
        """Método para graficar en el panel derecho"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(data, color='#333333')
        ax.set_title(title)
        ax.set_xlabel('Muestras')
        ax.set_ylabel('Amplitud')
        ax.grid(True, linestyle='--', alpha=0.7)
        self.canvas.draw()
        
        # Actualizar el botón de guardar según el tipo de audio
        if "Sintetizado" in title:
            self.save_button.setText("Guardar Audio Sintetizado")
        elif "Comprimido" in title:
            self.save_button.setText("Guardar Audio Comprimido")
        elif "Filtrado" in title:
            self.save_button.setText("Guardar Audio Filtrado")
        else:
            self.save_button.setText("Guardar Audio")
        self.save_button.show()

    # Método para guardar el audio actual
    def guardar_audio_actual(self):
        if hasattr(self, 'ultimo_audio_procesado'):
            self.preguntar_guardar(self.ultimo_audio_procesado, 
                                 self.save_button.text().replace(" Guardar ", ""))

    # Método para alternar reproducción/pausa
    def toggle_play_pause(self):
        if self.audio_player.playing:
            self.audio_player.pause()
            self.play_pause_button.setText(self.PLAY_SYMBOL)
        else:
            self.audio_player.resume()
            self.play_pause_button.setText(self.PAUSE_SYMBOL)

    # Método para buscar en el audio
    def seek_audio(self, position):
        self.audio_player.seek(position / 100.0)

    # Métodos para manejar eventos del slider
    def on_slider_pressed(self):
        self.slider_pressed = True
        if self.audio_player.playing:
            self.audio_player.pause()

    def on_slider_released(self):
        self.slider_pressed = False
        position = self.seek_slider.value() / 100.0
        self.audio_player.seek(position)
        if self.audio_player.playing:
            self.audio_player.resume()

    def on_slider_moved(self, position):
        if self.audio_player.audio_data is not None:
            total_time = len(self.audio_player.audio_data) / self.audio_player.sr
            current_time = position * total_time / 100
            self.time_label.setText(f"{int(current_time//60)}:{int(current_time%60):02d} / "
                                  f"{int(total_time//60)}:{int(total_time%60):02d}")

    # Método para actualizar la barra de progreso
    def update_progress(self):
        if self.audio_player.audio_data is not None and not self.slider_pressed:
            progress = self.audio_player.get_progress()
            self.progress_bar.setValue(int(progress))
            self.seek_slider.setValue(int(progress))
            
            current_time = self.audio_player.current_position / self.audio_player.sr
            total_time = len(self.audio_player.audio_data) / self.audio_player.sr
            self.time_label.setText(f"{int(current_time//60)}:{int(current_time%60):02d} / "
                                  f"{int(total_time//60)}:{int(total_time%60):02d}")

    # Método para guardar audio
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
                audio_normalizado = np.int16(self.ultimo_audio_procesado * 32767)
                wavfile.write(archivo_guardar, self.processor.sr, audio_normalizado)
                QMessageBox.information(self, "Éxito", "Audio guardado correctamente")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al guardar el archivo: {str(e)}")

    # Método para reproducir audio
    def reproducir_audio(self, datos, sr):
        try:
            if datos is not None and sr is not None:
                if self.audio_player.playing:
                    self.audio_player.stop()
                    self.play_pause_button.setText(self.PLAY_SYMBOL)
                
                self.audio_player.play(datos, sr)
                self.play_pause_button.setText(self.PAUSE_SYMBOL)
                self.currently_playing = datos
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al reproducir: {str(e)}")

    # Método para reproducir audio original
    def reproducir_original(self):
        if self.processor.y is None:
            QMessageBox.critical(self, "Error", "Por favor, carga un archivo de audio primero")
            return
        self.reproducir_audio(self.processor.y, self.processor.sr)

    # Método para reproducir audio filtrado
    def reproducir_filtrado(self):
        if self.y_filtrado is None:
            QMessageBox.critical(self, "Error", "Por favor, aplica un filtro primero")
            return
        self.reproducir_audio(self.y_filtrado, self.processor.sr)

    # Método para preguntar si guardar audio
    def preguntar_guardar(self, audio_modificado, titulo):
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
                    audio_normalizado = np.int16(audio_modificado * 32767)
                    wavfile.write(archivo_guardar, self.processor.sr, audio_normalizado)
                    QMessageBox.information(self, "Éxito", f"{titulo} guardado correctamente")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error al guardar el archivo: {str(e)}")

    # Método para aplicar filtro de ruido
    def aplicar_filtro(self):
        try:
            if self.processor.y is None:
                QMessageBox.critical(self, "Error", "Por favor, carga un archivo de audio primero")
                return
            
            if self.audio_player.playing:
                self.audio_player.stop()
                self.play_pause_button.setText("▶")
                
            self.y_filtrado = filtrar_ruido(self.processor.y, self.processor.sr)
            if self.y_filtrado is not None:
                self.plot_audio(self.y_filtrado, "Audio Filtrado")
                self.reproducir_audio(self.y_filtrado, self.processor.sr)
                self.ultimo_audio_procesado = self.y_filtrado
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al filtrar: {str(e)}")

    # Método para generar síntesis de audio
    def generar_sintesis(self):
        try:
            if self.audio_player.playing:
                self.audio_player.stop()
                self.play_pause_button.setText(self.PLAY_SYMBOL)
                
            y_sintetizado, sr_sintetizado = sintetizar_sonido(frecuencia=440)
            
            # Usar el método plot_audio para mostrar en la ventana principal
            self.plot_audio(y_sintetizado, "Sonido Sintetizado (440Hz)")
            self.reproducir_audio(y_sintetizado, sr_sintetizado)
            self.ultimo_audio_procesado = y_sintetizado
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en síntesis: {str(e)}")

    # Método para aplicar compresión de audio
    def aplicar_compresion(self):
        try:
            if self.processor.y is None:
                QMessageBox.critical(self, "Error", "Por favor, carga un archivo de audio primero")
                return
            if self.audio_player.playing:
                self.audio_player.stop()
                self.play_pause_button.setText(self.PLAY_SYMBOL)
                
            # Calcular FFT y aplicar compresión
            fft_data = fft(self.processor.y)
            frecuencias = fftfreq(len(self.processor.y), 1/self.processor.sr)
            
            # Aplicar compresión
            umbral = int(len(fft_data) * 50 / 100)
            fft_comprimido = fft_data.copy()
            fft_comprimido[umbral:-umbral] = 0
            y_comprimido = ifft(fft_comprimido).real
            
            # Mostrar en la ventana principal
            self.figure.clear()
            
            # Crear subplots en la figura principal
            gs = self.figure.add_gridspec(2, 2)
            ax1 = self.figure.add_subplot(gs[0, 0])
            ax2 = self.figure.add_subplot(gs[0, 1])
            ax3 = self.figure.add_subplot(gs[1, 0])
            ax4 = self.figure.add_subplot(gs[1, 1])
            
            # Graficar espectros y señales
            ax1.plot(frecuencias[:len(frecuencias)//2], np.abs(fft_data)[:len(frecuencias)//2])
            ax1.set_title("Espectro Original")
            ax1.set_xlabel("Frecuencia (Hz)")
            ax1.set_ylabel("Magnitud")
            
            ax2.plot(frecuencias[:len(frecuencias)//2], np.abs(fft_comprimido)[:len(frecuencias)//2])
            ax2.set_title("Espectro Comprimido")
            ax2.set_xlabel("Frecuencia (Hz)")
            ax2.set_ylabel("Magnitud")
            
            ax3.plot(self.processor.y)
            ax3.set_title("Señal Original")
            ax3.set_xlabel("Muestras")
            ax3.set_ylabel("Amplitud")
            
            ax4.plot(y_comprimido)
            ax4.set_title("Señal Comprimida")
            ax4.set_xlabel("Muestras")
            ax4.set_ylabel("Amplitud")
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            # Reproducir y preparar para guardar
            self.reproducir_audio(y_comprimido, self.processor.sr)
            self.ultimo_audio_procesado = y_comprimido
            
            # Actualizar el botón de guardar
            self.save_button.setText("Guardar Audio Comprimido")
            self.save_button.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en compresión: {str(e)}")

    # Método para manejar el cierre de la aplicación
    def closeEvent(self, event):
        try:
            if self.audio_player:
                self.audio_player.stop()
            if hasattr(self, 'timer'):
                self.timer.stop()
        except Exception as e:
            print(f"Error al cerrar: {e}")
        event.accept()

    # Método para cargar audio
    def cargar_audio(self):
        y, sr = self.processor.cargar_audio()
        if y is not None and sr is not None:
            self.plot_audio(y, "Audio Original")
            self.ultimo_audio_procesado = y

    def resizeEvent(self, event):
        """Manejar el redimensionamiento de la ventana"""
        super().resizeEvent(event)
        
        # Actualizar el tamaño de los elementos cuando se redimensiona la ventana
        width = self.width()
        height = self.height()
        
        # Ajustar el tamaño del botón de reproducción
        button_size = int(min(width, height) * 0.05)
        self.play_pause_button.setFixedSize(button_size, button_size)
        
        # Redibujar el canvas de matplotlib
        self.canvas.draw()

# Punto de entrada de la aplicación
if __name__ == '__main__':
    """
    Inicia la aplicación Qt y muestra la ventana principal.
    El programa se ejecutará hasta que se cierre la ventana.
    """
    app = QApplication(sys.argv)
    ventana = MainWindow()
    ventana.show()
    sys.exit(app.exec())  