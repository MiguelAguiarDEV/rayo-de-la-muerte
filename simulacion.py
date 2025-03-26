# demo_modelo_laser.py
import os
import time
import cv2
import numpy as np
import pygame
from pygame.locals import *
import glob
import random
import argparse
import datetime
from PIL import Image, ImageDraw, ImageOps, ImageEnhance, ImageFilter
import math

# Manejo de errores para importar YOLO
try:
    from ultralytics import YOLO
    YOLO_DISPONIBLE = True
except Exception:
    print("Modo simulación sin detección real.")
    YOLO_DISPONIBLE = False

# Configuración
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
FPS = 30
LASER_SIZE = 10
LASER_COLOR = (0, 255, 0)  # Verde
FOLLOW_DELAY = 2  # Segundos sobre cada objetivo
MAX_LASER_SPEED = 15  # Velocidad máxima
NUM_TEST_IMAGES = 10  # Número de imágenes de prueba a generar
SIMULACIONES_DIR = "simulaciones"  # Carpeta para guardar videos

# Configuración del modelo (puede ser modificado por línea de comandos)
MODEL_PATH = "best.pt"  # Ruta por defecto

# Nombres de clases y colores
NOMBRES_CLASES = {0: 'mariposa', 1: 'mariquita', 2: 'cucaracha', 3: 'limite', 4: 'laser'}
COLORES = {
    0: (255, 0, 0),    # Mariposa - Azul
    1: (0, 0, 255),    # Mariquita - Rojo
    2: (42, 42, 165),  # Cucaracha - Marrón
    3: (128, 0, 128),  # Límite - Morado
    4: (0, 255, 0)     # Láser - Verde
}

class TestImageGenerator:
    """Generador de imágenes de prueba para la simulación"""
    def __init__(self, output_dir="test_images"):
        self.output_dir = output_dir
        self.dir_recursos = "recursos"
        self.max_width = SCREEN_WIDTH
        self.max_height = SCREEN_HEIGHT
        self.resources = {}
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
    
    def load_resources(self):
        """Carga los recursos necesarios para generar imágenes"""
        print("Cargando recursos para generar imágenes de prueba...")
        
        try:
            # Cargar imágenes de recursos
            self.resources["limite"] = Image.open(os.path.join(self.dir_recursos, "cruz.png")).convert("RGBA")
            self.resources["mariposa"] = Image.open(os.path.join(self.dir_recursos, "mariposa.png")).convert("RGBA")
            self.resources["mariquita"] = Image.open(os.path.join(self.dir_recursos, "mariquita.png")).convert("RGBA")
            self.resources["cucaracha"] = Image.open(os.path.join(self.dir_recursos, "cucaracha.png")).convert("RGBA")
            self.resources["laser"] = Image.open(os.path.join(self.dir_recursos, "laser.png")).convert("RGBA")
            
            print("Recursos cargados correctamente")
            return True
        except Exception as e:
            print(f"Error al cargar recursos: {e}")
            print("Asegúrate de tener una carpeta 'recursos' con las imágenes necesarias.")
            return False
    
    def generar_fondo_aleatorio(self, width, height):
        """Genera un fondo aleatorio"""
        tipo = random.choice(["solido", "gradiente", "ruido", "textura"])
        
        if tipo == "solido":
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            fondo = Image.new('RGB', (width, height), color)
        
        elif tipo == "gradiente":
            fondo = Image.new('RGB', (width, height))
            array = np.zeros((height, width, 3), dtype=np.uint8)
            color1 = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
            color2 = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
            
            for y in range(height):
                factor = y / height
                color = color1 * (1 - factor) + color2 * factor
                array[y, :] = color
            
            fondo = Image.fromarray(array.astype('uint8'))
        
        elif tipo == "ruido":
            array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            if random.random() > 0.5:
                array = cv2.GaussianBlur(array, (0, 0), random.uniform(1.0, 5.0))
            fondo = Image.fromarray(array)
        
        else:  # textura
            array = np.zeros((height, width, 3), dtype=np.uint8)
            scale = random.randint(5, 30)
            
            for y in range(height):
                for x in range(width):
                    v = int(127 + 127 * np.sin(x/scale) * np.sin(y/scale))
                    array[y, x] = [v, v, v]
            
            color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
            for i in range(3):
                array[:,:,i] = (array[:,:,0] * color[i] / 255).astype(np.uint8)
            
            fondo = Image.fromarray(array)
        
        # Efectos adicionales
        if random.random() > 0.5:
            fondo = fondo.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 2.0)))
        
        if random.random() > 0.7:
            ruido = np.random.randint(0, 25, (height, width, 3), dtype=np.uint8)
            ruido_img = Image.fromarray(ruido)
            fondo = Image.blend(fondo, ruido_img, 0.1)
        
        return fondo
    
    def aplicar_transformaciones(self, imagen):
        """Aplica transformaciones aleatorias a una imagen"""
        # Escalar
        factor = random.uniform(0.5, 1.5)
        nuevo_w = int(imagen.width * factor)
        nuevo_h = int(imagen.height * factor)
        imagen = imagen.resize((nuevo_w, nuevo_h), Image.LANCZOS)
        
        # Rotar
        angulo = random.uniform(0, 360)
        imagen = imagen.rotate(angulo, expand=True, resample=Image.BICUBIC)
        
        # Voltear
        if random.random() > 0.5:
            imagen = ImageOps.mirror(imagen)
        if random.random() > 0.5:
            imagen = ImageOps.flip(imagen)
        
        # Ajustar brillo y contraste
        imagen = ImageEnhance.Brightness(imagen).enhance(random.uniform(0.8, 1.2))
        imagen = ImageEnhance.Contrast(imagen).enhance(random.uniform(0.8, 1.2))
        
        return imagen
    
    def generate_images(self, num_images=NUM_TEST_IMAGES, min_objetos=5, max_objetos=15):
        """Genera el número especificado de imágenes de prueba"""
        if not self.load_resources():
            return []
        
        print(f"Generando {num_images} imágenes de prueba...")
        generated_paths = []
        
        for idx in range(num_images):
            # Generar fondo aleatorio
            fondo = self.generar_fondo_aleatorio(self.max_width, self.max_height)
            labels = []
            
            # Colocar marcas de límite en las esquinas
            limite_img = self.resources["limite"]
            esquinas = [
                (0, 0),
                (self.max_width - limite_img.width, 0),
                (0, self.max_height - limite_img.height),
                (self.max_width - limite_img.width, self.max_height - limite_img.height)
            ]
            
            for esquina in esquinas:
                fondo.paste(limite_img, esquina, limite_img)
                
                # Etiqueta YOLO: clase x_center y_center width height
                x_center = (esquina[0] + limite_img.width / 2) / self.max_width
                y_center = (esquina[1] + limite_img.height / 2) / self.max_height
                width = limite_img.width / self.max_width
                height = limite_img.height / self.max_height
                labels.append(f"3 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Crear cuadrícula para distribuir objetos
            num_objetos = random.randint(min_objetos, max_objetos)
            grid_size = math.ceil(math.sqrt(num_objetos * 1.5))
            cell_width = self.max_width / grid_size
            cell_height = self.max_height / grid_size
            
            # Mezclar celdas disponibles
            celdas = [(r, c) for r in range(grid_size) for c in range(grid_size)]
            random.shuffle(celdas)
            
            # Clases disponibles y sus imágenes
            clases = ["mariposa", "mariquita", "cucaracha"]
            
            # Colocar objetos
            objetos_colocados = 0
            
            while objetos_colocados < num_objetos and celdas:
                row, col = celdas.pop()
                
                # Seleccionar clase aleatoria
                clase = random.choice(clases)
                img_objeto = self.resources[clase].copy()
                
                # Aplicar transformaciones
                img_objeto = self.aplicar_transformaciones(img_objeto)
                
                # Calcular posición
                pos_x = int(col * cell_width + random.uniform(0, cell_width - img_objeto.width))
                pos_y = int(row * cell_height + random.uniform(0, cell_height - img_objeto.height))
                
                # Asegurar que está dentro de los límites
                pos_x = max(0, min(pos_x, self.max_width - img_objeto.width))
                pos_y = max(0, min(pos_y, self.max_height - img_objeto.height))
                
                # Pegar objeto
                fondo.paste(img_objeto, (pos_x, pos_y), img_objeto)
                
                # Calcular etiqueta YOLO
                x_center = (pos_x + img_objeto.width / 2) / self.max_width
                y_center = (pos_y + img_objeto.height / 2) / self.max_height
                width = img_objeto.width / self.max_width
                height = img_objeto.height / self.max_height
                
                # Índice de clase (0=mariposa, 1=mariquita, 2=cucaracha)
                class_idx = clases.index(clase)
                labels.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                objetos_colocados += 1
                
                # Agregar puntero láser cerca de algunos objetos (30% probabilidad)
                # if random.random() < 0.3:
                #     laser_copy = self.resources["laser"].copy()
                    
                #     # Escalar láser aleatoriamente
                #     if random.random() > 0.5:
                #         scale_factor = random.uniform(0.7, 1.3)
                #         laser_width = int(laser_copy.width * scale_factor)
                #         laser_height = int(laser_copy.height * scale_factor)
                #         laser_copy = laser_copy.resize((laser_width, laser_height), Image.LANCZOS)
                    
                #     # Posicionar láser cerca del objeto
                #     pos_laser_x = pos_x + random.randint(-20, img_objeto.width + 20)
                #     pos_laser_y = pos_y + random.randint(-20, img_objeto.height + 20)
                    
                #     # Asegurar que está dentro de los límites
                #     pos_laser_x = max(0, min(pos_laser_x, self.max_width - laser_copy.width))
                #     pos_laser_y = max(0, min(pos_laser_y, self.max_height - laser_copy.height))
                    
                #     # Pegar láser
                #     fondo.paste(laser_copy, (pos_laser_x, pos_laser_y), laser_copy)
                    
                #     # Etiqueta para el láser (clase 4)
                #     x_center_laser = (pos_laser_x + laser_copy.width / 2) / self.max_width
                #     y_center_laser = (pos_laser_y + laser_copy.height / 2) / self.max_height
                #     width_laser = laser_copy.width / self.max_width
                #     height_laser = laser_copy.height / self.max_height
                    
                #     labels.append(f"4 {x_center_laser:.6f} {y_center_laser:.6f} {width_laser:.6f} {height_laser:.6f}")
            
            # Guardar imagen
            img_path = os.path.join(self.output_dir, f"test_image_{idx+1}.jpg")
            fondo.convert("RGB").save(img_path, quality=95)
            
            # Guardar etiquetas
            label_path = os.path.join(self.output_dir, f"test_image_{idx+1}.txt")
            with open(label_path, "w") as f:
                f.write("\n".join(labels))
            
            generated_paths.append(img_path)
            print(f"Generada imagen {idx+1}/{num_images}: {img_path}")
            
        print(f"Generación completada: {len(generated_paths)} imágenes creadas en '{self.output_dir}'")
        return generated_paths

class DemoLaser:
    def __init__(self, model_path=None, test_images_dir="test_images", save_video=True, video_dir=SIMULACIONES_DIR):
        # Configurar directorio para videos
        self.video_dir = video_dir
        os.makedirs(self.video_dir, exist_ok=True)
        
        # Inicializar Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Demostración de Seguimiento Láser")
        self.clock = pygame.time.Clock()
        
        # Cargar modelo YOLO
        self.model = None
        if YOLO_DISPONIBLE and model_path:
            try:
                self.model = YOLO(model_path)
                print(f"Modelo cargado: {model_path}")
            except Exception as e:
                print(f"Error al cargar el modelo: {e}")
                print("Continuando en modo simulación...")
        
        # Cargar imágenes de prueba
        self.test_images_dir = test_images_dir
        self.images = []
        
        if os.path.exists(test_images_dir):
            self.images = sorted(glob.glob(os.path.join(test_images_dir, "*.jpg")))
            print(f"Cargadas {len(self.images)} imágenes de prueba de {test_images_dir}")
        
        if not self.images:
            print("No se encontraron imágenes de prueba.")
            print("Generando imágenes de prueba...")
            generator = TestImageGenerator(test_images_dir)
            self.images = generator.generate_images(NUM_TEST_IMAGES)
        
        # Estado de la simulación
        self.current_image_idx = 0
        self.laser_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]
        self.target_pos = None
        self.current_target_idx = 0
        self.targets = []
        self.last_target_time = time.time()
        self.target_class = 0  # Mariposas por defecto
        
        # Configurar grabación de video
        self.save_video = save_video
        self.video_writer = None
        if save_video:
            # Nombre de archivo basado en fecha y hora
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = os.path.join(self.video_dir, f"demo_laser_{timestamp}.avi")
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (SCREEN_WIDTH, SCREEN_HEIGHT))
            print(f"Grabando video en: {video_filename}")
    
    def get_current_image(self):
        """Obtiene la imagen actual para mostrar."""
        if not self.images:
            return np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        
        img = cv2.imread(self.images[self.current_image_idx])
        if img.shape[0] != SCREEN_HEIGHT or img.shape[1] != SCREEN_WIDTH:
            img = cv2.resize(img, (SCREEN_WIDTH, SCREEN_HEIGHT))
        
        return img
    
    def cambiar_imagen(self, delta=1):
        """Cambia a la siguiente o anterior imagen."""
        if not self.images:
            return
        
        self.current_image_idx = (self.current_image_idx + delta) % len(self.images)
        self.laser_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]
        self.target_pos = None
        self.targets = []
    
    def detect_objects(self, frame):
        """Detecta objetos usando YOLO o etiquetas de archivo."""
        if self.model is None or not YOLO_DISPONIBLE:
            return self.extract_labels_from_file(frame)
        
        try:
            results = self.model.predict(source=frame, conf=0.25)
            return results[0]
        except Exception as e:
            print(f"Error en la detección: {e}")
            return self.extract_labels_from_file(frame)
    
    def extract_labels_from_file(self, frame):
        """Extrae etiquetas desde el archivo .txt correspondiente."""
        if not self.images:
            return None
        
        # Clase para resultados simulados
        class SimulatedResults:
            def __init__(self):
                self.boxes = []
        
        class SimulatedBox:
            def __init__(self, cls, x1, y1, x2, y2, conf=0.9):
                self.cls = cls
                self.xyxy = [[x1, y1, x2, y2]]
                self.conf = [conf]
        
        results = SimulatedResults()
        
        try:
            img_path = self.images[self.current_image_idx]
            label_path = os.path.splitext(img_path)[0] + ".txt"
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    altura, ancho = frame.shape[:2]
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls, x_center, y_center, width, height = map(float, parts)
                            
                            # Convertir coordenadas relativas a píxeles
                            x_center *= ancho
                            y_center *= altura
                            width *= ancho
                            height *= altura
                            
                            # Calcular esquinas
                            x1 = max(0, int(x_center - width / 2))
                            y1 = max(0, int(y_center - height / 2))
                            x2 = min(ancho - 1, int(x_center + width / 2))
                            y2 = min(altura - 1, int(y_center + height / 2))
                            
                            box = SimulatedBox(int(cls), x1, y1, x2, y2)
                            results.boxes.append(box)
        except Exception as e:
            print(f"Error al leer etiquetas: {e}")
            # Generar objetos aleatorios
            for cls in range(3):
                for _ in range(random.randint(1, 3)):
                    x = random.randint(50, SCREEN_WIDTH-50)
                    y = random.randint(50, SCREEN_HEIGHT-50)
                    w = random.randint(30, 60)
                    h = random.randint(30, 60)
                    
                    box = SimulatedBox(cls, x-w//2, y-h//2, x+w//2, y+h//2)
                    results.boxes.append(box)
        
        return results
    
    def process_detections(self, results, frame):
        """Procesa las detecciones y actualiza la lista de objetivos."""
        self.targets = []
        
        if results and hasattr(results, 'boxes'):
            for box in results.boxes:
                # Obtener clase
                cls = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                
                # Solo nos interesan las clases objetivo (insectos)
                if cls not in [0, 1, 2]:  # Mariposa, mariquita, cucaracha
                    continue
                
                # Si no es la clase que estamos siguiendo, ignorar
                if cls != self.target_class:
                    continue
                
                # Obtener coordenadas
                x1, y1, x2, y2 = box.xyxy[0].tolist() if hasattr(box.xyxy[0], 'tolist') else box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calcular centro
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                self.targets.append((cx, cy))
                
                # Dibujar bounding box en el frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLORES[cls], 2)
                
                # Obtener confianza
                conf = float(box.conf[0].item()) if hasattr(box.conf[0], 'item') else float(box.conf[0])
                
                # Añadir etiqueta
                label = f"{NOMBRES_CLASES[cls]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORES[cls], 2)
        
        return frame
    
    def update_laser_position(self):
        """Actualiza la posición del láser según los objetivos detectados."""
        if not self.targets:
            return
        
        # Cambiar de objetivo si ha pasado suficiente tiempo
        current_time = time.time()
        if self.target_pos is None or (current_time - self.last_target_time) > FOLLOW_DELAY:
            self.current_target_idx = (self.current_target_idx + 1) % len(self.targets)
            self.target_pos = self.targets[self.current_target_idx]
            self.last_target_time = current_time
        
        # Mover el láser hacia el objetivo actual
        if self.target_pos:
            # Calcular dirección y distancia
            dx = self.target_pos[0] - self.laser_pos[0]
            dy = self.target_pos[1] - self.laser_pos[1]
            distance = ((dx ** 2) + (dy ** 2)) ** 0.5
            
            # Si estamos cerca del objetivo, considerar que ya llegamos
            if distance < 5:
                self.laser_pos[0] = self.target_pos[0]
                self.laser_pos[1] = self.target_pos[1]
            else:
                # Normalizar dirección y aplicar velocidad
                if distance > 0:
                    dx = dx / distance * min(MAX_LASER_SPEED, distance)
                    dy = dy / distance * min(MAX_LASER_SPEED, distance)
                
                # Actualizar posición
                self.laser_pos[0] += dx
                self.laser_pos[1] += dy
    
    def draw_laser(self, frame):
        """Dibuja el láser en el frame."""
        x = max(0, min(int(self.laser_pos[0]), SCREEN_WIDTH - 1))
        y = max(0, min(int(self.laser_pos[1]), SCREEN_HEIGHT - 1))
        
        cv2.circle(frame, (x, y), LASER_SIZE, LASER_COLOR, -1)
        cv2.circle(frame, (x, y), LASER_SIZE // 2, (255, 255, 255), -1)
        
        return frame
    
    def run(self):
        """Ejecuta la demostración."""
        running = True
        paused = False
        
        while running:
            # Procesar eventos
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_1:
                        self.cambiar_imagen(1)
                    elif event.key == K_2:
                        self.cambiar_imagen(-1)
                    elif event.key == K_a:
                        self.target_class = 0  # Mariposas
                        print("Siguiendo: Mariposas")
                    elif event.key == K_s:
                        self.target_class = 1  # Mariquitas
                        print("Siguiendo: Mariquitas")
                    elif event.key == K_d:
                        self.target_class = 2  # Cucarachas
                        print("Siguiendo: Cucarachas")
                    elif event.key == K_SPACE:
                        paused = not paused
                        print("Simulación " + ("pausada" if paused else "reanudada"))
            
            # Si está pausado, solo actualizar la pantalla
            if paused:
                self.clock.tick(FPS)
                continue
            
            # Obtener imagen actual y detectar objetos
            frame = self.get_current_image()
            results = self.detect_objects(frame)
            frame = self.process_detections(results, frame)
            
            # Actualizar posición del láser
            self.update_laser_position()
            frame = self.draw_laser(frame)
            
            # Mostrar información en pantalla
            clase_actual = NOMBRES_CLASES[self.target_class]
            cv2.putText(frame, f"Siguiendo: {clase_actual}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, "1: Sig. Imagen | 2: Ant. Imagen | A/S/D: Cambiar clase | ESPACIO: Pausa | ESC: Salir", 
                      (10, SCREEN_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Si no estamos usando el modelo real, indicarlo
            if self.model is None:
                cv2.putText(frame, "MODO SIMULACIÓN", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Mostrar nombre de la imagen actual
            if self.images:
                img_name = os.path.basename(self.images[self.current_image_idx])
                cv2.putText(frame, f"Imagen: {img_name}", 
                          (SCREEN_WIDTH - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Mostrar número de objetivos
            cv2.putText(frame, f"Objetivos: {len(self.targets)}", 
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Convertir frame a formato Pygame y mostrar
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pygame_frame = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            self.screen.blit(pygame_frame, (0, 0))
            pygame.display.flip()
            
            # Guardar video
            if self.save_video and self.video_writer:
                self.video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            
            # Controlar FPS
            self.clock.tick(FPS)
        
        # Limpieza
        if self.video_writer:
            self.video_writer.release()
            print(f"Video guardado en la carpeta '{self.video_dir}'")
        
        pygame.quit()

def find_model():
    """Busca el modelo entrenado."""
    for ruta in ['runs/train/model/weights/best.pt', 'best.pt', 'runs/train/train/weights/best.pt']:
        if os.path.exists(ruta):
            return ruta
    
    if os.path.exists('runs/train'):
        for root, _, files in os.walk('runs/train'):
            for file in files:
                if file == 'best.pt':
                    return os.path.join(root, file)
    
    return None

def main():
    """Función principal."""
    # Crear el parser de argumentos
    parser = argparse.ArgumentParser(description='Demostración de Seguimiento Láser')
    parser.add_argument('--model', type=str, default=None, help='Ruta al modelo YOLO entrenado')
    parser.add_argument('--test-dir', type=str, default='test_images', help='Directorio de imágenes de prueba')
    parser.add_argument('--num-images', type=int, default=NUM_TEST_IMAGES, help='Número de imágenes de prueba a generar')
    parser.add_argument('--no-video', action='store_true', help='No grabar video de la simulación')
    parser.add_argument('--video-dir', type=str, default=SIMULACIONES_DIR, help='Directorio para guardar videos')
    parser.add_argument('--generate-only', action='store_true', help='Solo generar imágenes de prueba sin ejecutar la simulación')
    
    args = parser.parse_args()
    
    print("=== DEMOSTRACIÓN DE SEGUIMIENTO LÁSER ===")
    
    # Buscar el modelo si no se especificó
    model_path = args.model
    if model_path is None:
        model_path = find_model()
    
    if model_path:
        print(f"Modelo: {model_path}")
    else:
        print("No se encontró el modelo entrenado. Usando simulación básica.")
    
    # Generar imágenes de prueba si no existen o si se solicita explícitamente
    test_dir = args.test_dir
    if not os.path.exists(test_dir) or len(glob.glob(os.path.join(test_dir, "*.jpg"))) == 0 or args.generate_only:
        print(f"Generando {args.num_images} imágenes de prueba...")
        generator = TestImageGenerator(test_dir)
        generator.generate_images(args.num_images)
        
        if args.generate_only:
            print("Imágenes generadas. Saliendo.")
            return
    
    print("\nControles:")
    print("  1 - Siguiente imagen")
    print("  2 - Imagen anterior")
    print("  A - Seguir mariposas")
    print("  S - Seguir mariquitas")
    print("  D - Seguir cucarachas")
    print("  ESPACIO - Pausar/Reanudar")
    print("  ESC - Salir")
    
    # Iniciar la demostración
    demo = DemoLaser(
        model_path=model_path, 
        test_images_dir=test_dir, 
        save_video=not args.no_video,
        video_dir=args.video_dir
    )
    demo.run()

if __name__ == "__main__":
    main()