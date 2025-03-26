import os
import random
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from tqdm import tqdm
import math
import shutil
import matplotlib.pyplot as plt

class Config:
    def __init__(self):
        # Configuración básica
        self.num_clases = 3
        self.max_width = 640
        self.max_height = 480
        self.num_imagenes = 1000
        self.max_objetos = 20
        self.min_objetos = 10
        self.img_clases = ["mariposa.png", "mariquita.png", "cucaracha.png"]
        self.img_limite = "cruz.png"
        self.img_laser = "laser.png"
        self.train_ratio = 0.8
        
        # Directorios
        self.dir_recursos = "recursos"
        self.dir_dataset = "dataset"
        self.dir_images = os.path.join(self.dir_dataset, "images")
        self.dir_labels = os.path.join(self.dir_dataset, "labels")
        
        # Mapeo de clases
        self.class_mapping = {
            "limite": 0,
            "mariposa": 1,
            "mariquita": 2,
            "cucaracha": 3,
            "laser": 4
        }

def preparar_recursos(config):
    """Prepara las carpetas y recursos necesarios"""
    # Crear directorios
    os.makedirs(config.dir_recursos, exist_ok=True)
    os.makedirs(config.dir_images, exist_ok=True)
    os.makedirs(config.dir_labels, exist_ok=True)
    
    # Verificar recursos
    archivos_necesarios = [config.img_limite, config.img_laser] + config.img_clases
    
    if not os.path.exists(config.dir_recursos):
        print(f"Error: No existe el directorio {config.dir_recursos}")
        return False
    
    archivos_encontrados = os.listdir(config.dir_recursos)
    faltantes = [f for f in archivos_necesarios if f not in archivos_encontrados]
    
    if faltantes:
        print(f"Error: Faltan archivos: {', '.join(faltantes)}")
        print(f"Asegúrate de colocar todos los archivos necesarios en la carpeta '{config.dir_recursos}'")
        return False
    
    # Verificar transparencia
    for archivo in archivos_necesarios:
        try:
            img = Image.open(os.path.join(config.dir_recursos, archivo))
            if 'A' not in img.getbands():
                print(f"Advertencia: {archivo} no tiene transparencia (alpha)")
        except Exception as e:
            print(f"Error al cargar {archivo}: {e}")
            return False
    
    print("Recursos verificados correctamente")
    return True

def generar_fondo_aleatorio(width, height):
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

def aplicar_transformaciones(imagen):
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

def generar_dataset(config):
    """Genera el dataset completo de imágenes y etiquetas"""
    print(f"Generando dataset de {config.num_imagenes} imágenes...")
    
    # Cargar imágenes de recursos
    img_limite = Image.open(os.path.join(config.dir_recursos, config.img_limite)).convert("RGBA")
    img_laser = Image.open(os.path.join(config.dir_recursos, config.img_laser)).convert("RGBA")
    img_clases = [Image.open(os.path.join(config.dir_recursos, clase)).convert("RGBA") 
                 for clase in config.img_clases]
    
    # Crear archivo classes.txt
    with open(os.path.join(config.dir_dataset, "classes.txt"), "w") as f:
        f.write("limite\n")
        for clase in config.img_clases:
            f.write(f"{os.path.splitext(clase)[0]}\n")
        f.write("laser\n")
    
    # Generar imágenes
    imagenes_paths = []
    
    for idx in tqdm(range(config.num_imagenes)):
        # Generar fondo aleatorio
        fondo = generar_fondo_aleatorio(config.max_width, config.max_height)
        labels = []
        
        # Colocar marcas de límite en las esquinas
        esquinas = [
            (0, 0),
            (config.max_width - img_limite.width, 0),
            (0, config.max_height - img_limite.height),
            (config.max_width - img_limite.width, config.max_height - img_limite.height)
        ]
        
        for esquina in esquinas:
            fondo.paste(img_limite, esquina, img_limite)
            
            # Etiqueta YOLO: clase x_center y_center width height
            x_center = (esquina[0] + img_limite.width / 2) / config.max_width
            y_center = (esquina[1] + img_limite.height / 2) / config.max_height
            width = img_limite.width / config.max_width
            height = img_limite.height / config.max_height
            labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Crear cuadrícula para distribuir objetos
        num_objetos = random.randint(config.min_objetos, config.max_objetos)
        grid_size = math.ceil(math.sqrt(num_objetos * 1.5))
        cell_width = config.max_width / grid_size
        cell_height = config.max_height / grid_size
        
        # Mezclar celdas disponibles
        celdas = [(r, c) for r in range(grid_size) for c in range(grid_size)]
        random.shuffle(celdas)
        
        # Colocar objetos
        objetos_colocados = 0
        
        while objetos_colocados < num_objetos and celdas:
            row, col = celdas.pop()
            
            # Seleccionar clase aleatoria
            clase_idx = random.randint(0, len(img_clases) - 1)
            img_objeto = img_clases[clase_idx].copy()
            
            # Aplicar transformaciones
            img_objeto = aplicar_transformaciones(img_objeto)
            
            # Calcular posición
            pos_x = int(col * cell_width + random.uniform(0, cell_width - img_objeto.width))
            pos_y = int(row * cell_height + random.uniform(0, cell_height - img_objeto.height))
            
            # Asegurar que está dentro de los límites
            pos_x = max(0, min(pos_x, config.max_width - img_objeto.width))
            pos_y = max(0, min(pos_y, config.max_height - img_objeto.height))
            
            # Pegar objeto
            fondo.paste(img_objeto, (pos_x, pos_y), img_objeto)
            
            # Calcular etiqueta YOLO
            x_center = (pos_x + img_objeto.width / 2) / config.max_width
            y_center = (pos_y + img_objeto.height / 2) / config.max_height
            width = img_objeto.width / config.max_width
            height = img_objeto.height / config.max_height
            
            # Clase + 1 (limite=0, mariposa=1, etc.)
            labels.append(f"{clase_idx + 1} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            objetos_colocados += 1
            
            # Agregar puntero láser cerca de algunos objetos (30% probabilidad)
            if random.random() < 0.3:
                laser_copy = img_laser.copy()
                
                # Escalar láser aleatoriamente
                if random.random() > 0.5:
                    scale_factor = random.uniform(0.7, 1.3)
                    laser_width = int(laser_copy.width * scale_factor)
                    laser_height = int(laser_copy.height * scale_factor)
                    laser_copy = laser_copy.resize((laser_width, laser_height), Image.LANCZOS)
                
                # Posicionar láser cerca del objeto
                pos_laser_x = pos_x + random.randint(-20, img_objeto.width + 20)
                pos_laser_y = pos_y + random.randint(-20, img_objeto.height + 20)
                
                # Asegurar que está dentro de los límites
                pos_laser_x = max(0, min(pos_laser_x, config.max_width - laser_copy.width))
                pos_laser_y = max(0, min(pos_laser_y, config.max_height - laser_copy.height))
                
                # Pegar láser
                fondo.paste(laser_copy, (pos_laser_x, pos_laser_y), laser_copy)
                
                # Etiqueta para el láser (clase 4)
                x_center_laser = (pos_laser_x + laser_copy.width / 2) / config.max_width
                y_center_laser = (pos_laser_y + laser_copy.height / 2) / config.max_height
                width_laser = laser_copy.width / config.max_width
                height_laser = laser_copy.height / config.max_height
                
                labels.append(f"4 {x_center_laser:.6f} {y_center_laser:.6f} {width_laser:.6f} {height_laser:.6f}")
        
        # Guardar imagen
        img_path = os.path.join(config.dir_images, f"img{idx+1:04d}.jpg")
        fondo.convert("RGB").save(img_path, quality=95)
        
        # Guardar etiquetas
        label_path = os.path.join(config.dir_labels, f"img{idx+1:04d}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(labels))
        
        imagenes_paths.append(img_path)
    
    # Dividir en train/val
    dividir_dataset(config, imagenes_paths)
    
    print(f"Dataset generado: {config.num_imagenes} imágenes creadas.")
    return True

def dividir_dataset(config, image_paths):
    """Divide el dataset en conjuntos de entrenamiento y validación"""
    print("Dividiendo dataset en entrenamiento/validación...")
    
    # Mezclar aleatoriamente las imágenes
    random.shuffle(image_paths)
    split_idx = int(len(image_paths) * config.train_ratio)
    
    train_images = image_paths[:split_idx]
    val_images = image_paths[split_idx:]
    
    # Crear archivos train.txt y val.txt
    with open(os.path.join(config.dir_dataset, "train.txt"), "w") as f:
        f.write("\n".join(train_images))
    
    with open(os.path.join(config.dir_dataset, "val.txt"), "w") as f:
        f.write("\n".join(val_images))
    
    # Crear archivo data.yaml para YOLO
    class_names = ["limite"] + [os.path.splitext(cls)[0] for cls in config.img_clases] + ["laser"]
    yaml_content = f"""train: {os.path.join(config.dir_dataset, 'train.txt')}
val: {os.path.join(config.dir_dataset, 'val.txt')}
nc: {len(class_names)}
names: {class_names}
"""
    
    with open(os.path.join(config.dir_dataset, "data.yaml"), "w") as f:
        f.write(yaml_content)
    
    print(f"Dataset dividido: {len(train_images)} imágenes de entrenamiento, {len(val_images)} imágenes de validación")

def visualizar_muestra(config, num_samples=5):
    """Visualiza algunas imágenes de muestra con sus bounding boxes"""
    print("Visualizando ejemplos del dataset generado...")
    
    image_files = os.listdir(config.dir_images)
    random.shuffle(image_files)
    samples = image_files[:num_samples]
    
    # Colores para las diferentes clases
    colors = {
        0: (255, 0, 0),    # Límite - Rojo
        1: (0, 255, 0),    # Mariposa - Verde
        2: (0, 0, 255),    # Mariquita - Azul
        3: (255, 255, 0),  # Cucaracha - Amarillo
        4: (0, 255, 255)   # Láser - Cian
    }
    
    class_names = ["limite"] + [os.path.splitext(cls)[0] for cls in config.img_clases] + ["laser"]
    
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i, sample in enumerate(samples):
        img_path = os.path.join(config.dir_images, sample)
        label_path = os.path.join(config.dir_labels, sample.replace('.jpg', '.txt'))
        
        # Cargar imagen
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        
        # Cargar etiquetas
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = f.readlines()
        
        # Dibujar bounding boxes
        for label in labels:
            parts = label.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                w = float(parts[3]) * width
                h = float(parts[4]) * height
                
                # Calcular coordenadas
                x1 = int(x_center - w/2)
                y1 = int(y_center - h/2)
                x2 = int(x_center + w/2)
                y2 = int(y_center + h/2)
                
                # Dibujar rectángulo
                cv2.rectangle(img, (x1, y1), (x2, y2), colors.get(class_id, (255, 255, 255)), 2)
                
                # Añadir etiqueta
                cv2.putText(img, class_names[class_id], (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(class_id, (255, 255, 255)), 1)
        
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(img)
        plt.title(f'Muestra {i+1}: {sample}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Función principal para generar el dataset"""
    # Crear configuración
    config = Config()
    
    print("=== GENERADOR DE DATASET PARA YOLO ===")
    print(f"Asegúrate de colocar estos archivos en la carpeta '{config.dir_recursos}':")
    print(f"- Imágenes de clase: {', '.join(config.img_clases)}")
    print(f"- Imagen límite: {config.img_limite}")
    print(f"- Imagen láser: {config.img_laser}")
    print()
    
    # Preparar recursos
    if not preparar_recursos(config):
        print("Error preparando los recursos. Abortando.")
        return
    
    # Generar dataset
    if generar_dataset(config):
        # Visualizar algunas muestras
        visualizar_muestra(config)
        
        # Comprimir dataset (opcional)
        comprimir = input("¿Deseas comprimir el dataset en un archivo ZIP? (s/n): ").lower() == 's'
        if comprimir:
            print("Comprimiendo dataset...")
            shutil.make_archive("dataset", "zip", config.dir_dataset)
            print("Dataset comprimido como 'dataset.zip'")

if __name__ == "__main__":
    main()