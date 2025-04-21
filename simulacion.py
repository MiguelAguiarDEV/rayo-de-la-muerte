#!/usr/bin/env python3
import os
import time
import glob
import datetime
import random
import math

import cv2
import numpy as np
import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_1, K_2, K_a, K_s, K_d, K_SPACE
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

# -------------------------------
# Configuración general
# -------------------------------
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
FPS = 30

# Parámetros del láser
LASER_SIZE = 10            # Radio del punto láser
LASER_COLOR = (0, 255, 0)  # Verde
MAX_LASER_SPEED = 15       # Pixeles por frame
FOLLOW_DELAY = 2           # Segundos antes de cambiar de objetivo

# Directorios
TEST_IMAGES_DIR = "test_images"
VIDEO_DIR       = "simulaciones"
RECURSOS_DIR    = "recursos"

# Recursos para generación de prueba
IMG_CLASSES = ["mariposa.png", "mariquita.png", "cucaracha.png"]
IMG_LIMITE  = "cruz.png"
IMG_LASER   = "laser.png"

# Clases y colores (BGR)
CLASS_NAMES = {
    0: "limite",
    1: "mariposa",
    2: "mariquita",
    3: "cucaracha",
    4: "laser",
}
CLASS_COLORS = {
    0: (128,   0, 128),  # Límite - Morado
    1: (255,   0,   0),  # Mariposa - Azul
    2: (  0,   0, 255),  # Mariquita - Rojo
    3: ( 42,  42, 165),  # Cucaracha - Marrón
    4: (  0, 255,   0),  # Láser - Verde
}

# -------------------------------
# Generación de imágenes de prueba
# -------------------------------
def generar_fondo_aleatorio(width, height):
    tipo = random.choice(["solido", "gradiente", "ruido", "textura"])
    if tipo == "solido":
        color = tuple(random.randint(0,255) for _ in range(3))
        fondo = Image.new("RGB", (width, height), color)
    elif tipo == "gradiente":
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        c1 = np.array([random.randint(0,255) for _ in range(3)])
        c2 = np.array([random.randint(0,255) for _ in range(3)])
        for y in range(height):
            f = y / height
            arr[y, :] = (c1 * (1 - f) + c2 * f).astype(np.uint8)
        fondo = Image.fromarray(arr)
    elif tipo == "ruido":
        arr = np.random.randint(0,256, (height, width, 3), dtype=np.uint8)
        if random.random() > 0.5:
            arr = cv2.GaussianBlur(arr, (0, 0), random.uniform(1,5))
        fondo = Image.fromarray(arr)
    else:
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        scale = random.randint(5,30)
        for y in range(height):
            for x in range(width):
                v = int(127 + 127 * math.sin(x/scale) * math.sin(y/scale))
                arr[y, x] = [v, v, v]
        col = np.array([random.randint(0,255) for _ in range(3)])
        for i in range(3):
            arr[:,:,i] = (arr[:,:,0] * col[i] / 255).astype(np.uint8)
        fondo = Image.fromarray(arr)
    if random.random() > 0.5:
        fondo = fondo.filter(ImageFilter.GaussianBlur(random.uniform(0.5,2)))
    if random.random() > 0.7:
        ruido = np.random.randint(0,25, (height, width, 3), dtype=np.uint8)
        fondo = Image.blend(fondo, Image.fromarray(ruido), 0.1)
    return fondo

def aplicar_transformaciones(imagen):
    factor = random.uniform(0.5,1.5)
    w_new = int(imagen.width * factor)
    h_new = int(imagen.height * factor)
    imagen = imagen.resize((w_new, h_new), Image.LANCZOS)
    imagen = imagen.rotate(random.uniform(0,360), expand=True, resample=Image.BICUBIC)
    if random.random() > 0.5:
        imagen = ImageOps.mirror(imagen)
    if random.random() > 0.5:
        imagen = ImageOps.flip(imagen)
    imagen = ImageEnhance.Brightness(imagen).enhance(random.uniform(0.8,1.2))
    imagen = ImageEnhance.Contrast(imagen).enhance(random.uniform(0.8,1.2))
    return imagen

def generar_test_images(n=5, min_obj=5, max_obj=15):
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    recursos = {
        'clases': [Image.open(os.path.join(RECURSOS_DIR, f)).convert("RGBA") for f in IMG_CLASSES],
        'limite': Image.open(os.path.join(RECURSOS_DIR, IMG_LIMITE)).convert("RGBA"),
    }
    for i in range(n):
        fondo = generar_fondo_aleatorio(SCREEN_WIDTH, SCREEN_HEIGHT)
        labels = []
        esquinas = [
            (0, 0),
            (SCREEN_WIDTH - recursos['limite'].width, 0),
            (0, SCREEN_HEIGHT - recursos['limite'].height),
            (SCREEN_WIDTH - recursos['limite'].width, SCREEN_HEIGHT - recursos['limite'].height)
        ]
        for x, y in esquinas:
            fondo.paste(recursos['limite'], (x, y), recursos['limite'])
            cx = (x + recursos['limite'].width/2) / SCREEN_WIDTH
            cy = (y + recursos['limite'].height/2) / SCREEN_HEIGHT
            w  = recursos['limite'].width / SCREEN_WIDTH
            h  = recursos['limite'].height / SCREEN_HEIGHT
            labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        count = random.randint(min_obj, max_obj)
        grid = int(math.ceil(math.sqrt(count * 1.5)))
        cells = [(r, c) for r in range(grid) for c in range(grid)]
        random.shuffle(cells)
        placed = 0
        while placed < count and cells:
            r, c = cells.pop()
            cls_idx = random.randrange(len(recursos['clases']))
            obj = recursos['clases'][cls_idx].copy()
            obj = aplicar_transformaciones(obj)
            cell_w = SCREEN_WIDTH / grid
            cell_h = SCREEN_HEIGHT / grid
            x = int(c * cell_w + random.uniform(0, cell_w - obj.width))
            y = int(r * cell_h + random.uniform(0, cell_h - obj.height))
            x = max(0, min(x, SCREEN_WIDTH - obj.width))
            y = max(0, min(y, SCREEN_HEIGHT - obj.height))
            fondo.paste(obj, (x, y), obj)
            cx = (x + obj.width/2) / SCREEN_WIDTH
            cy = (y + obj.height/2) / SCREEN_HEIGHT
            w  = obj.width / SCREEN_WIDTH
            h  = obj.height / SCREEN_HEIGHT
            labels.append(f"{cls_idx+1} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            placed += 1
        img_path = os.path.join(TEST_IMAGES_DIR, f"test_{i+1}.jpg")
        txt_path = img_path.replace('.jpg', '.txt')
        fondo.convert("RGB").save(img_path, quality=95)
        with open(txt_path, 'w') as f:
            f.write("\n".join(labels))
    print(f"Generadas {n} imágenes de prueba en '{TEST_IMAGES_DIR}/'")

# -------------------------------
# Clase de simulación
# -------------------------------
class DemoLaser:
    def __init__(self, images_dir=TEST_IMAGES_DIR, video_dir=VIDEO_DIR):
        os.makedirs(video_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(video_dir, f"demo_laser_{ts}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(path, fourcc, 20.0, (SCREEN_WIDTH, SCREEN_HEIGHT))

        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Simulación Láser")
        self.clock = pygame.time.Clock()

        self.images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        if not self.images:
            raise FileNotFoundError(f"No hay imágenes en '{images_dir}'.")
        self.idx     = 0
        self.laser   = np.array([SCREEN_WIDTH/2, SCREEN_HEIGHT/2], dtype=float)
        self.targets = []
        self.ti      = 0
        self.last    = time.time()
        self.cls     = 1  # mariposa por defecto

    def load(self):
        img = cv2.imread(self.images[self.idx])
        return cv2.resize(img, (SCREEN_WIDTH, SCREEN_HEIGHT))

    def read_lbl(self):
        p = self.images[self.idx].replace('.jpg','.txt')
        det = []
        if os.path.exists(p):
            for line in open(p):
                c, xc, yc, ww, hh = map(float, line.split())
                if int(c) == self.cls:
                    det.append((int(c), xc, yc, ww, hh))
        return det

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.targets = []
        for c, xc, yc, ww, hh in self.read_lbl():
            x1 = int((xc - ww/2) * w)
            y1 = int((yc - hh/2) * h)
            x2 = int((xc + ww/2) * w)
            y2 = int((yc + hh/2) * h)
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            self.targets.append(np.array([cx, cy], dtype=float))
            col = CLASS_COLORS[c]
            nm  = CLASS_NAMES[c]
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, nm, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        if not self.targets:
            self.ti = 0
        else:
            self.ti %= len(self.targets)
        return frame

    def update(self):
        if not self.targets:
            return
        now = time.time()
        if now - self.last > FOLLOW_DELAY:
            self.ti = (self.ti + 1) % len(self.targets)
            self.last = now
        tgt = self.targets[self.ti]
        vec = tgt - self.laser
        dist = np.linalg.norm(vec)
        if dist > 1:
            step = vec/dist * min(MAX_LASER_SPEED, dist)
            self.laser += step

    def draw_laser(self, frame):
        x, y = map(int, self.laser)
        cv2.circle(frame, (x, y), LASER_SIZE, LASER_COLOR, -1)
        return frame

    def hud(self, frame):
        lines = [
            f"Siguiendo: {CLASS_NAMES[self.cls]}",
            f"Objetivos: {len(self.targets)}",
            "1:Next 2:Prev A/S/D:Clase SPACE:Pausa ESC:Salir"
        ]
        for i, t in enumerate(lines):
            cv2.putText(frame, t, (10, 30 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return frame

    def run(self):
        paused = False
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == QUIT:
                    running = False
                elif e.type == KEYDOWN:
                    if e.key == K_ESCAPE:
                        running = False
                    elif e.key == K_1:
                        self.idx = (self.idx + 1) % len(self.images)
                    elif e.key == K_2:
                        self.idx = (self.idx - 1) % len(self.images)
                    elif e.key in (K_a, K_s, K_d):
                        old = self.cls
                        self.cls = {K_a:1, K_s:2, K_d:3}[e.key]
                        if self.cls != old:
                            self.targets = []
                            self.ti      = 0
                            self.last    = time.time()
                    elif e.key == K_SPACE:
                        paused = not paused

            if not paused:
                frame = self.load()
                frame = self.detect(frame)
                self.update()
                frame = self.draw_laser(frame)
                frame = self.hud(frame)
                surf = pygame.surfarray.make_surface(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).swapaxes(0,1)
                )
                self.screen.blit(surf, (0,0))
                pygame.display.flip()
                self.writer.write(frame)
                self.clock.tick(FPS)

        self.writer.release()
        pygame.quit()

if __name__ == "__main__":
    # Generar nuevas imágenes de prueba cada vez que se ejecuta
    generar_test_images(5)
    # Iniciar simulación
    demo = DemoLaser()
    demo.run()
