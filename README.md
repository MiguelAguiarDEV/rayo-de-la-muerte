# Sistema de Localización y Seguimiento Automático con Láser

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Objetivos del Proyecto](#objetivos-del-proyecto)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Arquitectura del Sistema](#arquitectura-del-sistema)
5. [Generación del Dataset](#generación-del-dataset)
6. [Modelo YOLOv8](#modelo-yolov8)
7. [Sistema de Simulación](#sistema-de-simulación)
8. [Resultados y Evaluación](#resultados-y-evaluación)
9. [Conclusiones](#conclusiones)

## Introducción

Este proyecto implementa un sistema de visión por computadora que localiza y sigue automáticamente objetos específicos utilizando un puntero láser. El sistema es capaz de detectar tres clases de objetos (mariposas, mariquitas y cucarachas) en tiempo real, y dirigir un puntero láser para apuntar secuencialmente a cada uno de los objetivos detectados.

La solución combina técnicas de visión por computadora, aprendizaje profundo con YOLO (You Only Look Once), y proporciona una simulación completa del sistema de seguimiento, así como las bases para una futura implementación con hardware real.

## Objetivos del Proyecto

- Desarrollar un sistema de localización y seguimiento automático basado en visión por computadora
- Implementar un modelo de detección de objetos usando YOLOv8
- Crear un dataset personalizado con imágenes de mariposas, mariquitas y cucarachas
- Proporcionar una simulación visual del comportamiento del sistema
- Establecer las bases para una futura implementación con hardware físico

## Estructura del Proyecto

```
sistema-seguimiento-laser/
│
├── dataset/                  # Dataset generado para entrenamiento
│   ├── images/               # Imágenes para entrenamiento
│   ├── labels/               # Etiquetas en formato YOLO
│   └── classes.txt           # Definición de clases
│
├── recursos/                 # Imágenes originales para generar el dataset
│   ├── mariposa.png
│   ├── mariquita.png
│   ├── cucaracha.png
│   ├── cruz.png              # Marcador de límites
│   └── laser.png             # Imagen del láser
│
├── test_images/              # Imágenes de prueba para la simulación
│   ├── test_image_1.jpg
│   ├── test_image_1.txt
│   └── ...
│
├── simulaciones/             # Videos grabados de la simulación
│   └── demo_laser_*.avi
│
├── runs/                     # Resultados del entrenamiento
│   └── train/
│       └── model/
│           ├── weights/      # Modelos entrenados (best.pt, last.pt)
│           └── results.csv   # Métricas de entrenamiento
│
├── generar_dataset.py        # Script para generar el dataset
├── entrenar_modelo.py        # Script para entrenar el modelo
├── demo_modelo_laser.py      # Script de simulación
│
└── requirements.txt          # Dependencias del proyecto
```

## Arquitectura del Sistema

El sistema se divide en tres componentes principales:

1. **Generación del dataset**: Crea imágenes sintéticas con objetos (mariposas, mariquitas, cucarachas) y sus etiquetas.
2. **Entrenamiento del modelo**: Entrena un modelo YOLOv8 con el dataset generado.
3. **Sistema de seguimiento**: Utiliza el modelo entrenado para detectar objetos y simular/controlar el puntero láser.

El flujo de datos completo es el siguiente:

```
[Imágenes Originales] → [Generador de Dataset] → [Dataset Etiquetado] → [Entrenamiento YOLO] → [Modelo Entrenado]
→ [Simulador/Sistema Real] → [Detección de Objetos] → [Control del Láser] → [Seguimiento Visual]
```

## Generación del Dataset

### Proceso

El generador de dataset crea imágenes sintéticas que contienen:

- Objetos de las tres clases (mariposas, mariquitas, cucarachas)
- Marcadores de límites en las esquinas
- Opcionalmente, imágenes del puntero láser

La generación sigue estos pasos:

1. Crear fondos aleatorios (sólidos, gradientes, texturas, etc.)
2. Distribuir objetos aleatoriamente en una cuadrícula
3. Aplicar transformaciones aleatorias a cada objeto:
   - Rotaciones (0-360 grados)
   - Escalado (50%-150% del tamaño original)
   - Volteos horizontales y verticales
   - Ajustes de brillo y contraste
4. Colocar marcadores de límite en las esquinas
5. Opcionalmente añadir imágenes del láser cerca de algunos objetos
6. Generar etiquetas en formato YOLO

### Formato de etiquetas YOLO

Las etiquetas se guardan en archivos .txt con el mismo nombre que las imágenes, donde cada línea representa un objeto:

```
<clase> <x_center> <y_center> <width> <height>
```

Donde:

- `clase` es el ID numérico de la clase (0=mariposa, 1=mariquita, 2=cucaracha, 3=limite, 4=laser)
- `x_center`, `y_center` son las coordenadas del centro del objeto (normalizadas de 0 a 1)
- `width`, `height` son el ancho y alto del objeto (normalizados de 0 a 1)

## Modelo YOLOv8

### Arquitectura del Modelo

YOLOv8 es un modelo de una sola etapa (one-stage detector) para detección de objetos. Su arquitectura consta de:

1. **Backbone**: Red neuronal (generalmente basada en CSPDarknet) que extrae características de diferentes niveles de la imagen.
2. **Neck**: Módulos como FPN (Feature Pyramid Network) o PANet que combinan características de diferentes escalas.
3. **Head**: Capas finales que predicen las clases y bounding boxes de los objetos.

![Arquitectura YOLOv8](https://user-images.githubusercontent.com/26833433/202812368-67e08ed2-532a-46e1-b9f8-d9eb3c294853.png)

### Datos de Entrada y Salida

**Entrada**:

- Imágenes de tamaño 640x480 píxeles (RGB)
- Normalizadas con valores entre 0 y 1

**Salida**:

- Lista de detecciones, donde cada detección contiene:
  - Clase del objeto (mariposa, mariquita, cucaracha, límite, láser)
  - Coordenadas del bounding box (x1, y1, x2, y2)
  - Puntuación de confianza (0-1)

### Capas Principales

1. **Conv + Batch Normalization + SiLU**: Bloques convolutivos básicos
2. **C2f**: Bloques de cuello de botella (bottleneck) con conexiones residuales
3. **SPPF**: Spatial Pyramid Pooling - Fast para capturar contextos más amplios
4. **Capas de Detección**: Predicen las coordenadas y clases para cada ancla

### Entrenamiento

El modelo se entrena con los siguientes parámetros:

- **Optimizer**: AdamW
- **Loss**: CIoU loss (para bounding boxes) + BCE loss (para clasificación)
- **Learning rate**: 0.01 con programación coseno
- **Epochs**: 50-100
- **Batch size**: 16
- **Augmentaciones**: Mosaic, RandomAffine, ColorJitter, etc.

## Sistema de Simulación

### Arquitectura de la Simulación

La simulación del sistema implementa:

1. **Cargador de Imágenes**: Carga imágenes de prueba desde el directorio especificado.
2. **Detector de Objetos**: Utiliza el modelo YOLOv8 para detectar objetos de interés.
3. **Controlador del Láser**: Simula el movimiento del puntero láser entre objetivos.
4. **Visualizador**: Muestra en tiempo real la simulación con Pygame.
5. **Grabador de Video**: Guarda la simulación en un archivo de video.

### Flujo de Detección

1. Cargar imagen desde el directorio de prueba
2. Pasar la imagen al modelo YOLOv8 para detección
3. Filtrar las detecciones por clase seleccionada (mariposa, mariquita o cucaracha)
4. Calcular las coordenadas centrales de cada objeto detectado
5. Actualizar la lista de objetivos

### Algoritmo de Seguimiento

El algoritmo implementa un seguimiento secuencial simple:

1. Para cada frame:
   - Detectar todos los objetos de la clase seleccionada
   - Si es necesario cambiar de objetivo (tiempo expirado), seleccionar el siguiente en la lista
   - Calcular la dirección y distancia al objetivo actual
   - Mover el láser hacia el objetivo con velocidad limitada
   - Si está suficientemente cerca, marcar como "en objetivo" y iniciar temporizador
   - Después de un tiempo predefinido (FOLLOW_DELAY), pasar al siguiente objetivo

### Interfaz de Usuario

La simulación proporciona una interfaz visual con:

- Visualización en tiempo real de las detecciones (bounding boxes)
- Visualización del puntero láser
- Información sobre la clase objetivo actual
- Controles para cambiar de clase objetivo
- Navegación entre diferentes imágenes de prueba

## Resultados y Evaluación

### Métricas del Modelo

El modelo YOLOv8 entrenado alcanzó excelentes resultados:

| Clase      | Precisión | Recall    | mAP50     | mAP50-95  |
| ---------- | --------- | --------- | --------- | --------- |
| limite     | 0.999     | 1.000     | 0.995     | 0.994     |
| mariposa   | 1.000     | 0.999     | 0.995     | 0.982     |
| mariquita  | 1.000     | 1.000     | 0.995     | 0.974     |
| cucaracha  | 1.000     | 0.999     | 0.995     | 0.971     |
| laser      | 0.999     | 0.999     | 0.995     | 0.994     |
| **Global** | **1.000** | **0.999** | **0.995** | **0.983** |

### Rendimiento de la Simulación

- **Tiempo de Inferencia**: ~8ms por imagen
- **FPS**: 30 frames por segundo (limitado por configuración)
- **Precisión de Seguimiento**: Movimiento suave entre objetivos
- **Tiempo de Permanencia**: 2 segundos por objetivo

## Conclusiones

Este proyecto ha demostrado con éxito la viabilidad de un sistema de seguimiento láser basado en visión por computadora. El enfoque adoptado, que combina técnicas de generación de datasets sintéticos y modelos de detección de objetos de última generación, ha permitido desarrollar un sistema con alta precisión en la detección y seguimiento de objetos específicos.

La simulación proporciona una plataforma sólida para entender y visualizar el comportamiento del sistema antes de su implementación física, y el código está estructurado de manera modular para facilitar la transición a hardware real.

Las futuras mejoras podrían enfocarse en:

1. Implementación con hardware real
2. Mejoras en la velocidad y suavidad del seguimiento
3. Ampliación a más clases de objetos
4. Desarrollo de algoritmos más sofisticados para priorizar objetivos
5. Integración con otras tecnologías de visión por computadora
