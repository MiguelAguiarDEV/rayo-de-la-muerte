import os
import glob
import yaml
import random
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

def preparar_yolo_dataset(dataset_dir, output_dir="yolo_dataset", train_ratio=0.8):
    """Prepara la estructura de directorios para entrenar YOLO"""
    print(f"Preparando estructura YOLO en {output_dir}...")
    
    # Crear directorios
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)
    
    # Obtener imágenes
    images_dir = os.path.join(dataset_dir, "images")
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
    
    # Dividir dataset
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"División: {len(train_images)} para entrenamiento, {len(val_images)} para validación")
    
    # Copiar archivos de entrenamiento
    for img_path in tqdm(train_images, desc="Copiando archivos de entrenamiento"):
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(dataset_dir, "labels", label_name)
        
        # Copiar imagen
        shutil.copy(img_path, os.path.join(output_dir, "images", "train", img_name))
        
        # Copiar etiqueta si existe
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(output_dir, "labels", "train", label_name))
    
    # Copiar archivos de validación
    for img_path in tqdm(val_images, desc="Copiando archivos de validación"):
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(dataset_dir, "labels", label_name)
        
        # Copiar imagen
        shutil.copy(img_path, os.path.join(output_dir, "images", "val", img_name))
        
        # Copiar etiqueta si existe
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(output_dir, "labels", "val", label_name))
    
    # Leer clases o usar valores por defecto
    classes_file = os.path.join(dataset_dir, "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
    else:
        classes = ["limite", "mariposa", "mariquita", "cucaracha", "laser"]
    
    # Crear archivo YAML
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(classes)}
    }
    
    # Guardar YAML
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    print(f"Dataset YOLO preparado en {output_dir}")
    return yaml_path

def entrenar_modelo(yaml_path, epochs=50, img_size=640, batch_size=16, model_size="n", save_dir="runs/train"):
    """Entrena un modelo YOLOv8"""
    print("Iniciando entrenamiento...")
    
    # Seleccionar modelo según tamaño
    model_name = f"yolov8{model_size}.pt"
    
    # Cargar modelo preentrenado
    model = YOLO(model_name)
    
    # Configurar ruta para guardar
    absolute_save_dir = os.path.join(os.getcwd(), save_dir)
    os.makedirs(absolute_save_dir, exist_ok=True)
    
    # Entrenar modelo
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=20,  # Early stopping
        save=True,
        project=absolute_save_dir,
        name="model"
    )
    
    # Verificar archivos guardados
    best_model_path = os.path.join(absolute_save_dir, "model", "weights", "best.pt")
    last_model_path = os.path.join(absolute_save_dir, "model", "weights", "last.pt")
    
    if os.path.exists(best_model_path):
        print(f"Mejor modelo guardado en: {best_model_path}")
    else:
        print("Advertencia: No se encontró best.pt")
    
    if os.path.exists(last_model_path):
        print(f"Último modelo guardado en: {last_model_path}")
    
    print(f"Entrenamiento completado en {absolute_save_dir}/model")
    return results, model, best_model_path

def evaluar_modelo(model, yaml_path):
    """Evalúa el modelo entrenado"""
    print("Evaluando modelo...")
    results = model.val(data=yaml_path)
    
    # Mostrar métricas
    print("Resultados de la evaluación:")
    if hasattr(results, 'box'):
        if hasattr(results.box, 'map'):
            print(f"mAP50-95: {results.box.map:.6f}")
        if hasattr(results.box, 'map50'):
            print(f"mAP50: {results.box.map50:.6f}")
    
    return results

def visualizar_resultados(results_path):
    """Visualiza gráficos de los resultados del entrenamiento"""
    try:
        import pandas as pd
        
        results_csv = os.path.join(results_path, "results.csv")
        if os.path.exists(results_csv):
            results = pd.read_csv(results_csv)
            
            plt.figure(figsize=(15, 10))
            
            # Pérdidas
            plt.subplot(2, 2, 1)
            for col in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']:
                if col in results.columns:
                    plt.plot(results['epoch'], results[col], label=col.split('/')[-1])
            plt.title('Pérdidas de entrenamiento')
            plt.xlabel('Época')
            plt.ylabel('Pérdida')
            plt.legend()
            
            # mAP50
            plt.subplot(2, 2, 2)
            if 'metrics/mAP50(B)' in results.columns:
                plt.plot(results['epoch'], results['metrics/mAP50(B)'])
                plt.title('mAP50')
                plt.xlabel('Época')
            
            # mAP50-95
            plt.subplot(2, 2, 3)
            if 'metrics/mAP50-95(B)' in results.columns:
                plt.plot(results['epoch'], results['metrics/mAP50-95(B)'])
                plt.title('mAP50-95')
                plt.xlabel('Época')
            
            # Precision y Recall
            plt.subplot(2, 2, 4)
            for col, lbl in [('metrics/precision(B)', 'Precisión'), ('metrics/recall(B)', 'Recall')]:
                if col in results.columns:
                    plt.plot(results['epoch'], results[col], label=lbl)
            plt.title('Precisión y Recall')
            plt.xlabel('Época')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_path, "metricas_entrenamiento.png"))
            plt.show()
            
            print(f"Gráficos guardados en {os.path.join(results_path, 'metricas_entrenamiento.png')}")
        else:
            print(f"No se encontró archivo de resultados en {results_csv}")
    except Exception as e:
        print(f"Error al visualizar resultados: {e}")

def probar_modelo(model_path, test_images_dir, conf_threshold=0.25):
    """Ejecuta inferencia en imágenes de prueba"""
    print(f"Probando modelo en {test_images_dir}...")
    
    # Cargar modelo
    model = YOLO(model_path)
    
    # Obtener imágenes
    test_images = glob.glob(os.path.join(test_images_dir, "*.jpg"))
    
    if not test_images:
        print("No se encontraron imágenes de prueba.")
        return
    
    # Seleccionar muestra aleatoria
    test_samples = random.sample(test_images, min(4, len(test_images)))
    
    plt.figure(figsize=(16, 12))
    
    for i, img_path in enumerate(test_samples):
        # Predicción
        results = model.predict(img_path, conf=conf_threshold, save=True)
        
        # Mostrar imagen con predicciones
        img_pred = results[0].plot()
        
        plt.subplot(2, 2, i+1)
        plt.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
        plt.title(f"Predicción {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("predicciones_muestra.png")
    plt.show()
    
    print("Predicciones guardadas como 'predicciones_muestra.png'")

def main():
    """Función principal"""
    print("=== ENTRENAMIENTO DE MODELO YOLO ===")
    
    # Configurar rutas
    dataset_dir = input("Ruta al directorio del dataset (por defecto: 'dataset'): ") or "dataset"
    yolo_dataset_dir = input("Ruta para el dataset YOLO (por defecto: 'yolo_dataset'): ") or "yolo_dataset"
    modelo_save_dir = input("Ruta para guardar resultados (por defecto: 'runs/train'): ") or "runs/train"
    
    # Preparar dataset
    yaml_path = preparar_yolo_dataset(dataset_dir, yolo_dataset_dir)
    
    # Configuración de entrenamiento
    epochs = int(input("Número de épocas (recomendado: 50-100, por defecto: 50): ") or "50")
    model_size = input("Tamaño del modelo (n/s/m/l/x, por defecto: n): ") or "n"
    batch_size = int(input("Tamaño del batch (por defecto: 16): ") or "16")
    
    # Entrenar modelo
    results, model, best_model_path = entrenar_modelo(
        yaml_path=yaml_path,
        epochs=epochs,
        model_size=model_size,
        batch_size=batch_size,
        save_dir=modelo_save_dir
    )
    
    # Evaluar y visualizar
    evaluar_modelo(model, yaml_path)
    
    results_dir = os.path.join(os.getcwd(), modelo_save_dir, "model")
    visualizar_resultados(results_dir)
    
    # Probar en imágenes
    validation_images_dir = os.path.join(yolo_dataset_dir, "images", "val")
    probar_modelo(best_model_path, validation_images_dir)
    
    print(f"¡Proceso de entrenamiento completado!")
    print(f"Modelo guardado en: {best_model_path}")

if __name__ == "__main__":
    main()