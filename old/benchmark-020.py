from datasets import load_dataset
import ollama
import base64
from io import BytesIO
from PIL import Image
import random
import re
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Función para convertir imagen a base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Función para parsear la respuesta de Gemma 3
def parse_response(output):
    juicio_match = re.search(r'Juicio:\s*(Sí|No)', output, re.IGNORECASE)
    confianza_match = re.search(r'Confianza:\s*([0-1](?:\.\d+)?)', output)
    justificacion_match = re.search(r'Justificación:\s*(.*)', output, re.DOTALL)

    juicio = juicio_match.group(1).capitalize() if juicio_match else "Desconocido"
    confianza = float(confianza_match.group(1)) if confianza_match else None
    justificacion = justificacion_match.group(1).strip() if justificacion_match else "No proporcionada"
    return juicio, confianza, justificacion

# Cargar el dataset
try:
    dataset = load_dataset("pyronear/pyro-sdis")
    train_dataset = dataset["train"]
    total_images = len(train_dataset)
    print(f"Total de imágenes en el dataset: {total_images}")
except Exception as e:
    print(f"Error al cargar el dataset: {e}")
    exit()

# Seleccionar 50 imágenes aleatorias
batch_size = 20
random.seed(42)
random_indices = random.sample(range(total_images), batch_size)
print(f"Índices seleccionados: {random_indices}")

# Preparar datos para el benchmark
records = []
prompt = (
    "¿Hay humo en esta imagen?\n\n"
    "Respondé en este formato:\n"
    "Juicio: [Sí/No]\n"
    "Confianza: [número entre 0.0 y 1.0]\n"
    "Justificación: [texto breve]"
)

for idx in random_indices:
    entry = train_dataset[idx]
    image = entry["image"]
    annotations = entry["annotations"]

    # Ground truth: "Sí" si hay anotaciones con class_id 0 (smoke)
    juicio_gt = "Sí" if annotations else "No"

    print (juicio_gt, ':',annotations)
