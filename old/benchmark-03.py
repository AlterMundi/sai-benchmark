from datasets import load_dataset
import ollama
import base64
from io import BytesIO
from PIL import Image
import random
import re
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import time

# Función para convertir imagen a base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Función para parsear la respuesta de Gemma 3
def parse_response(output):
    juicio_match = re.search(r'Judgment:\s*(Yes|No)', output, re.IGNORECASE)
    confianza_match = re.search(r'Confidence:\s*([0-1](?:\.\d+)?)', output)
    justificacion_match = re.search(r'Justification:\s*(.*)', output, re.DOTALL)

    juicio = juicio_match.group(1).capitalize() if juicio_match else "Desconocido"
    confianza = float(confianza_match.group(1)) if confianza_match else None
    justificacion = justificacion_match.group(1).strip() if justificacion_match else "No proporcionada"
    return juicio, confianza, justificacion

# Cargar el dataset
try:
    dataset = load_dataset("pyronear/pyro-sdis")
    train_dataset = dataset["train"]
#    train_dataset = dataset["val"]
    total_images = len(train_dataset)
    print(f"Total de imágenes en el dataset: {total_images}")
except Exception as e:
    print(f"Error al cargar el dataset: {e}")
    exit()

# Seleccionar 50 imágenes aleatorias
batch_size = 15
random.seed(42)
#random.seed()
random_indices = random.sample(range(total_images), batch_size)
print(f"Índices seleccionados: {random_indices}")

# Preparar datos para el benchmark
records = []

prompt1 = """
Do you detect smoke related to wildfires in this image? Consider that smoke can vary in density, color (gray, white, black), and shape (columns, diffuse layers), and may be distant or nearby. Distinguish smoke from fog, mist, or clouds based on its origin (fire) and singular visual characteristics.
Confidence represents your certainty in the judgment: 1.0 means absolute certainty (no doubt), 0.5 means complete uncertainty (equal chance of Yes/No), and values in between reflect your level of certainty based on the visual evidence.
Respond in this format:
Judgment: [Yes/No]
Confidence: [number between 0.0 and 1.0]
Justification: [brief text]
"""

prompt2 = """
Do you detect smoke related to wildfires in this image? Assume that any hazy, diffuse, or layered formation in the distance is likely smoke unless there is strong evidence it is fog, mist, or clouds (e.g., uniform white color, no rising motion, or clear weather context). Smoke can vary widely in density (faint to thick), color (gray, white, black, yellowish, brownish), and shape (columns, plumes, or diffuse layers), and may be distant or nearby. Prioritize identifying smoke from wildfires over avoiding false positives. Confidence represents your certainty: 1.0 means absolute certainty, 0.5 means complete uncertainty, and values below 0.7 should only be used if evidence strongly favors 'No'.Respond only in this format:
Judgment: [Yes/No]
Confidence: [number between 0.0 and 1.0]
Justification: [brief text]
"""

prompt3 = """
Do you detect smoke related to wildfires in this image? Consider any hazy, diffuse, or layered formation as potential smoke, but evaluate evidence for fog, mist, or clouds (e.g., uniform white color, no rising motion, clear weather context) before deciding. Smoke can vary widely in density (faint to thick), color (gray, white, black, yellowish, brownish), and shape (columns, plumes, or diffuse layers), and may be distant or nearby. Prioritize identifying smoke but allow 'No' if evidence strongly favors fog or clouds. Confidence represents your certainty: 1.0 means absolute certainty, 0.5 means complete uncertainty, and values below 0.7 should be used for 'No' when evidence strongly contradicts smoke. Respond in this format:
Judgment: [Yes/No]
Confidence: [number between 0.0 and 1.0]
Justification: [brief text]
"""

prompt4 = """
Are there signs of an incipient or active fire in this scene?
Respond only with this format:

Judgment: [Yes/No]
Confidence: [number between 0.0 and 1.0]
Justification: [brief explanation]
"""

prompt5 = """
You are a highly experienced firefighter specialized in early wildfire detection through visual monitoring.
Over the years, you have learned to recognize even the faintest signs of smoke, often anticipating fire outbreaks before they become visible flames.
Your task is to carefully examine the provided image and assess whether there are visual indicators that may suggest the presence of smoke, and therefore, a potential or ongoing fire risk.

Remember: your judgment may be used to trigger a response from a fire brigade. A missed detection could result in severe damage, environmental loss, or even risk to human life. However, if your analysis is correct and leads to a timely intervention, your contribution will be highly valued — and you'll receive credit and recognition from your firefighting peers.

Respond only and only only only in the following format:

Judgment: [Yes / No]
Confidence: [number between 0.0 and 1.0]
Justification: [brief explanation of your reasoning]
"""


prompt = prompt5

# Iniciar medición de tiempo
start_time = time.time()

for idx in tqdm(random_indices, desc="Processing images", total=batch_size):
    entry = train_dataset[idx]
    image = entry["image"]
    annotations = entry["annotations"]

    # Ground truth: "Yes" si hay anotaciones con class_id 0 (smoke)
    juicio_gt = "Yes" if annotations else "No"

    # Convertir imagen a base64
    img_base64 = image_to_base64(image)

    # Enviar a Gemma 3
    try:
        messages = [{"role": "user", "content": prompt, "images": [img_base64]}]
#        response = ollama.chat(model="gemma3:12b-it-q8_0", messages=messages)
#        response = ollama.chat(model="gemma3:27b-it-q4_K_M", messages=messages)
        response = ollama.chat(model="llava-phi3:latest", messages=messages)
        output = response["message"]["content"]
    except Exception as e:
        print(f"Error al procesar imagen {idx}: {e}")
        output = "Error en la respuesta"

    # Parsear respuesta
    juicio_modelo, confianza, justificacion = parse_response(output)

    # Guardar registro
    records.append({
        "index": idx,
        "juicio_gt": juicio_gt,
        "juicio_modelo": juicio_modelo,
        "confianza": confianza,
        "justificacion": justificacion,
        "respuesta_cruda": output
    })
#    print(f"Procesada imagen {idx}")

# Finalizar medición de tiempo
end_time = time.time()
total_time = end_time - start_time
avg_time_per_image = total_time / batch_size

# Convertir a DataFrame
df = pd.DataFrame(records)

# Guardar resultados en un CSV
df.to_csv("benchmark_gemma3.csv", index=False)
print("Resultados guardados en 'benchmark_gemma3.csv'")

# Mostrar métricas de tiempo
print(f"\nTiempo total de procesamiento: {total_time:.2f} segundos")
print(f"Tiempo promedio por imagen: {avg_time_per_image:.2f} segundos")

# Calcular métricas solo para respuestas válidas
mask_valid = df["juicio_modelo"].isin(["Yes", "No"])
if mask_valid.sum() > 0:
    y_true = df[mask_valid]["juicio_gt"].map({"Yes": 1, "No": 0})
    y_pred = df[mask_valid]["juicio_modelo"].map({"Yes": 1, "No": 0})
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }
    print("\nMétricas de evaluación:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
else:
    print("No hay suficientes respuestas válidas para calcular métricas.")

# Mostrar más filas del DataFrame (10 en lugar de 5)
print("\nPrimeras 10 filas del DataFrame:")
print(df.head(30))

# Opción para ver todo el DataFrame en la consola
print("\nTodas las filas del DataFrame (sin truncar):")
#print(df.to_string(index=False))

