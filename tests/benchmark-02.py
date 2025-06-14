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
#    train_dataset = dataset["val"]
    total_images = len(train_dataset)
    print(f"Total de imágenes en el dataset: {total_images}")
except Exception as e:
    print(f"Error al cargar el dataset: {e}")
    exit()

# Seleccionar 50 imágenes aleatorias
batch_size = 1000
random.seed(42)
#random.seed()
random_indices = random.sample(range(total_images), batch_size)
print(f"Índices seleccionados: {random_indices}")

# Preparar datos para el benchmark
records = []

prompt2 = (
    "¿Hay humo en esta imagen?\n\n"
    "Respondé en este formato:\n"
    "Juicio: [Sí/No]\n"
    "Confianza: [número entre 0.0 y 1.0]\n"
    "Justificación: [texto breve]"
)

prompt = """
Estás colaborando con un sistema de monitoreo forestal para la detección temprana de incendios.

Observá la imagen proporcionada y determiná si existen signos visuales de la presencia de humo, incluyendo:

- Columnas de humo visibles
- Humo delgado, difuso o parcialmente oculto
- Humo lejano o difícil de distinguir, siempre que se diferencie de nubes o niebla

Tené en cuenta que estas imágenes son amplias y panorámicas, y no necesariamente muestran llamas.

Respondé únicamente en el siguiente formato:

Juicio: [Sí / No]  
Confianza: [número entre 0.0 y 1.0]  
Justificación: [breve texto explicando tu decisión]
"""



for idx in random_indices:
    entry = train_dataset[idx]
    image = entry["image"]
    annotations = entry["annotations"]

    # Ground truth: "Sí" si hay anotaciones con class_id 0 (smoke)
    juicio_gt = "Sí" if annotations else "No"

    # Convertir imagen a base64
    img_base64 = image_to_base64(image)

    # Enviar a Gemma 3
    try:
        messages = [{"role": "user", "content": prompt, "images": [img_base64]}]
        response = ollama.chat(model="gemma3:12b-it-q8_0", messages=messages)
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
    print(f"Procesada imagen {idx}")

# Convertir a DataFrame
df = pd.DataFrame(records)

# Guardar resultados en un CSV
df.to_csv("benchmark_gemma3.csv", index=False)
print("Resultados guardados en 'benchmark_gemma3.csv'")

# Calcular métricas solo para respuestas válidas
mask_valid = df["juicio_modelo"].isin(["Sí", "No"])
if mask_valid.sum() > 0:
    y_true = df[mask_valid]["juicio_gt"].map({"Sí": 1, "No": 0})
    y_pred = df[mask_valid]["juicio_modelo"].map({"Sí": 1, "No": 0})
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
print(df.head(10))

# Opción para ver todo el DataFrame en la consola
print("\nTodas las filas del DataFrame (sin truncar):")
#print(df.to_string(index=False))

