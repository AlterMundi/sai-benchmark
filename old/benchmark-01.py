from datasets import load_dataset
import ollama
import base64
from io import BytesIO
from PIL import Image

print("Loading dataset...")
# Cargar dataset
dataset = load_dataset("pyronear/pyro-sdis")
batch_size = 50
image_batch = dataset["train"][:batch_size]["image"]

print("Converting images into base64...")
# Convertir a base64
images_base64 = []
for img in image_batch:
    print(".\r")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    images_base64.append(img_base64)

# Prompt
prompt = """
    ¿Observás signos de incendio activo o incipiente en esta imagen?
    
    Respondé con este formato:
    Juicio: [Sí/No]
    Confianza: [número entre 0.0 y 1.0]
    Justificación: [texto breve]
    """
responses = []

# Procesar con Gemma 3
for i, img_base64 in enumerate(images_base64):
    messages = [{"role": "user", "content": prompt, "images": [img_base64]}]
    response = ollama.chat(model="gemma3:12b-it-q8_0", messages=messages)
    responses.append(f"Imagen {i+1}: {response['message']['content']}")
    print(f"Procesada imagen {i+1}/50")

# Mostrar resultados
#for resp in responses:
#    print(resp)
annotations = dataset["train"][:batch_size]["annotations"]
for i, (resp, ann) in enumerate(zip(responses, annotations)):
    print('----------------------------------')
    print(f"| Anotación: {ann}")
    print(f"{resp}")
