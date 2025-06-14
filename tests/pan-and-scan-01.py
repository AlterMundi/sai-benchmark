# Instalar bibliotecas necesarias (descomentar si no están instaladas)
# !pip install transformers torch pillow

# Importar bibliotecas
from transformers import Gemma3ForConditionalGeneration, Gemma3Processor
from PIL import Image
import torch
from transformers import __version__
print(__version__)                 # Asegúrate de 4.51+

print('Authenticating...')
from huggingface_hub import login
login(token="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")

print('Loading model...')

# Paso 1: Cargar el modelo y el procesador
model_id = "google/gemma-3-4b-it"  # Modelo de 4B parámetros para eficiencia
processor = Gemma3Processor.from_pretrained(model_id)
model = Gemma3ForConditionalGeneration.from_pretrained(model_id)
# processor.image_processor.overlap_ratio = 0.15   # solape moderado

# Paso 2: Cargar una imagen de alta resolución
# Reemplazar con la ruta real de tu imagen
# image_path = "camera-1284459_640.jpg"
image_path = "camera-1284459_3840.jpg"
image = Image.open(image_path)

# Paso 3: Definir el prompt para el modelo
prompt = """
Se enviarán 1 imagen global + 2 crops (fila superior-izq, superior-der, inferior).
Devuelve objetos fusionados por IoU > 0.3 con bbox_norm global y crop_ids.
Responde SOLO JSON válido ⚠.

{
  "objetos": [
    { "nombre": "...", "descripcion": "...",
      "bbox_norm": [x0,y0,x1,y1], "crop_ids":[…], "confianza": … },
    …
  ],
  "areas_vacias": [[x0,y0,x1,y1],…],
  "pan_and_scan": {"enabled": true, "tiles": 3}
}

<start_of_image>
"""

# Paso 4: Procesar la imagen con pan-and-scan habilitado
inputs = processor(
    text=prompt,
    images=[image],  # Las imágenes deben ser una lista
    images_kwargs={     # Forzar pan-and-scan
        "do_pan_and_scan": True,
      #  "pan_and_scan_min_pixels": 0, # IGNORED
        "pan_and_scan_min_ratio_to_activate": 0.0,
        "pan_and_scan_max_num_crops": 8,
      #  "max_iou_entre_cajas": 0.3 # IGNORED
    },
    return_tensors="pt"
)

n_image_tokens = (inputs["input_ids"] == processor.tokenizer.image_token_id).sum()
print("Image tokens:", n_image_tokens.item())

# Para saber el Nº de crops calculado:
# print(inputs)              # (N_crops, 3, 896, 896)

# verifica cuántos crops creó:
print("Crops:", inputs["pixel_values"].shape[0])
print('Starting Inference on file:',image_path)

# Paso 5: Mover el modelo y los inputs al dispositivo adecuado (GPU si está disponible, sino CPU)
device = "cuda" # if torch.cuda.is_available() else "cpu"   # Force Gpu vs Cpu
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Paso 6: Generar la salida
outputs = model.generate(**inputs, max_new_tokens=10000)

# Paso 7: Decodificar e imprimir la salida
print(processor.decode(outputs[0], skip_special_tokens=False))
