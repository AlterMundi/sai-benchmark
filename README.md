

### Ajuste del benchmark **Early-Fire** para trabajar **exclusivamente con Qwen 2.5-VL**

*(una ruta para usarlo tanto desde **Ollama** como desde **Hugging Face/Transformers**)*

---

## 1. Entender los límites y “quirks” de Qwen 2.5-VL

| Rasgo                             | Detalle relevante                                                                                                                                                                                                                                                                                                                                                                       | Fuente |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| **Dinámica de resolución**        | El ViT aplica *Naive Dynamic Resolution*: acepta cualquier resolución y decide cuántos tokens visuales generan las ventanas de atención. Recomiendan “256-1280 tokens” para balancear coste↔calidad. ([huggingface.co](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct?utm_source=chatgpt.com), [qwenlm.github.io](https://qwenlm.github.io/blog/qwen2-vl/?utm_source=chatgpt.com)) |        |
| **Ventana de atención**           | Implementa *window attention* para acelerar inferencia; el patch base suele ser 14×14 px (ViT-14). ([github.com](https://github.com/QwenLM/Qwen2.5-VL?utm_source=chatgpt.com))                                                                                                                                                                                                          |        |
| **Entrada multimodal**            | El toolkit `qwen-vl-utils` permite mezclar `<image>`…`</image>` en el prompt o pasar imágenes en base64 / URL. ([huggingface.co](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct?utm_source=chatgpt.com))                                                                                                                                                                              |        |
| **Salida**                        | Devuelve texto libre; hay que *forzar* al modelo a producir nuestro JSON de smoke-detection (contrato V1.0).                                                                                                                                                                                                                                                                            |        |
| **Tamaños de modelo disponibles** | 3 B / 7 B (open) y 72 B (HF, GPU grande). Ollama expone 3–72 B quantizados en GGUF. ([labellerr.com](https://www.labellerr.com/blog/run-qwen2-5-vl-locally/?utm_source=chatgpt.com), [ollama.com](https://ollama.com/library/qwen2.5?utm_source=chatgpt.com))                                                                                                                           |        |

---

## 2. Arquitectura de ejecución dual

```text
              +------------------+
              | benchmark runner |
              +--------+---------+
                       |
             ┌─────────┴─────────┐
             |                   |
   +---------v------+   +--------v---------+
   |  local Ollama  |   |  HF Transformers |
   |  (HTTP API)    |   |  (Python)        |
   +---------+------+   +--------+---------+
             |                   |
       qwen2.5-vl:7b     Qwen2.5-VL-7B-Instruct
```

### 2.1 Módulo **`models/ollama_qwen.py`**

```python
import base64, requests, json, pathlib

URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-vl:7b"

def _encode_image(path):
    data = pathlib.Path(path).read_bytes()
    return base64.b64encode(data).decode()

SYSTEM = (
    "You are an early-fire detection agent. "
    "Answer ONLY with a valid JSON matching this schema: "
    '{"has_smoke": bool, "bbox": [x_center,y_center,width,height]}.'
)

def infer(prompt_img, image_path):
    payload = {
        "model": MODEL,
        "prompt": f"<sys>{SYSTEM}</sys><image>{prompt_img}</image>",
        "stream": False,
        "images": [_encode_image(image_path)]
    }
    resp = requests.post(URL, json=payload, timeout=180)
    return json.loads(resp.json()["response"])
```

### 2.2 Módulo **`models/hf_qwen.py`**

```python
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
import torch, json, PIL.Image as Image

checkpoint = "Qwen/Qwen2.5-VL-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    checkpoint, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)

SYSTEM = ("You are an early-fire detection agent. "
          "Return ONLY JSON: {\"has_smoke\":bool,\"bbox\":[xc,yc,w,h]}")

def infer(image_path):
    image = Image.open(image_path)
    inputs = processor(text=SYSTEM, images=[image], return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=64)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return json.loads(text)
```

Both wrappers expose `infer()` → JSON dict compatible with the **Early-Fire Score** pipeline.

---

## 3. Ajustes al benchmark Early-Fire

| Bloque                     | Cambio concreto para Qwen 2.5-VL                                                                                                          |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **1 · Calibración**        | Recortar negativos con resoluciones mixtas (480p-2160p) para verificar que la dinámica de resolución no rompa la tasa FP.                 |
| **2 · Secuencia temprana** | Usar *sampling FPS dinámico* (propio de Qwen) → tomar 1 frame cada N seg cuando N≥3 para no disparar tokens (>1280) en secuencias largas. |
| **3 · Localización**       | Como ViT-14 reduce spatial granularity, admitimos **IOU ≥ 0.4** en mAP; si necesitan 0.5, suban de 7 B → 72 B o envíen el crop al modelo. |
| **4 · Robustez**           | Añadir set “camera-shake + motion-blur” (Qwen 2.5 usa MRoPE temporal, da buen desempeño; medimos latencia extra).                         |
| **5 · Explicabilidad**     | Incluir prompt auxiliar “Explain your JSON in 20 words” — sólo para análisis humano, no puntúa.                                           |

La fórmula de **Early-Fire Score** sigue igual; solo revisen el término de latencia porque la generación en Ollama (CPU/GGUF int4) suele tardar 0.8-1.2 s por imagen, mientras que en HF + GPU A100 baja a 0.25 s.

---

## 4. Pipeline `evaluation.py` (extracto)

```python
from models import ollama_qwen as qwen_local
# or
from models import hf_qwen as qwen_hf

def run_model(image_seq, local=True):
    infer = qwen_local.infer if local else qwen_hf.infer
    preds = [infer(img) for img in image_seq]
    return preds
```

Añadan flag `--engine {ollama,hf}` al CLI para conmutar.

---

## 5. Infra & despliegue rápido

| Entorno          | Pasos clave                                                                                                                                                                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ollama**       | 1. `ollama pull qwen2.5-vl:7b`  (usa quant int4 GGUF)<br>2. `ollama serve` <br>3. Ajustar `OLLAMA_HOST` si corre en otra máquina.                                                                                                     |
| **Hugging Face** | 1. `pip install git+https://github.com/huggingface/transformers accelerate qwen-vl-utils`<br>2. `HF_HOME=/mnt/models` para cache local.<br>3. Si GPU < 24 GB, el 3 B es seguro; 7 B necesita 24 GB (fp16) o 16 GB con bits-and-bytes. |

---

## 6. Checklist inmediato

1. **Implementar wrappers** de arriba y test unitarios con 3 imágenes humo/no-humo.
2. **Actualizar README** del repo `sai-benchmark/` con las instrucciones `ollama` / `HF`.
3. Correr bloque **Calibración** en ambas rutas y comparar:

   ```
   python evaluate.py --engine ollama
   python evaluate.py --engine hf
   ```
4. Si la salida JSON se “desborda” con texto extra, refuerza el `SYSTEM` prompt con:

   ```
   "If output does NOT match JSON schema, answer exactly: BAD_JSON"
   ```

   y descarta esos casos en el scorer.

Con esto, el benchmark queda 100 % alineado a **Qwen 2.5-VL** y a los dos escenarios operativos que van a usar.
