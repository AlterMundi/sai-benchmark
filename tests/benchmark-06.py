#!/usr/bin/env python3
import os
import base64
import time
import re
import json
import argparse
import concurrent.futures
from io import BytesIO
from typing import Dict, List
from PIL import Image
import ollama

# Configuración mínima - sin logging para máxima velocidad
DEFAULT_PROMPT = """
You are looking at a sequence of images. Analyze them carefully and determine if the target pattern is present.
Judgment: [Yes/No]
Confidence: [number between 0.0 and 1.0]
Justification: [brief explanation]
"""

# Cache para ground truth - evita leer el mismo archivo varias veces
gt_cache = {}

def parse_response(output):
    """Extrae juicio, confianza y justificación de la respuesta del modelo"""
    judgment_match = re.search(r'Judg?ment:\s*(Yes|No|yes|no)', output, re.IGNORECASE)
    confidence_match = re.search(r'Confidence:\s*([0-9](?:\.\d+)?)', output)
    justification_match = re.search(r'Justification:\s*(.*)', output, re.DOTALL)

    judgment_bool = False
    if judgment_match:
        judgment = judgment_match.group(1).lower()
        judgment_bool = judgment == "yes"
    
    confidence = None
    if confidence_match and confidence_match.group(1):
        try:
            confidence = float(confidence_match.group(1))
        except:
            pass
    
    justification = "Not provided"
    if justification_match:
        justification = justification_match.group(1).strip()
    
    return judgment_bool, confidence, justification

def process_sequence(sequence_data):
    """Procesa una sola secuencia - diseñado para ejecución independiente en un worker"""
    sequence_path, dataset_path, model_name, prompt = sequence_data
    
    # Obtener nombre de la secuencia e inicializar temporizador
    sequence_id = os.path.basename(sequence_path)
    start_time = time.time()
    
    try:
        # Recopilación rápida de imágenes y ground truth
        images = []
        ground_truth = False
        
        for filename in sorted(os.listdir(sequence_path)):
            file_path = os.path.join(sequence_path, filename)
            
            # Procesar solo imágenes
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(file_path)
                
                # Comprobar ground truth solo una vez por archivo
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                txt_path = os.path.join(sequence_path, txt_filename)
                
                if txt_path in gt_cache:
                    if gt_cache[txt_path]:
                        ground_truth = True
                elif os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        content = f.read().strip()
                        is_positive = content != ""
                        gt_cache[txt_path] = is_positive
                        if is_positive:
                            ground_truth = True
                            
        # No hay imágenes - devolver error
        if not images:
            return {
                "sequence_id": sequence_id,
                "error": "No images found in sequence folder",
                "processing_time": time.time() - start_time
            }
            
        # Convertir imágenes a base64 - optimizado
        images_base64 = []
        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    # Redimensionar imágenes grandes para mayor velocidad
                    max_size = 1024
                    if img.width > max_size or img.height > max_size:
                        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG", quality=85, optimize=True)
                    images_base64.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
            except Exception as e:
                return {
                    "sequence_id": sequence_id,
                    "error": f"Error processing image {img_path}: {str(e)}",
                    "processing_time": time.time() - start_time
                }
        
        # Crear mensaje para Ollama - simple y directo
        message = {
            "role": "user", 
            "content": prompt,
            "images": images_base64
        }
        
        # Llamar a Ollama API
        response = ollama.chat(model=model_name, messages=[message])
        output = response["message"]["content"]
        
        # Parsear respuesta
        prediction, confidence, justification = parse_response(output)
        
        # Calcular tiempo total
        processing_time = time.time() - start_time
        
        # Construir resultado
        return {
            "sequence_id": sequence_id,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "confidence": confidence,
            "justification": justification,
            "processing_time": processing_time,
            "is_correct": prediction == ground_truth,
            "raw_response": output
        }
        
    except Exception as e:
        return {
            "sequence_id": sequence_id,
            "error": f"Error analyzing sequence: {str(e)}",
            "processing_time": time.time() - start_time
        }

def calculate_metrics(results):
    """Calcula métricas a partir de los resultados"""
    valid_results = [r for r in results if "error" not in r]
    total = len(valid_results)
    
    if total == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
    
    true_positives = sum(1 for r in valid_results if r["prediction"] and r["ground_truth"])
    true_negatives = sum(1 for r in valid_results if not r["prediction"] and not r["ground_truth"])
    false_positives = sum(1 for r in valid_results if r["prediction"] and not r["ground_truth"])
    false_negatives = sum(1 for r in valid_results if not r["prediction"] and r["ground_truth"])
    
    accuracy = (true_positives + true_negatives) / total
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "total_sequences": total,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_processing_time": sum(r["processing_time"] for r in valid_results)
    }

def main():
    """Función principal - procesamiento paralelo de secuencias"""
    parser = argparse.ArgumentParser(description="Análisis ultrarrápido de secuencias de imágenes")
    parser.add_argument("--dataset", required=True, help="Ruta al dataset con carpetas de secuencias")
    parser.add_argument("--model", default="gemma3:12b-it-q8_0", help="Modelo Ollama a usar")
    parser.add_argument("--prompt", help="Prompt personalizado para el análisis")
    parser.add_argument("--max-sequences", type=int, help="Número máximo de secuencias a procesar")
    parser.add_argument("--workers", type=int, default=3, help="Número de workers paralelos")
    parser.add_argument("--output", help="Archivo de salida para resultados (formato JSON)")
    parser.add_argument("--no-cache", action="store_true", help="Desactivar cache de ground truth")
    
    args = parser.parse_args()
    
    # Validación básica
    if not os.path.isdir(args.dataset):
        print(f"Error: La ruta {args.dataset} no existe o no es un directorio")
        return 1
    
    start_time = time.time()
    
    # Usar prompt personalizado o el predeterminado
    prompt = args.prompt or DEFAULT_PROMPT
    
    # Encontrar carpetas de secuencias - rápido y directo
    all_folders = [f for f in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, f))]
    all_folders.sort()
    
    if args.max_sequences:
        folders = all_folders[:args.max_sequences]
    else:
        folders = all_folders
    
    print(f"Procesando {len(folders)} secuencias de {len(all_folders)} encontradas")
    
    # Preparar datos para procesamiento paralelo
    sequence_data = [
        (os.path.join(args.dataset, folder), args.dataset, args.model, prompt)
        for folder in folders
    ]
    
    # Procesar secuencias en paralelo con multithreading
    results = []
    errors = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_sequence, data): data[0] for data in sequence_data}
        
        # Contador para seguimiento
        completed = 0
        
        for future in concurrent.futures.as_completed(futures):
            folder_path = futures[future]
            folder_name = os.path.basename(folder_path)
            try:
                result = future.result()
                completed += 1
                
                # Verificar si hay error
                if "error" in result:
                    print(f"[{completed}/{len(folders)}] ❌ {folder_name}: {result['error']}")
                    errors.append(result)
                else:
                    # Mostrar resultado mínimo
                    status = "✓" if result["is_correct"] else "✗"
                    print(f"[{completed}/{len(folders)}] {status} {folder_name}: Pred={result['prediction']} GT={result['ground_truth']} ({result['processing_time']:.2f}s)")
                    results.append(result)
                
                # Guardar resultados incrementalmente
                if args.output and (completed % 5 == 0 or completed == len(folders)):
                    all_data = results + errors
                    metrics = calculate_metrics(all_data)
                    
                    with open(args.output, 'w') as f:
                        json.dump({
                            "test_info": {
                                "dataset_path": args.dataset,
                                "model": args.model,
                                "prompt": prompt,
                                "workers": args.workers,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "progress": f"{completed}/{len(folders)}",
                                "elapsed_time": time.time() - start_time
                            },
                            "metrics": metrics,
                            "results": all_data
                        }, f)
                
            except Exception as e:
                completed += 1
                print(f"[{completed}/{len(folders)}] ❌ Error procesando {folder_name}: {str(e)}")
                errors.append({
                    "sequence_id": folder_name,
                    "error": str(e),
                    "processing_time": 0
                })
    
    # Calcular métricas finales
    all_data = results + errors
    metrics = calculate_metrics(all_data)
    
    # Guardar resultados finales
    output_file = args.output or f"results_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "test_info": {
                "dataset_path": args.dataset,
                "model": args.model,
                "prompt": prompt,
                "workers": args.workers,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_time": time.time() - start_time
            },
            "metrics": metrics,
            "results": all_data
        }, f)
    
    # Mostrar resumen de resultados
    print("\nRESUMEN DE RESULTADOS:")
    print(f"Total secuencias: {len(all_data)}")
    print(f"Secuencias exitosas: {len(results)}")
    print(f"Secuencias con error: {len(errors)}")
    
    valid_results = [r for r in all_data if "error" not in r]
    if valid_results:
        print(f"\nAccuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    print(f"\nTiempo total: {time.time() - start_time:.2f} segundos")
    print(f"Resultados guardados en: {output_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())
