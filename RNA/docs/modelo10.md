# Evaluación profunda de arquitecturas para el SAI y propuesta final (Servidor RTX 3090 y Edge Raspberry Pi) 

Objetivo: evaluar críticamente las arquitecturas propuestas en los documentos troncales del SAI y recomendar una arquitectura final “equilibrada”: suficientemente probada para operar con fiabilidad hoy, pero con innovaciones que aporten mejoras reconocidas sin empujar al sistema a una zona de $\mathrm {I}+\mathrm {D}$  incierta. La propuesta incluye dos variantes: (a) servidor con RTX 3090 (inferencia centralizada y validación de alarmas) y (b) inferencia local en Raspberry Pi (despliegue progresivo en los nodos). 

## 1) Resumen ejecutivo

• Diagnóstico. Los dos documentos principales del SAI recomiendan un modelo híbrido en servidor que combine backbone CNN moderno (ConvNeXt/EfficientNet), módulo temporal 

(LSTM/Transformer) y cabeza de detección tipo YOLOv8 (anchor-free); y, en el edge, MobileNetV3/EfficientNet-Lite o YOLOv8-Nano con pruning + quantization + distillation. Esta línea surge de una revisión del estado del arte y de las restricciones prácticas del proyecto. (AlterMundi, 2025a; 2025b). 

• Juicio técnico. Esta recomendación está bien alineada con el objetivo de “innovar sin caer en exploración frágil”: 

o Los modelos espaciotemporales tipo SmokeyNet han demostrado mejor recall y detección más temprana en FIgLib ( $\mathrm {F}1\approx 82.6$ ; detección ≈3 min tras la aparición del humo) (Chernyavsky et al., 2022), y son adecuados para ejecutarse en servidor con GPU. (AlterMundi, 2025a; Chernyavsky et al., 2022). 

o YOLOv8 (anchor-fr $\mathrm {ee}+\mathrm {C}2\mathrm {f})$ aporta localización y mejora de precisión reportada en detección de humo/fuego en benchmarks específicos, manteniendo latencia razonable en GPU. (AlterMundi, 2025a). 

o En edge, MobileNetV3/EfficientNet-Lite (o YOLOv8-Nano) cuantizados tienen el mejor compromiso precisión/consumo/latencia para Raspberry Pi, y el uso de buffers temporales de 3–5 frames reduce falsos positivos sin aumentar demasiado el cómputo. (AlterMundi, 2025b). 

• Decisión. Mantener la línea híbrida propuesta, pero estabilizarla con un plan por etapas (entrenamiento y despliegue), y dos ajustes de bajo riesgo: 

o Formación por etapas (pre-entrenar/ajustar por partes) para reducir complejidad de la integración temporal+detección en servidor; 

o Agregación temporal ligera en edge (votación sobre $3-5frames+framedifferencing)$ y distillation desde el modelo servidor. (AlterMundi, 2025a; 2025b). 

Conclusión: Las arquitecturas propuestas son pertinentes y prudentes para el SAI; con los ajustes indicados,ofrecen un camino operativo inmediato en $3090y$  un sendero de adopción progresiva en Raspberry Pi, sin caer en apuestas experimentales frágiles. 

## 2) Contexto, requisitos y restricciones del SAI

• Arquitectura del sistema: red distribuida de nodos SAI-CAM ( $RaspberryPi+cmarasI$ P) y servidor central (RTX 3090) para entrenamiento, inferencia avanzada y validación de alarmas. El flujo contempla captura periódica, pre-filtros ligeros y subida al servidor, que consolida alarmas y notifica (p. ej., bot de Telegram). (AlterMundi, 2025a). 

• Criterios de calidad: detección temprana, baja tasa de falsos positivos (nubes, niebla, reflejos), robustez a variaciones geográficas/meteorológicas y latencias compatibles con respuesta local; con reproducibilidad y comparabilidad vía SAI-Benchmark. (AlterMundi, 2025a). 

• Estrategia de datos: combinar datasets con temporalidad (FIgLib) y datasets de detección con cajas (FASDD, D-Fire, Nemo, PyroNear-2024) para robustez y mejor generalización inter-dominio. (AlterMundi, 2025a; 2025b). 

# 3) Evidencia técnica clave (síntesis)

• SmokeyNet $(\mathrm {CNN}+\mathrm {LSTM}+\mathrm {ViT})$ sobre FIgLib reporta $\mathbf {F}1\approx \mathbf {82.59}$ , $precision\approx 89.84\%$ , recall $\approx 76.45\%$ , con detecció $n\approx 3$ min tras aparición del humo; integra mosaicos $224\times 224$ , $LSTMyViT$  para razonamiento global. (Chernyavsky et al., 2022; resumen en AlterMundi, $2025a/2025b$ ). 

• YOLOv8: mejoras anchor-free y C2f; en humo/fuego se reportan AP@0.5 cercanos a 0.90 en ciertos escenarios y m $AP\approx 80\%$  para variantes previas (YOLOv5x) en FASDD; buen compromiso precisión/velocidad (según tamaño s/m/x). (AlterMundi, 2025a). 

• Edge models: MobileNetV3/EfficientNet-Lite y YOLOv8-Nano son candidatos realistas en Raspberry Pi; con quantization y pruning logran latencias bajas y bajo consumo; usar buffer de 3–5frames para confirmar detecciones y reducir falsos positivos. (AlterMundi, 2025b). 

Implicación: Los datos sustentan una arquitectura dual servidor+edge, con temporalidad explícita en servidor (donde más rinde) y temporalidad ligera en edge (para robustez sin penalizar recursos). 

# 4) Evaluación crítica de la propuesta del documento

Propuesta base (documentos SAI)

• Servidor (3090): ConvNeXt/EfficientNet + Módulo temporal $(LSTM/Transformer)+cabeza$  YOLOv8 anchor-free; entrenamiento con mezcla de datasets (FIgLib, FASDD, D-Fire, Nemo, PyroNear-2024) y aumentos tipo Mosaic, MixUp, CutMix. (AlterMundi, 2025a; 2025b). 

• Edge (Pi): MobileNetV3/EfficientNet-Lite o YOLOv8-Nano cuantizados, pruning y distillation;buffer de 3–5 frames para confirmar; comunicación al servidor de frames/regiones candidatos. (AlterMundi, 2025b). 

# Fortalezas (APA: AlterMundi, 2025a; 2025b; Chernyavsky et al., 2022):

1. Equilibrio riesgo-beneficio: reutiliza bloques probados (ConvNeXt/EfficientNet, LSTM, YOLOv8) y técnicas estándar (transfer learning, quantization, distillation). 

2. Sensibilidad temprana: incorporar temporalidad emula lo que funcionó en SmokeyNet para detectar columnas tenues y en crecimiento sostenido. 

3. Localización útil: la cabeza YOLOv8 aporta ROI para validación humana/operativa y para tareas futuras (estimación de tamaño de columna). 

4. Escalabilidad operativa: edge liviano + servidor potente; y SAI-Benchmark asegura comparabilidad y reproducibilidad. 

# Riesgos/Desafíos:

• Complejidad de entrenamiento del “trifásico” (backbone + temporal + detección). Mitigación: entrenamiento por etapas, congelamiento progresivo y multi-task con ponderación de pérdidas.(AlterMundi, 2025a; 2025b). 

• Dominio y anotaciones: FIgLib aporta temporalidad pero no es un dataset de cajas a gran escala; la integración con datasets de detección (FASDD/D-Fire/Nemo) es necesaria para la cabeza YOLO, con potencial desbalance de dominios. (AlterMundi, 2025a; 2025b). 

• Edge: incluso con QAT, MobileNet/YOLO-Nano puede sufrir en niebla, reflejos o humo muy sutil;se requiere calibración de umbrales y clase “Other” para distractores. (AlterMundi, 2025a; 2025b). 

Veredicto: La propuesta del documento es correcta y recomendable para el SAI; sólo pide ingeniería de entrenamiento y curaduría de datos cuidadosas. No fuerza al proyecto a exploración SOTA incierta; aprovecha innovaciones consolidadas. 

# 5) Propuesta final para SAI

5.1. Servidor (RTX 3090) — SAI-Server v1: “Híbrido Espaciotemporal con Localización”

Objetivo: máxima sensibilidad para humo incipiente con baja tasa de falsos positivos, manteniendo latencia y throughput adecuados para operar como validador central. 

# Arquitectura

• Backbone: ConvNeXt-T (o EfficientNet-B4/B5 si el reentrenamiento prefiere MBConv), pre-entrenado (ImageNet; opcionalmente CLIP para mejor transfer). (AlterMundi, 2025a; 2025b).

• Bloque temporal: LSTM bidireccional ligero (2 capas) sobre embeddings por frame de una ventana corta de 3–5 frames (espaciados 20–60 s); opción alternativa: Temporal Transformer si el rendimiento lo justifica. (AlterMundi, 2025a; 2025b). 

• Cabeza de detección: YOLOv8-s/m anchor-free (reutilizando FPN del backbone). Se usa multi-task: $=\lambda \text {_det}\text {L_YOLO}+\lambda \text {_cls}\text {L_BCE}+\lambda$ temp L_temporal, donde L_cls es una cabeza global binaria tipo SmokeyNet (para densidad/consistencia de humo). (AlterMundi, 2025a; 2025b; Chernyavsky et al., 2022). 

Entrada: resolución 640–1024 px (dependiendo de FOV); mosaicos/tiling opcional para escenas ultra-anchas, como en SmokeyNet. (Chernyavsky et al., 2022). 

# Entrenamiento por etapas (recomendado para estabilidad)

1. Etapa A (Detección): entrenar YOLOv8 puro en FASDD/D-Fire/Nemo; congelar luego la cabeza (o re-entrenar con LR bajo). (AlterMundi, 2025b). 

2. Etapa B (Clasificación temporal): entrenar Backbone+LSTM con FIgLib (secuencias) para maximizar recall temprano y robustez temporal. (Chernyavsky et al., 2022). 

3. Etapa C (Fusión multi-task): fine-tuning conjunto con mezcla de datasets (muestreo estratificado por dominio), pérdidas ponderadas y aumento (Mosaic, MixUp, CutMix). (AlterMundi, 2025b). 

# Buenas prácticas

• Regularización y compresión (dropout, label smoothing; opcional pruning suave para servir varias instancias en paralelo). (AlterMundi, 2025a). 

• Umbrales y lógica: umbral bajo para recordar casos $tenues+confirmacion$  temporal (detención sostenida $\text {en}\geq 2-3$ frames) para emisión de alarma final; registro de TTD (time-to-detect). (AlterMundi,2025b). 

Salida operativa: (a) score global; (b) cajas con confianza; (c) capa de explicación básica (Grad-CAM o heatmaps) opcional para inspección. (AlterMundi, 2025a). 

# Por qué esta variante es suficiente al inicio

• SmokeyNet-like $(temporal)+YOLO$  (localización) cubren tanto sensibilidad como accionabilidad (ROI). El 3090 puede servir concurrencia de varias cámaras si se lotea por ventanas y se usa inferencia asíncrona. (AlterMundi, 2025a). 

5.2. Edge (Raspberry Pi) — SAI-Edge v1: “Clasificación ligera con voto temporal”

Objetivo: filtrar eventos triviales in situ y acortar tiempo de reacción, sin reemplazar al servidor (que valida). Primera versión con clasificación, no detección, para simplicidad y robustez. 

# Arquitectura

• Backbone: MobileNetV3-Large 0.75× (o EfficientNet-Lite0 si la toolchain lo facilita). QAT-INT8 para TFLite/ONNX Runtime; distillation desde SAI-Server. (AlterMundi, 2025b). 

• Agregado temporal ligero: votación sobre 3–5 frames (espaciados); activar alerta local sólo si ≥ k/ N frames superan umbral $(k=203).$  (AlterMundi, 2025b). 

• Pre-proceso opcional: frame differencing / background subtraction para enmascarar regiones estáticas; máscaras per-cámara para excluir zonas proclives a falsos positivos (p. ej., cielo extremo o humo industrial conocido). (AlterMundi, 2025a). 

Alternativa (cuando se requiera ROI en edge): YOLOv8-Nano INT8, con poda y cuantización; sólo si la latencia y la memoria son aceptables en la Pi concreta. (AlterMundi, 2025b). 

# Comunicación 

• Enviar al servidor frames etiquetados o crops candidatos + metadatos; rate-limit configurable y backoff para evitar picos. (AlterMundi, 2025b). 

## 6) Plan de datos y entrenamiento

• Datasets 

o Temporalidad: FIgLib para aprender curvas de aparición de humo y reducir falsos positivos de nubes/niebla. (Chernyavsky et al., 2022). 

o Detección: FASDD, D-Fire, Nemo (cajas de humo/fuego; incluye condiciones meteorológicas difíciles); incorporar PyroNear-2024 para robustez inter-dominio. (AlterMundi, 2025a; 2025b).

• Aumentos: Mosaic/MixUp/CutMix, jitter de color/haze, recortes por región (cuando haya ROI de humo), y sintéticos (p. ej., StyleGAN2-ADA) si aporta casos de humo tenue. (AlterMundi, 2025b).

• Currículo de entrenamiento

o Pre-entrenar detector en datasets con cajas.

o Ajustar clasificador temporal con FIgLib.

o Fine-tuning conjunto con mezcla estratificada (por dataset y condición meteo). (AlterMundi,2025b). 

• Distillation a edge: usar salidas calibradas del SAI-Server como teacher ( $logits+mapasde$  atención/cajas) para el SAI-Edge. (AlterMundi, 2025a). 

## 7) Métricas, validación y umbrales

• Métricas primarias: Recall de humo incipiente, F1, TTD (time-to-detect, min desde primeras señales),mAP/AP@0.5 para la cabeza de detección. (Chernyavsky et al., 2022; AlterMundi, 2025a). 

• Criterios de aceptación iniciales (recomendados):

o Servidor: reca $11\geq 0$ $.80$ con $\text {TTD}\leq 4$ min en reproducciones de secuencias (target basado en SmokeyNet). (Chernyavsky et al., 2022). 

o Edge: reducir $\text {en}\geq \mathbf {50}$ % los falsos positivos triviales previo al envío, manteniendo recall local $\geq 0.$ 65 en condiciones vistas. (AlterMundi, 2025b). 

• Validación con SAI-Benchmark: suites reproducibles (YAML), reporte de métricas y trazabilidad de experimentos. (AlterMundi, 2025a). 

## 8) Despliegue y operación

• Fase 1 (hoy): toda la inferencia en servidor (RTX 3090) con SAI-Server v1; nodos envían frames periódicos. (AlterMundi, 2025a). 

• Fase 2: habilitar SAI-Edge v1 $(clasificacionligera+vototempor$ al) en Pi para pre-filtrar y sólo subir candidatos; calibrar umbrales por cámara y franja horaria. (AlterMundi, 2025b; 2025a). 

• Fase 3: cuando se requiera ROI local, evaluar YOLOv8-Nano INT8 en edge, manteniendo la validación final en servidor. (AlterMundi, 2025b). 

## 9) Riesgos y mitigaciones

• Dominio/Generalización: mezclar datasets $\text {(FIgLib+FASDD+D-Fire+Nemo+PyroNear-2024)}y$  curar negativos difíciles (clase “Other”). (AlterMundi, 2025b). 

• Complejidad de entrenamiento: entrenamiento escalonado, congelamiento progresivo, validación por módulos (detector solo, temporal solo, fusión). (AlterMundi, 2025a). 

• Edge FPs: buffers temporales, máscaras per-cámara, distillation y QAT con calibración representativa; umbrales dependientes de luz/meteorología. (AlterMundi, 2025a; 2025b). 

# 10) Conclusión

A la luz de la evidencia, las arquitecturas propuestas en los documentos SAI son las más indicadas para el objetivo del proyecto: innovar con prudencia y funcionar hoy. Para el servidor, un híbrido espaciotemporal + YOLOv8 maximiza sensibilidad y utilidad operativa; para el edge, MobileNetV3/EfficientNet-Lite (con QAT y voto temporal) ofrece una vía realista y escalable hacia la inferencia local. Esta combinación no exige exploración SOTA incierta y se apoya en prácticas consolidadas y en datasets relevantes, con evaluación reproducible mediante SAI-Benchmark. (AlterMundi, 2025a; 2025b; Chernyavsky et al., 2022). 

# Referencias (formato APA)

• AlterMundi. (2025a). Sistema de Alerta de Incendios (SAI): Diseño Arquitectónico y Redes Neuronales para Detección Temprana de Humo (informe técnico interno). 

• AlterMundi. (2025b). Investigación sobre arquitecturas de redes neuronales para el SAI (informe técnico interno). 

• Chernyavsky, A., Dewangan, R., et al. (2022). SmokeyNet: Deep Learning Model for Wildland Fire Smoke Detection using FIgLib. Remote Sensing, 14(4), 1007. (Resultados y diseño $CNN+LSTM+ViT$  empleados como referencia para SAI). 

• Huang, W., Jiao, L., Liu, J., et al. (2021). FASDD: A Large-Scale Dataset for Flame and Smoke Detection. Fire, 4(4), 100. (Dataset de detección masivo usado en la etapa del detector). 

• Jadon, A., Varshney, L., Ansari, M.S., & Sharma, R. (2020). D-Fire Dataset: Object Detection for Fire & Smoke. arXiv:2008.01963. (Dataset de detección con cajas de humo/fuego). 

• Muhammad, K., et al. (2022). FFireNet: Deep Learning Based Forest Fire Detection in Smart Cities.Symmetry, 14(10), 2155. (Modelo ligero para edge). 

Nota: Los detalles operativos (SAI-CAM, SAI-Benchmark, configuración de orquestación, parámetros de suites) se toman del corpus técnico del SAI y su revisión del estado del arte. Para rastrear los pasajes exactos,ver los documentos internos citados. 

# Anexo A — Checklist de implementación mínima

## • Servidor (RTX 3090)

o Entrenar YOLOv8-s/m en FASDD/D-Fire/Nemo (Etapa A).

o Entrenar $Backbone+LSTM$  con FIgLib (Etapa B).

o Fine-tuning multi-task (Etapa C) con mezcla de datasets y aumentos (Mosaic/MixUp/CutMix).

o Calibrar umbral global + confirmación temporal $(\geq 2-3$ frames) y TTD objetivo $\leq 4$  min.

## • Edge (Raspberry Pi)

o Compilar MobileNetV3-Large 0.75 INT8 con $QAT+distillation$  (teach $=\text {SAI-Server)}.$ 

o Implementar buffer 3–5 frames con voto k/ N y frame differencing.

o Enviar sólo candidatos ( $frames/crops+metadatos$ ) al servidor para validación.

# SAI — Evaluación de arquitecturas y propuesta final 

Readme técnico 

Objetivo. Evaluar críticamente las arquitecturas de redes neuronales propuestas para el Sistema de Alerta de Incendios (SAI) y entregar una propuesta final para (a) el servidor con GPU RTX 3090 (único punto de inferencia en la fase inicial) y (b) la Raspberry Pi en nodos de vigilancia (inferencia local futura). El enfoque busca innovar con prudencia: combinar componentes probados con innovaciones que ya han mostrado mejoras consistentes y no requieren investigación riesgosa. (AlterMundi, 2025). 

# Tabla de contenido

1. Resumen ejecutivo 

2. Criterios de evaluación 

3. Contexto y evidencia clave 

4. Análisis crítico de las opciones 

5. Propuesta final de arquitectura 

o Servidor (RTX 3090) 

o Edge (Raspberry Pi) 

6. Plan de entrenamiento, validación y despliegue 

7. Riesgos y mitigaciones 

8. Checklist de implementación (MVP → Fase 2) 

<!-- 9. Referencias (APA)  -->

# Resumen ejecutivo

• Qué propone el material base. Los documentos principales recomiendan, para el servidor, un modelo híbrido: backbone ConvNeXt o EfficientNet-B4/B5 + módulo temporal (LSTM/Transformer) + cabeza de detección anchor-free tipo YOLOv8; para edge, un 

MobileNetV3-Small/EfficientNet-Lite0 o YOLOv8-Nano cuantizado/prunado, con verificación temporal de 3–5 frames. (AlterMundi, 2025). 

## • Qué demuestra la evidencia externa clave.

o SmokeyNet $(\mathrm {CNN}+\mathrm {LSTM}+\mathrm {ViT})$ entrenado en FIgLib logra $\text {F1}\approx 82.6\%,$  precisió $\mathrm {n}\approx 89.8\%,$ reca $\text {ll}\approx 76.5\%$ y detección a ~3.1 min de la ignición. (Dewangan et al., 2022). 

o En el ecosistema Pyronear/Pyro-SDIS se ha privilegiado YOLOv8 por su cabeza anchor-free y eficiencia $\text {(C2f+PANet+SPPF),}$  con buenos resultados prácticos y despliegue optimizado. 

## • Nuestra evaluación.

o Servidor: la idea híbrida (ConvNeXt/Efficient $\text {Net+temporal}$  + cabeza YOLOv8) es adecuada para SAI: combina piezas probadas y explica bien la variación espacial/temporal del humo, sin ser investigación de frontera. No obstante, integrar todo en un único grafo eleva la complejidad de ingeniería. Proponemos ir por una cascada de dos modelos (detector YOLOv8-s + verificador temporal tipo SmokeyNet-Lite sobre ROIs) en el MVP, y evolucionar luego al híbrido integrado. 

o Edge: MobileNetV3-Small / EfficientNet-Lite0 o YOLOv8-Nano cuantizados a INT8 con verificación temporal local son las opciones más realistas para Raspberry Pi; están alineadas con la literatura y con las recomendaciones del material base. 

# • Decisión final. 

1. Servidor (Fase inicial): Cascada YOLOv8-s (detector de alto recall) → Verificador temporal SmokeyNet-Lite ( $CNN+LSTM$  sobre tiles de ROIs) con lógica de persistencia 2–3frames. 

2. Servidor (Fase 2): Híbrido integrado $ConvNeXt-T/EfficientNet-B4+Temporal(LSTM)+$ Cabeza YOLOv8 anchor-free. 

3. Edge (Raspberry Pi): MobileNetV3-Small-INT8 (clasificación humo/no-humo) o YOLOv8-Nano-INT8 (detección), ambos con $\mathbf {QAT}+\mathbf {pr}$ uning y verificación temporal de 3–5frames. 

# Criterios de evaluación

• Desempeño temprano: recall alto en humo incipiente y tiempo a detección (min) reducido (Dewangan et al., 2022). 

• Robustez operacional: tolerancia a nubes, niebla, flare, lluvia; baja tasa de falsas alarmas por cámara/día. (AlterMundi, 2025). 

• Viabilidad ingenieril: complejidad de entrenamiento/inferencia, huella de memoria, trazabilidad y facilidad de MLOps (SAI-Benchmark/SAI-CAM). 

• Portabilidad edge: compatibilidad con Raspberry Pi (CPU ARM), INT8, latencia por frame y consumo. 

# Contexto y evidencia clave

•  $\text {FIgLib+SmokeyNet.}$  Dataset de ~24.8 k imágenes en secuencia (81 frames/ignición, 40 min antes/después). SmokeyNet $\text {(CNN+LSTM+ViT}$  con supervisión por tiles) rindió F $1\approx 82.6\%\mathrm {y}$ detección a ~3.1 min tras la aparición inicial de humo. (Dewangan et al., 2022). 

• Pyro-SDIS / Pyronear. Integración con Ultralytics YOLO (v8 y variantes optimizadas) sobre imágenes de torres; la elección anchor-free y los módulos C2f/PANet/SPPF favorecen objetos de formas/amplitudes variables (como el humo). 

• SAI-CAM / SAI-Benchmark. Infraestructura lista para capturar, subir y evaluar sistemáticamente modelos (suites YAML, resultados JSON, métricas – precisión, recall, F1, IoU, Early Fire Score). (AlterMundi, 2025). 

• Revisión de arquitecturas (doc interno). Se recomiendan ConvNeXt/EfficientNet como backbones eficientes, LSTM/Transformer para temporalidad y YOLOv8 para detección; en edge, MobileNetV3-Small/EfficientNet-Lite0 o YOLOv8-Nano con quantization/pruning/distillation y verificación temporal. (AlterMundi, 2025). 

# Análisis crítico de las opciones

¿SmokeyNet “tal cual” $detector+temporal?$ 

• SmokeyNet es sólido como clasificador espaciotemporal y muestra un rendimiento cercano a “humano” en FIgLib (F1 ~ 82.6 %). (Dewangan et al., 2022). 

o Fortalezas: explota bien la temporalidad (LSTM) y relaciones espaciales globales (ViT).

o Límites: no localiza explícitamente humo (difícil trazabilidad para operadores, más sensible a nubes globales); el pipeline de tiles a resolución alta consume memoria. 

• YOLOv8 (detector) + verificación temporal (p. ej., LSTM):

o Fortalezas: localización explícita, alto recall con umbrales bajos, mejor manejable con SAI-CAM/operadores; se puede verificar solo ROIs con un verificador temporal liviano (menor costo). (AlterMundi, 2025; Pyronear). 

o Riesgo: diseño de lógica de persistencia y de umbrales (balancear latencia vs falsas alarmas).(Dewangan et al., 2022 – ver discusión sobre umbrales dinámicos en trabajos relacionados). 

Conclusión: Para MVP en servidor, cascada detector→temporal ofrece mejor trazabilidad, mejor control de trade-offs y menor riesgo de integración que un gran modelo monolítico; a la vez, mantiene la innovación moderada (temporal + anchor-free + fusión ROIs) usando piezas probadas. En Fase 2, la arquitectura integrada recomendada en los informes internos es el camino natural de consolidación. 

## ¿Qué backbone temporal y de detección conviene?

• Backbone: ConvNeXt-T o EfficientNet-B4: ambos han mostrado alta relación precisión/eficiencia;ConvNeXt mantiene inductivas de CNN modernas; EfficientNet escala bien con costo controlado. (AlterMundi, 2025). 

• Temporal: LSTM simple gana en estabilidad/costo frente a Transformer temporal para ventanas cortas (2–5 frames). (Dewangan et al., 2022; AlterMundi, 2025). 

• Detección: YOLOv8 anchor-free por rapidez en GPU y buen manejo multi-escala; además es lo que propone la documentación de SAI para servidor. (AlterMundi, 2025; Pyronear). 

## ¿Qué es realista en Raspberry Pi?

• MobileNetV3-Small / EfficientNet-Lite0 (clasificación) o YOLOv8-Nano (detección), con INT8 y pruning; verificación de 3–5 frames para robustez. (AlterMundi, 2025). 

La documentación del proyecto insiste en la compresión para alcanzar tiempo real y consumo bajo, lo que valida esta elección. (AlterMundi, 2025). 

# Propuesta final de arquitectura

# Servidor (RTX 3090)

Fase inicial (recomendada para operación desde el día 1): Cascada “Detector + Verificador temporal”

<!-- Frame (1040×1856) ──► YOLOv8-s (conf. baja, p.ej. 0.3)   │   ├─► Si no hay ROI: no-alarma (y log)   ▼   ROIs (tiles 224×224) reciente[t-1:t]   │   └─► SmokeyNet-Lite (CNN backbone liviano + LSTM 2-frames)   │   └─► Verificación de persistencia ≥2–3 frames (por cámara)  ⇒ Alarma + bounding boxes + evidencia  -->
![](https://web-api.textin.com/ocr_image/external/ca369cc47155896c.jpg)

• YOLOv8-s como detector de alto recall (umbral 0.3–0.4, NMS moderado) para proponer ROIs.(Pyronear/Ultralytics; SAI). 

• SmokeyNet-Lite como verificador: backbone EfficientNet-B0/ResNet-34 sobre tiles de ROIs + LSTM (2 frames) (sin ViT para simplificar) + head binaria (humo/no humo). Basado en la eficacia temporal de SmokeyNet. (Dewangan et al., 2022). 

• Regla temporal: exigir persistencia de la detección en ≥2–3 frames para elevar precisión; si se baja el umbral de detección para ganar recall, compensar con esta persistencia (trade-off tiempo a alerta vs falsas alarmas). (AlterMundi, 2025; discusión en FIgLib/SmokeyNet). 

• Justificación: máxima trazabilidad (cajas), robustez ante artefactos, complejidad moderada (dos modelos estándar) y despliegue inmediato en la 3090. 

Fase 2 (consolidación): Modelo integrado (lo propuesto en los informes)

• Backbone: ConvNeXt-T o EfficientNet-B4/B5 (pre-entrenado).

• Temporal: LSTM (3–5 frames / 1–2 min); opción Transformer si el costo lo permite.

• Cabeza: YOLOv8 anchor-free con fusión espaciotemporal (p. ej., atención sobre mapas de distintas marcas temporales). (AlterMundi, 2025). 

Notas de entrenamiento (aplican a ambas fases): usar FIgLib (contornos → cajas para YOLO), PyroNear-2024, Nemo y FASDD, con Mosaic/CutMix/MixUp y síntesis (StyleGAN2-ADA) para cubrir humo tenue y dominios; optimizar mAP@0.5 y recall temprano, estableciendo umbral bajo + persistencia temporal. (Dewangan et al., 2022; AlterMundi, 2025). 

# Edge (Raspberry Pi)

Opción A — Clasificador ultraligero (cuando la Pi esté muy limitada):

• MobileNetV3-Small-INT8 (TFLite/ONNXRuntime-ARM/ncnn) con distillation desde el verificador del servidor; verificación temporal local (3–5 frames, mayoría). (AlterMundi, 2025). 

• Uso: pre-filtro local (alto recall); ante positivo persistente, enviar frame + metadatos al servidor (SAI-CAM ya gestiona colas/subida/health). 

Opción B — Detector ligero (si la Pi lo permite):

• YOLOv8-Nano-INT8 + pruning estructurado; NMS agresivo; confirmación por 3–5 frames.(AlterMundi, 2025). 

• Uso: enviar cajas + crops al servidor para corroboración.

Por qué estas opciones: el material del proyecto enfatiza que la compresión (INT8, pruning, distillation) es obligatoria para alcanzar tiempo real y consumo bajo en Raspberry Pi. (AlterMundi, 2025). 

# Plan de entrenamiento, validación y despliegue

# Datos y preparación 

• FIgLib: recorte de cielo, resize a $\sim 1040\times 1856$ , tiling $224\times 224$ ; usar contornos para tiles y cajas derivadas (entrenar detector y verificador). (Dewangan et al., 2022). 

 PyroNear-2024: diversidad geográfic $a+si$ ntético; mejora estabilidad inter-dataset. (AlterMundi,2025). 

• Nemo/FASDD: condiciones adversas/objetos pequeños, complementan cobertura.

• Aumentos: Mosaic, CutMix, MixUp, jitter de color/brillo; síntesis con StyleGAN2-ADA para humo tenue. (AlterMundi, 2025; Dewangan et al., 2022). 

# Entrenamiento 

• Servidor (fase inicial):

1. Entrenar YOLOv8-s con mezcla FIgL $ib+PyroNear-2024+Nemo+FASDD.$ 

2. Entrenar SmokeyNet-Lite con tiles de ROIs; positivizar con persistencia de 2 frames (data scheduling). 

3. Ajuste de umbrales: conf. detector baja (0.3–0.4), persistencia 2–3 frames; optimizar Early Fire Score y F1 en suites de SAI-Benchmark. 

• Edge: $\text {QAT+INT8+pruning}$ ; validar en la Pi con latencia, uso de RAM y falsos positivos/día;exportar a ncnn / TFLite / ONNXRuntime-ARM. (AlterMundi, 2025). 

# Validación (SAI-Benchmark) 

• Suites YAML con:

o Métricas: recall@temprano, F1, mAP@0.5, tiempo a detección (min), FAs/day/cam.

o Estratificación por cámara, hora del día, clima; hard-negative mining (nubes/bruma).

o Comparar: cascada vs integrado (si se entrena), ablar con/sin persistencia. (AlterMundi, 2025).

# Despliegue

• Servidor: exportar a ONNX y TensorRT (3090) para inferencia; pipeline asíncrono (SAI-CAM ya sube imágenes con metadatos y gestiona almacenamiento/health). 

• Edge: empaquetar binarios, watchdog y umbrales en config.yaml; modo local y subida diferida ya soportados por SAI-CAM. 

# Riesgos y mitigaciones

• Shift de dominio (Argentina vs CA/FR): iniciar con modelos entrenados multi-dataset y afinar con datos locales capturados por SAI-CAM (aprendizaje continuo/federado futuro). (AlterMundi, 2025). 

• Falsas alarmas por nubes/bruma: umbral bajo + persistencia temporal, hard-negative mining; posibilidad de umbral dinámico por franja horaria (véase enfoque similar discutido sobre umbralización dinámica en trabajos relacionados a FIgLib). (Dewangan et al., 2022). 

• Complejidad del modelo integrado: empezar por cascada (dos modelos estándares) con trazabilidad y tuning claros; migrar al integrado cuando SAI-Benchmark confirme mejoras. 

• Limitaciones en Raspberry Pi: obligatoria INT8 + pruning + verificación temporal; si no se alcanza rendimiento, usar clasificador pre-filtro y delegar detección al servidor. (AlterMundi, 2025). 

# Checklist de implementación (MVP → Fase 2)

• Entrenar YOLOv8-s (detector servidor) con mezcla FIgLib+PyroNear-2024+Nemo+FASDD.

• Entrenar SmokeyNet-Lite (verificador temporal en ROIs, 2 frames).

• Definir umbrales/persistencia (0.3–0.4; 2–3 frames) y evaluar con SAI-Benchmark.

• Exportar ONNX/TensorRT y desplegar en 3090; cerrar loop con operadores (cajas/mosaicos como evidencia). 

• Edge: QAT + INT8 de MobileNetV3-Small (opción A) o YOLOv8-Nano (opción B); validar latencia/consumo en Pi; integrar con SAI-CAM. 

• Fase 2: entrenar modelo integrado (ConvNeXt/EfficientNet + LSTM + cabeza YOLOv8) y comparar objetivamente vs cascada. 

# Referencias (APA)

• AlterMundi. (2025). Sistema de Alerta de Incendios (SAI): Diseño Arquitectónico y Redes Neuronales para Detección Temprana de Humo. (Informe técnico). 

• AlterMundi. (2025). Investigación sobre arquitecturas de redes neuronales para detección temprana de humo (SAI). (Informe técnico). 

• Dewangan, A., Pande, Y., Braun, H.-W., Vernon, F., Perez, I., Altintas, I., Cottrell, G. W., & Nguyen,M. H. (2022). FIgLib & SmokeyNet: Dataset and Deep Learning Model for Real-Time Wildland Fire Smoke Detection. Remote Sensing, 14(4), 1007. https://doi.org/10.3390/rs14041007. 

• AlterMundi. (2025). Revisión de repositorios: SAI-CAM y SAI-Benchmark. (Informe técnico DeepWiki).

• AlterMundi. (2025). Investigación arquitecturas neuronales (Pyro-SDIS, FIgLib y SmokeyNet). (Notas internas sobre datasets/arquitecturas). 

Notas finales sobre alcance y trazabilidad

• Este informe toma como base prioritaria los dos documentos subidos por ustedes y los complementa con archivos de proyecto pertinentes para asegurar una evaluación integral (SAI-CAM/Benchmark, Pyro-SDIS/FIgLib/SmokeyNet). 

• La propuesta final respeta el objetivo de innovar con seguridad: la cascada en servidor permite salir a producción con piezas robustas y explicables; el integrado (Fase 2) materializa la ambición de arquitectura única con temporalidad + detección; y en edge se garantiza viabilidad con modelos INT8 y verificación temporal local. (AlterMundi, 2025; Dewangan et al., 2022). 

## Resumen en una línea: 

Servidor hoy: YOLOv8-s → SmokeyNet-Lite (persistencia 2–3 frames). Servidor mañana:

ConvNeXt/EfficientNet + LSTM + cabeza YOLOv8. Edge: MobileNetV3-Small-INT8 o YOLOv8-Nano-INT8 + verificación 3–5 frames. 

