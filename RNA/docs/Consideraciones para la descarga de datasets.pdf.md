<!--  -->

# Consideraciones para la descarga y uso de los

# datasets

Este documento complementa la guía de descarga y repasa los aspectos prácticos que debe tener en cuenta antes de bajar y preparar los conjuntos de datos necesarios para el Sistema de Alerta de Incendios (SAI). Incluye información sobre tamaño aproximado, formato de los archivos, licencias conocidas y dependencias para procesarlos.

Resumen comparativo de los datasets


| Dataset  | Nº<br>imágenes /secuencias  | Tamaño<br>aproximado  | Formato /<br>clases  | Licencia* | Notas y requisitos  | Notas y requisitos  | Notas y requisitos  | Notas y requisitos  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FASDD (ofcial)  | ND –<br>según el<br>artículo<br>son más<br>de 100 000imágenes<br>pero el<br>dataset no está<br>publicado<br>1  | n/a | Imágenes con bounding<br>boxes (fame, smoke)  | Información no<br>disponible  | El dataset ofcial no se puede descargar;<br>únicamente existe el subconjunto <br>FASDD_CV en Kaggle (≈12 GB, 286 000<br>imágenes) que exige autenticación. | El dataset ofcial no se puede descargar;<br>únicamente existe el subconjunto <br>FASDD_CV en Kaggle (≈12 GB, 286 000<br>imágenes) que exige autenticación. | El dataset ofcial no se puede descargar;<br>únicamente existe el subconjunto <br>FASDD_CV en Kaggle (≈12 GB, 286 000<br>imágenes) que exige autenticación. | El dataset ofcial no se puede descargar;<br>únicamente existe el subconjunto <br>FASDD_CV en Kaggle (≈12 GB, 286 000<br>imágenes) que exige autenticación. |
| D‑Fire | 21 k<br>imágenes<br>con cajas<br>2  | Se estima<br>~3–4 GB | ZIP con<br>carpetas <br>images/ y <br>labels/ ;<br>anotaciones<br>en formato<br>YOLO<br>normalizado  | Puede usarse para investigación;<br>verifcar términos<br>del repo  | Las UR<br>a veces token; opción dispos que wg nombr  | Ls de O  requie utilice la --con ition  | Ls de O  requie utilice la --con ition  | neDrive ren<br>tent- para<br>pete el chivo.  |
| D‑Fire | 21 k<br>imágenes<br>con cajas<br>2  | Se estima<br>~3–4 GB | ZIP con<br>carpetas <br>images/ y <br>labels/ ;<br>anotaciones<br>en formato<br>YOLO<br>normalizado  | Puede usarse para investigación;<br>verifcar términos<br>del repo  | Las UR<br>a veces token; opción dispos que wg nombr  | et <br>e de  |  res<br>l ar  | neDrive ren<br>tent- para<br>pete el chivo.  |
| NEMO  | 2934<br>imágenes<br>3  | ≈1.12 GB<br>(formato<br>Supervisely);1.0 GB via <br>dataset-<br>tools4  | Imágenes<br>JPEG y<br>anotaciones<br>COCO;<br>subclases low/mid/high<br>smoke  | Apache 2.0 según<br>dataset-ninja;<br>revisar sección de<br>licencia  | Puede descargarse<br>con la librería <br>dataset-tools o<br>vía Kaggle.<br>Dependiendo del<br>método, generará<br>una jerarquía <br>images/ , <br>annotations.json<br>o un paquete<br>Supervisely.  | Puede descargarse<br>con la librería <br>dataset-tools o<br>vía Kaggle.<br>Dependiendo del<br>método, generará<br>una jerarquía <br>images/ , <br>annotations.json<br>o un paquete<br>Supervisely.  | Puede descargarse<br>con la librería <br>dataset-tools o<br>vía Kaggle.<br>Dependiendo del<br>método, generará<br>una jerarquía <br>images/ , <br>annotations.json<br>o un paquete<br>Supervisely.  | Puede descargarse<br>con la librería <br>dataset-tools o<br>vía Kaggle.<br>Dependiendo del<br>método, generará<br>una jerarquía <br>images/ , <br>annotations.json<br>o un paquete<br>Supervisely.  |
| Dataset  | Nº<br>imágenes /secuencias  | Tamaño<br>aproximado  | Formato /Licencia*<br>clases  | Formato /Licencia*<br>clases  | Notas y requisitos  | Notas y requisitos  | Notas y requisitos  | Notas y requisitos  |
| Figlib /<br>SmokeExample  | ~19 000<br>imágenes<br>5<br>agrupadas en<br>secuencias;<br>3 clases<br>(cloud,<br>other,<br>smoke)  | ~132 MB | Archivo TAR<br>que<br>descomprime<br>MIT (según<br>repositorio <br>sagecontinuum/smokedataset ) 6<br><img src="https://web-api.textin.com/ocr_image/external/c894af229d6b3702.jpg"> | Archivo TAR<br>que<br>descomprime<br>MIT (según<br>repositorio <br>sagecontinuum/smokedataset ) 6<br><img src="https://web-api.textin.com/ocr_image/external/c894af229d6b3702.jpg"> | Para descomprimir<br>basta tar -xvf .<br>Los CSV indican la<br>clase de cada imagen.  | Para descomprimir<br>basta tar -xvf .<br>Los CSV indican la<br>clase de cada imagen.  | Para descomprimir<br>basta tar -xvf .<br>Los CSV indican la<br>clase de cada imagen.  | Para descomprimir<br>basta tar -xvf .<br>Los CSV indican la<br>clase de cada imagen.  |
| Pyro‑SDIS<br>(PyroNear 2024) | 33.6 k<br>imágenes<br>(29.5 k<br>train, 4.1 k<br>val)7<br><img src="https://web-api.textin.com/ocr_image/external/93793f52bbdadb1a.jpg"> | ~3.28 GB | Conjunto de<br>archivos .parquet<br>y <br>data.yamlApache 2.0<br>; una<br>7<br>única clase<br>“smoke” | Conjunto de<br>archivos .parquet<br>y <br>data.yamlApache 2.0<br>; una<br>7<br>única clase<br>“smoke” | Necesita datasets de Hugging Face o <br>git lfs . Las<br>imágenes están<br>embebidas en<br>columnas tipo <br>image dentro de los Parquet; se requiere conversión a JPEG<br>para modelos YOLO.  | Necesita datasets de Hugging Face o <br>git lfs . Las<br>imágenes están<br>embebidas en<br>columnas tipo <br>image dentro de los Parquet; se requiere conversión a JPEG<br>para modelos YOLO.  | Necesita datasets de Hugging Face o <br>git lfs . Las<br>imágenes están<br>embebidas en<br>columnas tipo <br>image dentro de los Parquet; se requiere conversión a JPEG<br>para modelos YOLO.  | Necesita datasets de Hugging Face o <br>git lfs . Las<br>imágenes están<br>embebidas en<br>columnas tipo <br>image dentro de los Parquet; se requiere conversión a JPEG<br>para modelos YOLO.  |


<!-- 1 -->

*La tabla resume licencias según la información disponible en agosto de 2025. Revise siempre los archivos LICENSE de cada repositorio o dataset para asegurarse de cumplir con los términos.

# Dependencias y herramientas necesarias

1. Línea de comandos y paquetes de Python: las descargas emplean wget , unzip , tar y las bibliotecas de Python dataset-tools , datasets , huggingface-hub y pandas . Se instalan con pip install y con el gestor de paquetes del sistema ( sudo apt install unzip git-lfs ). 

2. Git LFS: para datasets hospedados en Hugging Face con archivos grandes (Parquet), es recomendable tener instalado Git LFS y ejecutar git lfs install antes de clonar. 

3. Kaggle CLI: imprescindible para descargar FASDD_CV y, si se desea, la versión Kaggle de NEMO.Requiere una cuenta y token de API. 

4. Espacio en disco: aunque algunos archivos se pueden descomprimir parcialmente, se recomienda disponer de al menos 10 GB libres para alojar todos los datasets sin problemas.

# Consideraciones sobre el tratamiento de datos

• Normalización de anotaciones: D‑Fire y Pyro‑SDIS proporcionan cajas normalizadas. Para entrenar con frameworks como YOLOv8, los parquets de Pyro‑SDIS deben convertirse a archivos *.txt y las coordenadas normalizadas deben transformarse a píxeles. La librería datasets permite exportar los Parquet a arrays NumPy y con Pandas se pueden reconstruir las cajas. 

• Datos no publicados: dado que FASDD ofcial no está disponible, se sugiere complementar con otros datasets (D‑Fire, Nemo, Pyro‑SDIS y smoke-example) que sí cubren detección de humo. • Licencias: asegúrese de revisar las licencias. Pyro‑SDIS y NEMO se distribuyen bajo Apache 2.0,mientras que smoke-example utiliza licencia MIT y D‑Fire no especifca explícitamente una licencia pública. Para usos comerciales o proyectos públicos, obtenga el consentimiento de los autores. 

<!-- 2 -->

• Ética y privacidad: aunque estos datasets se centran en incendios forestales, algunos incluyen imágenes de infraestructuras o paisajes captadas por cámaras públicas. Considere las implicaciones de privacidad y cumpla con las regulaciones locales al usar los datos. 

• Verifcación de integridad: tras cada descarga, verifque que los archivos no estén corruptos (por ejemplo, con md5sum ). Errores de descarga pueden provocar fallos en el entrenamiento.

• Curación y limpieza: algunos datasets incluyen imágenes sin humo o con condiciones meteorológicas adversas. La documentación del SAI recomienda fltrar y etiquetar negativos duros para mejorar la robustez de los modelos.

# Recomendaciones fnales

• Documente cada paso de la descarga y extracción, de modo que el proceso sea reproducible para otros miembros del equipo. 

• Considere escribir scripts que automaticen las descargas y verifquen las dependencias,minimizando el error humano. 

• Integre las rutas de los datasets en la confguración del SAI (p. ej., en el archivo data.yaml de YOLO) para que el pipeline de entrenamiento localice correctamente las imágenes. 

• Mantenga un registro de las versiones de los datasets utilizadas; algunos repositorios como Pyro‑SDIS pueden actualizarse periódicamente.

1 An open fame and smoke detection dataset for deep learning in remote sensing based fre detection

https://www.scidb.cn/en/detail

2 GitHub - gaiasd/DFireDataset: D-Fire: an image data set for fre and smoke detection.

https://github.com/gaiasd/DFireDataset

34 NEMO - Dataset Ninja

https://datasetninja.com/nemo

5 README.md · sagecontinuum/smokedataset at main

https://huggingface.co/datasets/sagecontinuum/smokedataset/blob/main/README.md

6 smokedataset.py · sagecontinuum/smokedataset at main

https://huggingface.co/datasets/sagecontinuum/smokedataset/blob/main/smokedataset.py

7 data.yaml · pyronear/pyro-sdis at main

https://huggingface.co/datasets/pyronear/pyro-sdis/blob/main/data.yaml

<!-- 3 -->

