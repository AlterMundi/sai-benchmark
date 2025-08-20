<!--  -->

# Guía de descarga de datasets para el SAI

Este documento resume los pasos concretos para descargar en bash los datasets necesarios para entrenar las redes neuronales del sistema de alerta de incendios (SAI). Se asume un entorno Debian y que se trabaja dentro del directorio del proyecto ( RNA/data/raw/ ). Antes de ejecutar cualquier descarga conviene crear la estructura de carpetas mostrada en la especifcación:

mkdir -p RNA/data/raw/{fasdd,dfire,nemo,figlib,pyronear-2024}

Posteriormente, siga las instrucciones específcas para cada conjunto de datos.

# 1. FASDD (Flame and Smoke Detection Dataset)

Estado: el equipo de investigación que publica FASDD anunció el dataset en 2023–2024, pero a la fecha (19‑08‑2025) no hay una publicación ofcial; la página de la base de datos china scidb.cn indica que no 1hay registros publicados. Sin embargo, existe un subconjunto no ofcial denominado FASDD_CV alojado en Kaggle, con ~286 000 imágenes y anotaciones. Para usarlo se necesita una cuenta de Kaggle.

## Descarga con Kaggle CLI:

1. Crear y validar una cuenta en kaggle.com y generar un «API Token» desde Account → API.

** Información del dataset dentro de kaggle:

"import kagglehub

# Download latest version
path = kagglehub.dataset_download("yuulind/fasdd-cv-coco")

print("Path to dataset files:", path)" 

Credenciales de usuario:

user: marianofm@gmail.com
pass: ScP9hu8mXiSDhwq

2. Instalar la herramienta de Kaggle: 


| pip install --upgrade kaggle  |
| --- |


# 3. Copiar el archivo kaggle.json descargado desde Kaggle a ~/.kaggle/ y proteger los permisos: 


| mkdir -p ~/.kaggle<br>cp ~/Downloads/kaggle.json ~/.kaggle/<br>chmod 600 ~/.kaggle/kaggle.json |
| --- |


4. Descargar el conjunto FASDD_CV. El identifcador del dataset puede cambiar; actualmente se llama juanpaucarmona/fasdd-cv-fire-smoke (ver la página de Kaggle). Sitúese en la carpeta correspondiente y ejecute: 


| cd RNA/data/raw/fasdd<br>kaggle datasets download -d juanpaucarmona/fasdd-cv-fire-smoke --unzip  |
| --- |


Alternativa: si en el futuro el consorcio publica FASDD de forma ofcial, se deberá obtener el enlace de descarga indicado en la publicación y usar wget --content-disposition $"<url_del_zip>"-0$  fasdd.zip para recuperarlo.

<!-- 1 -->

# 2. D‑Fire (fre & smoke detection)

El repositorio DFireDataset explica que D‑Fire contiene más de 21 000 imágenes con anotaciones en 2formato YOLO. Se distribuye como dos archivos en la nube de Microsoft (OneDrive): uno con imágenes y etiquetas y otro con los archivos de división en train/val/test. Para descargarlos:2

cd RNA/data/raw/dfire

# 1. Descargue las imágenes y etiquetas

wget --content-disposition "https: $//1drv.ms/u/s!<identificador>?e=<token>"-$ O dfire_images_labels.zip

# 2. Descargue los conjuntos de entrenamiento, validación y prueba wget --content-disposition "https://1drv.ms/u/s!&lt;identificador $2>\text {?e}=<\text {token}>\text {"}$ -O dfire_splits.zip

# 3. Descomprima los contenidos

unzip dfire_images_labels.zip

unzip dfire_splits.zip

# 4. (opcional) Convierta las cajas normalizadas a píxeles usando el script utils/yolo2pixel

Nota: deberá sustituir los marcadores &lt;identificador&gt; y &lt;token&gt; por las URL completas que fguran en el README del repositorio. La opción --content-disposition hace que wget utilice el nombre de archivo sugerido por el servidor, lo que facilita la extracción.

# 3. NEMO (Nevada Smoke Detection Dataset)

NEMO es un benchmark de detección de humo con ~2934 imágenes y tres niveles de densidad. El portal Dataset Ninja explica que puede descargarse en formato Supervisely $\approx 1.12\mathrm {\sim GB}$ ) o mediante la 3librería dataset-tools. Existen también versiones alojadas en Kaggle.

Opción A: Descargar con dataset-tools (recomendado)

1. Instale la librería: 


| pip install --upgrade dataset-tools  |
| --- |


2. Descargue el dataset ejecutando el siguiente script de Python, que almacenará las imágenes y anotaciones en RNA/data/raw/nemo : 


| cd RNA/data/raw/nemo<br>python3 -c "import dataset_tools as dtools; <br>dtools.download(dataset='NEMO', $\text {"}}$ |
| --- |


<!-- 2 -->

## Opción B: Descargar vía Kaggle

Si prefere usar Kaggle, tras confgurar la API como se indica en la sección de FASDD:


| cd RNA/data/raw/nemo<br>kaggle datasets download -d nemares/nemo-smoke-detection-dataset --unzip  |
| --- |


El identifcador exacto ( nemares/... ) puede variar; consúltelo en Kaggle.

# 4. Figlib / SmokeExample (secuencias temporales)

El documento de investigación del SAI sugiere usar un dataset de secuencias temporales derivado de HPWREN, referenciado como FIgLib y distribuido en el proyecto SmokeyNet. La librería de Hugging Face sagecontinuum/smokedataset señala que las imágenes se descargan desde un archivo tar externo llamado smoke-example.tar. Para descargarlo:4

cd RNA/data/raw/figlib

# Descargue el archivo tar (~132 MB)

wget https://web.lcrc.anl.gov/public/waggle/datasets/smoke-example.tar

# Extraiga las imágenes y etiquetas (creará carpetas train/val/test)tar -xvf smoke-example.tar

Este dataset contiene ~19 000 imágenes distribuidas en tres clases (cloud, other, smoke) con un tamaño total de unos 132 MB, según la descripción de la librería. Su estructura ya incluye los fcheros5train.csv , val.csv y test.csv que indican las correspondencias entre imágenes y etiquetas.

# 5. PyroNear 2024 / Pyro‑SDIS

El dataset Pyro‑SDIS (también llamado PyroNear   2024) provee más de 33 000 imágenes con anotaciones de humo, divididas en 29.5   k para entrenamiento y 4.1   k para validación. Está6licenciado bajo Apache 2.0 y hospedado en Hugging Face, donde se distribuye en formato Parquet (6archivos .parquet de entrenamiento y 1 de validación) y un archivo data.yaml que describe la ruta para entrenamiento y la clase “smoke”. Para descargarlo:6

# Opción A: Usar la librería datasets (rápido y simple)

1. Instale las dependencias: 

pip install datasets huggingface-hub pandas

2. Ejecute el siguiente script de Python para descargar los datos en RNA/data/raw/pyronear-2024 : ```bash cd RNA/data/raw/pyronear-2024 python3 - &lt;&lt;'PY' from datasets import load_dataset

<!-- 3 -->

# Descargar el conjunto de entrenamiento

dataset = load_dataset('pyronear/pyro-sdis', split='train') dataset.save_to_disk('pyro_train')

# Descargar el conjunto de validación

val = load_dataset('pyronear/pyro-sdis', split='val') val.save_to_disk('pyro_val') PY 

 Este método descarga y transforma automáticamente los archivos Parquet a un formato local legible por datasets`. Tenga en cuenta que el peso total ronda los 3.3 GB.6

Opción B: Clonar vía git lfs

Alternativamente, puede clonar el repositorio de datos entero (incluyendo data.yaml ) mediante Git LFS:

sudo apt-get install git-lfs

# Inicializar Git LFS

git lfs install

cd RNA/data/raw/pyronear-2024

git clone https://huggingface.co/datasets/pyronear/pyro-sdis

Una vez clonado, las imágenes no están en formato JPEG sino incrustadas en archivos .parquet ; se pueden convertir a YOLO v5/v8 mediante scripts específcos de Ultralytics o cargando los Parquet con Pandas y extrayendo las imágenes (columna image ). El archivo data.yaml situado en la raíz del repositorio contiene las rutas y clases.6

# Resumen de comandos

El siguiente bloque sintetiza los comandos principales. Reemplace variables según corresponda y ejecute en la raíz del proyecto:

mkdir -p RNA/data/raw/{fasdd,dfire,nemo,figlib,pyronear-2024}

# FASDD (Kaggle)

cd RNA/data/raw/fasdd

pip install --upgrade kaggle

# Configurar credenciales Kaggle antes de ejecutar:

kaggle datasets download -d juanpaucarmona/fasdd-cv-fire-smoke --unzip

# D‑Fire

cd ../dfire

wget --content-disposition "&lt;enlace_onedrive_imagenes_labels&gt;" -O

<!-- 4 -->


| dfire_images_labels.zip<br>wget --content-disposition "&lt;enlace_onedrive_splits&gt;" -O dfire_splits.zip<br>unzip dfire_images_labels.zip<br>unzip dfire_splits.zip<br># NEMO<br>cd ../nemo<br>pip install --upgrade dataset-tools<br>python3 -c "import dataset_tools as dtools; dtools.download(dataset='NEMO', $\text {"}}$# Figlib / smoke-example<br>cd ../figlib<br>wget https://web.lcrc.anl.gov/public/waggle/datasets/smoke-example.tar<br>tar -xvf smoke-example.tar<br># PyroNear‑2024<br>cd ../pyronear-2024<br>pip install datasets huggingface-hub<br>python3 - &lt;&lt;'PY'<br>from datasets import load_dataset<br>dataset = load_dataset('pyronear/pyro-sdis', $\text {split='train'}$)<br>dataset.save_to_disk('pyro_train')<br>$val=load_dataset('pyronear/pyro-sdis',$ $\text {split='val}$')<br>val.save_to_disk('pyro_val')<br>PY  |
| --- |


Importante: algunas descargas requieren autenticación (Kaggle) o un token (OneDrive).Verifque que su equipo disponga de conexión estable y de al menos 10 GB de espacio libre para albergar todos los datos.

1 An open fame and smoke detection dataset for deep learning in remote sensing based fre detection

https://www.scidb.cn/en/detail

2 GitHub - gaiasd/DFireDataset: D-Fire: an image data set for fre and smoke detection.

https://github.com/gaiasd/DFireDataset

3 NEMO - Dataset Ninja

https://datasetninja.com/nemo

4 smokedataset.py · sagecontinuum/smokedataset at main

https://huggingface.co/datasets/sagecontinuum/smokedataset/blob/main/smokedataset.py

5 README.md · sagecontinuum/smokedataset at main

https://huggingface.co/datasets/sagecontinuum/smokedataset/blob/main/README.md

6 data.yaml · pyronear/pyro-sdis at main

https://huggingface.co/datasets/pyronear/pyro-sdis/blob/main/data.yaml

<!-- 5 -->

