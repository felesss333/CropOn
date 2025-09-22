# CropOn! - Segmentación de Imágenes con SAM 2.1

**CropOn!** es una interfaz gráfica intuitiva y poderosa que te permite segmentar objetos en imágenes con precisión milimétrica usando el modelo **SAM 2.1** de Meta AI. Ideal para diseñadores, investigadores y entusiastas de la IA.

## Características

- Segmentación por puntos (foreground/background) y por cuadro delimitador.
- 3 máscaras sugeridas con scores de confianza.
- Post-procesamiento: suavizado de bordes y eliminación de ruido.
- Exporta en múltiples formatos: máscara PNG, imagen recortada con fondo transparente, o contornos en JSON.
- Interfaz moderna y responsiva con Gradio.

## Requisitos Previos

Antes de instalar CropOn!, asegúrate de tener lo siguiente:

1.  **Python 3.10 o superior**: Descárgalo desde [python.org](https://www.python.org/downloads/).
2.  **Git**: Para clonar este repositorio. Descárgalo desde [git-scm.com](https://git-scm.com/).
3.  **CUDA Toolkit (Opcional pero Recomendado)**: Si tienes una GPU NVIDIA, instala CUDA para un rendimiento óptimo. Descárgalo desde [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).
4.  **Modelo SAM 2.1**: Debes descargar manualmente el checkpoint `sam2.1_hiera_large.pt` desde el [repositorio oficial de SAM 2](https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints) y colocarlo en la carpeta `checkpoints/` de este proyecto.

## Instalación

Sigue estos pasos para tener CropOn! funcionando en tu computadora:

```bash
# 1. Clona este repositorio
git clone https://github.com/tu_usuario/CropOn.git
cd CropOn

# 2. Crea y activa un entorno virtual (¡Muy recomendado!)
python -m venv venv
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# 3. Instala las dependencias
pip install -r requirements.txt

# 4. Descarga el modelo SAM 2.1
# Visita: https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints
# Descarga 'sam2.1_hiera_large.pt' y colócalo en la carpeta './checkpoints/'

# 5. ¡Ejecuta la aplicación!
python crop_on_app.py 