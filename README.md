<p align="center">
  <img src="./logo_cropon.svg" alt="CropOn! Logo" width="100" />
</p>

# CropOn! 
**Segmentación de Imágenes con SAM 2.1**

---

### 📖 Descripción

**CropOn!** es una interfaz gráfica intuitiva y poderosa que te permite segmentar objetos en imágenes con precisión milimétrica usando el modelo **SAM 2.1** de Meta AI. Ideal para diseñadores, investigadores y entusiastas de la IA.

- Segmentación por puntos (foreground/background) y por cuadro delimitador.
- 3 máscaras sugeridas con scores de confianza.
- Post-procesamiento: suavizado de bordes y eliminación de ruido.
- Exporta en múltiples formatos: máscara PNG, imagen recortada con fondo transparente, o contornos en JSON.

---

### ⚙️ Construido con

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Python_logo_and_wordmark.svg/2560px-Python_logo_and_wordmark.svg.png" height="42"> 
<img src="https://raw.githubusercontent.com/gradio-app/gradio/main/readme_files/gradio.svg" height="48">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/PyTorch_logo_black.svg/488px-PyTorch_logo_black.svg.png?20200318230141" height="30">

---

### 📦 Dependencias

* `torch`
* `torchvision`
* `gradio`
* `numpy`
* `Pillow`
* `opencv-python`
* `pynput`
* `hydra-core`
* `omegaconf`

---

### 📥 Instalación

Sigue estos pasos para tener CropOn! funcionando en tu computadora:

**1. Clona este repositorio**
git clone https://github.com/felesss333/CropOn.git
cd CropOn

**2. Crea y activa un entorno virtual**
python -m venv venv

***En Windows:***
venv\Scripts\activate

 ***En macOS/Linux:***
source venv/bin/activate

**3. Instala las dependencias**
pip install -r requirements.txt

**4. Instala el paquete SAM2 en modo editable**
cd segment-anything-2
pip install -e .
cd ..

**5. Descarga el modelo SAM 2.1**
- Visita: https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints
- Descarga ***'sam2.1_hiera_large.pt'*** y colócalo en la carpeta ***'./checkpoints/'***

**6. ¡Ejecuta la aplicación!**
python cropon_full.py

---

**Nota:** La primera ejecución puede tardar unos minutos mientras se cargan los modelos y se compilan componentes.

---

### 🧭 Uso
Una vez que la aplicación se esté ejecutando, se abrirá automáticamente en tu navegador web predeterminado:  
👉 [http://127.0.0.1:7860](http://127.0.0.1:7860)

Sigue estos pasos:

- **Carga una imagen**:  
  Arrástrala y suéltala en el cuadro "Entrada" o haz clic para seleccionarla.

- **Segmenta**:  
  - Haz clic en el objeto que deseas seleccionar (punto verde).  
  - Mantén presionada `Shift` y haz clic para marcar áreas que deseas excluir (punto rojo).  
  - Activa **"Modo Rectángulo"** y haz dos clics para dibujar un cuadro alrededor del objeto.

- **Refina**:  
  Usa los controles de **"Tono"**, **"Opacidad"** y **"Post-Procesamiento"** para ajustar la vista.

- **Exporta**:  
  Elige el formato de exportación, ponle un nombre y haz clic en **"Guardar Máscara"**.  
  Los archivos se guardarán en la carpeta `./masks/`.
---

### 📜 Licencia
CropOn! está bajo una licencia de Uso Libre. Puedes usarlo, modificarlo y redistribuirlo libremente, incluso para uso comercial. Pero no puedes vender el código fuente original tal cual sin permiso del autor.

Consulta el archivo LICENSE para más detalles.

---

### 📬 Datos de contacto

[Linkedin](https://www.linkedin.com/in/felipe-dorado-29315232/)
[Github](https://github.com/felesss333/)
[Behance](https://www.behance.net/Felipedorado)

Email: felipehdorado@gmail.com

