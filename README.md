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

### 📥 Requisitos del sistema 

🔹 Si tienes GPU NVIDIA (recomendado):
No necesitas hacer nada extra — el requirements.txt ya incluye las versiones correctas.

🔹 Si NO tienes GPU NVIDIA (solo CPU):
Puedes reemplazar esas 3 líneas en requirements.txt por:

torch==2.8.0+cpu
torchvision==0.23.0+cpu
torchaudio==2.8.0+cpu

---

### 📥 Instalación

🔹 **Programas necesarios:**  
✅ [Git](https://git-scm.com/)  
✅ [Python 3.9 o superior](https://www.python.org/downloads/)


<br>

**1. Clonar el repositorio CropOn**
```text
git clone https://github.com/felesss333/CropOn.git
```

<br>

**2. Entrar en la carpeta del proyecto**
```text
cd CropOn
```

<br>

**3. Crear entorno virtual**
```text
python -m venv venv
```
<br>

**4. Activar entorno virtual**
```text
source venv/Scripts/activate
```

<br>

**5. Descargar Modelo SAM2 en la ruta específica del proyecto**
```text
git clone https://github.com/facebookresearch/sam2.git segment-anything-2
```

<br>

**6. Entrar en la carpeta clonada**
```text
cd segment-anything-2
```

<br>

**7. Instalar el paquete en modo editable**
```text
pip install -e .
```

<br>

**8. Crear carpeta "conf" dentro de "sam2"**
```text
mkdir sam2/conf
```

<br>

**9. Mover TODO el contenido de "configs" a "conf"**
```text
mv sam2/configs/* sam2/conf/
```

<br>

**10. Copiar el archivo de configuración clave al nivel superior de 'conf'**
```text
cp sam2/conf/sam2.1/sam2.1_hiera_l.yaml sam2/conf/
```

<br>

**11. Volver a la carpeta CropOn**
```text
cd ..
```

<br>

**12. Instalar dependencias del proyecto**
```text
pip install -r requirements.txt
```
<br>

**13. Crear carpeta para el modelo**
```text
mkdir checkpoints
```
<br>

**14. Descargar modelo SAM2.1**
Descarga sam2.1_hiera_large.pt desde:

🔗 [https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints](https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints)
```text
Guarda el archivo dentro de la carpeta ./checkpoints/ que acabas de crear.
```
<br>

**15. Crear carpeta para las máscaras**
```text
mkdir masks
```
<br>

**16. Verificar que los archivos del modelo están en las rutas correctas.**
*(SAM2 no funciona con rutas relativas, por ello la app funciona con rutas específicas)*

**17. Archivo "sam2.1_hiera_large.pt" debe estar en:**
```text
CropOn/checkpoints
```
<br>

**18. Archivo "sam2.1_hiera_l.yaml" debe estar en:**
```text
CropOn/segment-anything-2/sam2/conf
```
<br>

**19. ¡Ejecutar la aplicación!**
```text
python cropon_full.py
```

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


