# --- IMPORTS ---
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import torch
import os
import uuid
import shutil
import subprocess
import platform
from datetime import datetime
import sys
import time
import tempfile
import threading
from pynput import keyboard
import cv2
import json 

# --- Cargar modelo SAM 2 ---
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Cargando modelo SAM 2.1...")

# --- Rutas relativas al proyecto ---
CHECKPOINT_PATH = os.path.join("checkpoints", "sam2.1_hiera_large.pt")
CONFIG_PATH = os.path.join("segment-anything-2", "sam2", "conf", "sam2.1_hiera_l.yaml")

# Verificar existencia de archivos
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"No se encontró el checkpoint: {CHECKPOINT_PATH}")
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"No se encontró el archivo de configuración: {CONFIG_PATH}")

# --- Cargar el modelo con rutas absolutas ---
sam2_model = build_sam2(os.path.abspath(CONFIG_PATH), os.path.abspath(CHECKPOINT_PATH), device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)
print("✅ ¡Modelo SAM 2.1 cargado correctamente!")

# --- Variables globales ---
points = []
labels = []
brush_strokes = []
current_image = None
current_image_np = None
mask_color_hue = 125 
mask_alpha = 0.5 
is_processing = False
image_loaded = False
shift_pressed = False
original_image_pil = None
original_width = 0
original_height = 0
sam_max_size = 1024
point_radius = 20
inner_radius = 10
# --- NUEVAS VARIABLES PARA FUNCIONALIDADES AGREGADAS ---
box_points = []          # Para modo bounding box
box_mode_active = False  # Control visual del modo cuadro
current_masks = None     # Almacena las 3 máscaras multimask
current_scores = None    # Almacena los 3 scores
selected_mask_idx = 0    # Índice de la máscara seleccionada (0, 1, 2)
# --- Función para manejar eventos del teclado con pynput ---
def on_key_press(key):
    global shift_pressed
    try:
        if key in [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r]:
            shift_pressed = True
    except AttributeError:
        pass
def on_key_release(key):
    global shift_pressed
    try:
        if key in [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r]:
            shift_pressed = False
    except AttributeError:
        pass
# --- Iniciar el listener de teclado en un hilo separado ---
listener_thread = threading.Thread(target=lambda: keyboard.Listener(on_press=on_key_press, on_release=on_key_release).start(), daemon=True)
listener_thread.start()
print("Listener de teclado (pynput) iniciado.")
MASKS_DIR = "masks"
os.makedirs(MASKS_DIR, exist_ok=True)
# --- UTILS ---
def resize_output_image(pil_img, max_size=1200):
    """Redimensiona la imagen manteniendo proporción si excede max_size en ancho o alto."""
    if pil_img is None:
        return None
    w, h = pil_img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return pil_img
# --- NUEVA FUNCIÓN: Manejar carga de imagen ---
def handle_image_load(input_img):
    global points, labels, brush_strokes, image_loaded, original_image_pil, original_width, original_height, current_image, current_image_np
    global box_points, current_masks, current_scores, selected_mask_idx
    if input_img is None:
        # Limpiar todo
        points.clear()
        labels.clear()
        brush_strokes.clear()
        box_points.clear()
        current_masks = None
        current_scores = None
        selected_mask_idx = 0
        image_loaded = False
        original_image_pil = None
        original_width = 0
        original_height = 0
        current_image = None
        current_image_np = None
        predictor._features = None
        predictor._is_image_set = False
        return None, "Imagen y resultados limpiados.", gr.update(value=False), gr.update(value="Máscara 1"), gr.update(value="Esperando segmentación...")
    else:
        # Mostrar la imagen directamente en output_image
        return input_img, "✅ Imagen cargada. Haz clic para agregar puntos.", gr.update(value=False), gr.update(value="Máscara 1"), gr.update(value="Esperando segmentación...")
# --- RECARGAR IMAGEN ---
def reload_image():
    return gr.update(value=None), "⚠️ Por favor, vuelva a arrastrar o seleccionar la misma imagen en el cuadro 'Imagen Original'."
# --- POST-PROCESAMIENTO OPCIONAL ---
def postprocess_mask(mask, smooth=False, clean=False, min_area=300, sigma=1.0):
    mask_uint8 = mask.astype(np.uint8) * 255  # Convertir a 0-255
    if clean and min_area > 0:
        # Eliminar componentes pequeños
        num_labels, labels_im = cv2.connectedComponents(mask_uint8)
        for i in range(1, num_labels):
            component_mask = (labels_im == i).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours and cv2.contourArea(contours[0]) < min_area:
                mask_uint8[labels_im == i] = 0
        # --- NUEVO: LLENAR AGUJEROS si clean está activado ---
        if clean:
            mask_bool = mask_uint8 > 127
            mask_bool = fill_holes(mask_bool)
            mask_uint8 = mask_bool.astype(np.uint8) * 255
    if smooth and sigma > 0:
        mask_uint8 = cv2.GaussianBlur(mask_uint8, (0, 0), sigma)
        mask_uint8 = (mask_uint8 > 127).astype(np.uint8) * 255
    return mask_uint8 > 127  # Devolver como bool

# --- NUEVA FUNCIÓN AUXILIAR: LLENAR AGUJEROS ---
def fill_holes(mask):
    """Llena los agujeros en una máscara binaria."""
    if mask is None or mask.size == 0:
        return mask
    # Convertir a uint8
    mask_uint8 = mask.astype(np.uint8)
    h, w = mask_uint8.shape
    # Crear una copia para trabajar
    filled = mask_uint8.copy()
    # Crear una imagen con borde para floodFill
    bordered = np.zeros((h + 2, w + 2), np.uint8)
    # Llenar desde los bordes exteriores
    # Primero, llenamos todo el fondo conectado al borde con un valor temporal (2)
    cv2.floodFill(filled, bordered, (0, 0), 2, flags=4)
    # Luego, convertimos todo lo que NO es 2 (es decir, el objeto y sus agujeros) a 1 (foreground)
    filled = (filled != 2).astype(np.uint8)
    return filled.astype(bool)
    
# --- Función de segmentación (puntos o bounding box) ---
def segment_image(image, force_background_mode_checkbox, box_mode_toggle, evt: gr.SelectData):
    global points, labels, brush_strokes, current_image, current_image_np, mask_color_hue, mask_alpha, is_processing, image_loaded
    global original_image_pil, original_width, original_height, sam_max_size, point_radius, inner_radius
    global box_points, box_mode_active, current_masks, current_scores, selected_mask_idx
    if is_processing:
        return None, "⏳ Por favor, espera a que termine el procesamiento anterior.", gr.update(value=box_mode_toggle), gr.update(value="Máscara 1"), gr.update(value="Esperando...")
    is_processing = True
    try:
        if image is None:
            is_processing = False
            return None, "⚠️ Por favor, sube una imagen primero.", gr.update(value=box_mode_toggle), gr.update(value="Máscara 1"), gr.update(value="Esperando...")
        if not image_loaded:
            points.clear()
            labels.clear()
            brush_strokes.clear()
            box_points.clear()
            image_loaded = True
            if hasattr(image, 'filename') and image.filename:
                original_path = image.filename
                try:
                    temp_dir = tempfile.gettempdir()
                    unique_suffix = uuid.uuid4().hex[:8]
                    temp_copy_path = os.path.join(temp_dir, f"gradio_sam2_{unique_suffix}.jpg")
                    shutil.copy2(original_path, temp_copy_path)
                    original_image_pil = Image.open(temp_copy_path).convert('RGB')
                except Exception as e:
                    is_processing = False
                    return None, f"❌ Error al copiar y recargar imagen: {str(e)}", gr.update(value=box_mode_toggle), gr.update(value="Máscara 1"), gr.update(value="Esperando...")
            else:
                if image.mode != 'RGB':
                    original_image_pil = image.convert('RGB')
                else:
                    original_image_pil = image
            original_width, original_height = original_image_pil.size
            def resize_for_sam(pil_img, max_size=sam_max_size):
                w, h = pil_img.size
                if max(w, h) > max_size:
                    scale = max_size / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                return pil_img
            image_for_sam = resize_for_sam(original_image_pil, sam_max_size)
            try:
                image_np_for_sam = np.array(image_for_sam)
                if image_np_for_sam is None or image_np_for_sam.size == 0:
                    raise ValueError("Array vacío después de conversión de imagen para SAM")
            except Exception as e:
                is_processing = False
                return None, f"❌ Error al convertir imagen para SAM a array: {str(e)}", gr.update(value=box_mode_toggle), gr.update(value="Máscara 1"), gr.update(value="Esperando...")
            if len(image_np_for_sam.shape) == 2:
                image_np_for_sam = np.stack([image_np_for_sam] * 3, axis=-1)
            elif image_np_for_sam.shape[2] == 4:
                image_np_for_sam = image_np_for_sam[:, :, :3]
            elif image_np_for_sam.shape[2] != 3:
                is_processing = False
                return None, f"❌ Formato no soportado para SAM: {image_np_for_sam.shape}", gr.update(value=box_mode_toggle), gr.update(value="Máscara 1"), gr.update(value="Esperando...")
            if image_np_for_sam.dtype != np.uint8:
                image_np_for_sam = np.clip(image_np_for_sam, 0, 255).astype(np.uint8)
            current_image_np = image_np_for_sam.copy()
            current_image = image_for_sam
            predictor._features = None
            predictor._is_image_set = False
            predictor.set_image(current_image_np)
        if not hasattr(evt, 'index') or evt.index is None:
            is_processing = False
            return None, "❌ Error: No se detectó el clic. Intenta de nuevo.", gr.update(value=box_mode_toggle), gr.update(value="Máscara 1"), gr.update(value="Esperando...")
        if len(evt.index) < 2:
            is_processing = False
            return None, "❌ Error: Coordenadas inválidas.", gr.update(value=box_mode_toggle), gr.update(value="Máscara 1"), gr.update(value="Esperando...")
        x_raw, y_raw = evt.index[0], evt.index[1]
        x_original = int(round(float(x_raw))) if x_raw is not None else 0
        y_original = int(round(float(y_raw))) if y_raw is not None else 0
        lowres_width, lowres_height = current_image.size
        scale_x = lowres_width / original_width
        scale_y = lowres_height / original_height
        x_sam = int(x_original * scale_x)
        y_sam = int(y_original * scale_y)
        h_sam, w_sam = current_image_np.shape[:2]
        if x_sam >= w_sam or y_sam >= h_sam or x_sam < 0 or y_sam < 0:
            is_processing = False
            return None, f"❌ Coordenada fuera de límites en imagen reducida: ({x_sam}, {y_sam}). Imagen SAM: {w_sam}x{h_sam}.", gr.update(value=box_mode_toggle), gr.update(value="Máscara 1"), gr.update(value="Esperando...")
        # --- MODO BOUNDING BOX ---
        if box_mode_toggle:
            if len(box_points) == 0:
                box_points.append([x_sam, y_sam])
                # Solo actualizar vista con punto parcial
                is_processing = False
                return update_preview_with_selected_mask(box_mode_toggle, smooth_toggle=False, clean_toggle=False, min_area_slider=300, smooth_sigma=1.0)
            elif len(box_points) == 1:
                box_points.append([x_sam, y_sam])
                x1, y1 = box_points[0]
                x2, y2 = box_points[1]
                input_box = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
                masks, scores, _ = predictor.predict(
                    box=input_box,
                    multimask_output=True,
                )
                current_masks = masks
                current_scores = scores
                selected_mask_idx = int(np.argmax(scores))
                lowres_mask = masks[selected_mask_idx].astype(bool)
                # Limpiar para siguiente uso
                points.clear()
                labels.clear()
                box_points.clear()
                # Escalar máscara a resolución original
                lowres_mask_uint8 = (lowres_mask * 255).astype(np.uint8)
                lowres_mask_pil = Image.fromarray(lowres_mask_uint8)
                highres_mask_pil = lowres_mask_pil.resize((original_width, original_height), Image.NEAREST)
                highres_mask_np = np.array(highres_mask_pil) if highres_mask_pil.mode == 'L' else np.array(highres_mask_pil.convert('L'))
                final_mask = highres_mask_np > 127
                # Renderizar
                original_image_np = np.array(original_image_pil.convert('RGB'))
                overlay = original_image_np.copy()
                import colorsys
                r, g, b = colorsys.hsv_to_rgb(mask_color_hue / 360.0, 1.0, 1.0)
                mask_color = [int(r*255), int(g*255), int(b*255)]
                overlay[final_mask] = mask_color
                blended = (original_image_np * (1 - mask_alpha) + overlay * mask_alpha).astype(np.uint8)
                img_pil_for_display = Image.fromarray(blended)
                draw = ImageDraw.Draw(img_pil_for_display)
                # Dibujar rectángulo final
                x1_orig = int(min(x1, x2) / scale_x)
                y1_orig = int(min(y1, y2) / scale_y)
                x2_orig = int(max(x1, x2) / scale_x)
                y2_orig = int(max(y1, y2) / scale_y)
                draw.rectangle([x1_orig, y1_orig, x2_orig, y2_orig], outline="yellow", width=3)
                output_img = resize_output_image(img_pil_for_display)
                scores_text = f"**Scores**: 1: {scores[0]:.3f} | 2: {scores[1]:.3f} | 3: {scores[2]:.3f} → Seleccionada: {selected_mask_idx+1} ({scores[selected_mask_idx]:.3f})"
                is_processing = False
                return output_img, f"✅ Cuadro aplicado.", gr.update(value=False), gr.update(value=f"Máscara {selected_mask_idx+1}"), gr.update(value=scores_text)
        # --- MODO PUNTOS NORMAL ---
        label = 1
        if force_background_mode_checkbox:
            label = 0
        else:
            if shift_pressed:
                label = 0
        points.append([x_sam, y_sam])
        labels.append(label)
        point_coords = np.array(points) if points else None
        point_labels = np.array(labels) if labels else None
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        current_masks = masks
        current_scores = scores
        selected_mask_idx = int(np.argmax(scores))
        lowres_mask = masks[selected_mask_idx].astype(bool)
        lowres_mask_uint8 = (lowres_mask * 255).astype(np.uint8)
        lowres_mask_pil = Image.fromarray(lowres_mask_uint8)
        highres_mask_pil = lowres_mask_pil.resize((original_width, original_height), Image.NEAREST)
        if highres_mask_pil.mode != 'L':
            highres_mask_pil = highres_mask_pil.convert('L')
        highres_mask_np = np.array(highres_mask_pil)
        final_mask = highres_mask_np > 127
        original_image_np = np.array(original_image_pil.convert('RGB'))
        overlay = original_image_np.copy()
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(mask_color_hue / 360.0, 1.0, 1.0)
        mask_color = [int(r*255), int(g*255), int(b*255)]
        overlay[final_mask] = mask_color
        blended = (original_image_np * (1 - mask_alpha) + overlay * mask_alpha).astype(np.uint8)
        img_pil_for_display = Image.fromarray(blended)
        draw = ImageDraw.Draw(img_pil_for_display)
        for i, (px_sam, py_sam) in enumerate(points):
            px_original = int(px_sam / scale_x)
            py_original = int(py_sam / scale_y)
            color = (0, 255, 0) if labels[i] == 1 else (255, 0, 0)
            draw.ellipse((px_original-point_radius, py_original-point_radius, px_original+point_radius, py_original+point_radius), outline=color, width=3)
            draw.ellipse((px_original-inner_radius, py_original-inner_radius, px_original+inner_radius, py_original+inner_radius), fill="white")
        output_img = resize_output_image(img_pil_for_display)
        scores_text = f"**Scores**: 1: {scores[0]:.3f} | 2: {scores[1]:.3f} | 3: {scores[2]:.3f} → Seleccionada: {selected_mask_idx+1} ({scores[selected_mask_idx]:.3f})"
        is_processing = False
        return output_img, f"✅ Puntos: {len(points)}", gr.update(value=box_mode_toggle), gr.update(value=f"Máscara {selected_mask_idx+1}"), gr.update(value=scores_text)
    except Exception as e:
        is_processing = False
        return None, f"❌ Error interno: {str(e)}", gr.update(value=box_mode_toggle), gr.update(value="Máscara 1"), gr.update(value="Error")
# --- Actualizar vista con máscara seleccionada y post-procesamiento ---
def update_preview_with_selected_mask(box_mode_toggle=None, smooth_toggle=False, clean_toggle=False, min_area_slider=300, smooth_sigma=1.0):
    global points, labels, brush_strokes, current_image, current_image_np, mask_color_hue, mask_alpha, is_processing, point_radius, inner_radius
    global box_points, current_masks, current_scores, selected_mask_idx
    if is_processing:
        return None, "⏳ Por favor, espera...", gr.update(value=box_mode_toggle) if box_mode_toggle is not None else None, None, None
    if current_image_np is None:
        return None, "⚠️ No hay imagen cargada.", gr.update(value=box_mode_toggle) if box_mode_toggle is not None else None, None, None
    try:
        if not predictor._is_image_set or predictor._features is None:
             predictor.set_image(current_image_np)
        if current_masks is None:
            # Si no hay máscaras, forzar predicción inicial
            point_coords = np.array(points) if points else None
            point_labels = np.array(labels) if labels else None
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            current_masks = masks
            current_scores = scores
            selected_mask_idx = int(np.argmax(scores)) if len(scores) > 0 else 0
        mask = current_masks[selected_mask_idx].astype(bool)
        # Aplicar post-procesamiento si está activado
        if smooth_toggle or clean_toggle:
            mask = postprocess_mask(
                mask,
                smooth=smooth_toggle,
                clean=clean_toggle,
                min_area=min_area_slider,
                sigma=smooth_sigma
            )
        # Aplicar brush strokes
        for stroke in brush_strokes:
            brush_mask = stroke['mask']
            if stroke['label'] == 1:
                mask = np.logical_or(mask, brush_mask)
            else:
                mask = np.logical_and(mask, np.logical_not(brush_mask))
        overlay = current_image_np.copy()
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(mask_color_hue / 360.0, 1.0, 1.0)
        mask_color = [int(r*255), int(g*255), int(b*255)]
        overlay[mask] = mask_color
        blended = (current_image_np * (1 - mask_alpha) + overlay * mask_alpha).astype(np.uint8)
        img_pil = Image.fromarray(blended)
        draw = ImageDraw.Draw(img_pil)
        # Dibujar puntos
        for i, (px, py) in enumerate(points):
            color = (0, 255, 0) if labels[i] == 1 else (255, 0, 0)
            draw.ellipse((px-point_radius, py-point_radius, px+point_radius, py+point_radius), outline=color, width=3)
            draw.ellipse((px-inner_radius, py-inner_radius, px+inner_radius, py+inner_radius), fill="white")
        # Dibujar rectángulo parcial
        if len(box_points) == 1 and box_mode_toggle:
            px, py = box_points[0]
            draw.rectangle((px - 5, py - 5, px + 5, py + 5), outline="yellow", width=2)
        elif len(box_points) == 2:
            x1, y1 = box_points[0]
            x2, y2 = box_points[1]
            draw.rectangle((x1, y1, x2, y2), outline="yellow", width=3)
        output_img = resize_output_image(img_pil)
        if current_scores is not None and len(current_scores) >= 3:
            scores_text = f"**Scores**: 1: {current_scores[0]:.3f} | 2: {current_scores[1]:.3f} | 3: {current_scores[2]:.3f} → Seleccionada: {selected_mask_idx+1} ({current_scores[selected_mask_idx]:.3f})"
        else:
            scores_text = "Esperando segmentación..."
        return output_img, "🎨 Vista actualizada.", gr.update(value=box_mode_toggle) if box_mode_toggle is not None else None, gr.update(value=f"Máscara {selected_mask_idx+1}"), gr.update(value=scores_text)
    except Exception as e:
        return None, f"❌ Error al actualizar vista: {str(e)}", gr.update(value=box_mode_toggle) if box_mode_toggle is not None else None, None, None
# --- Seleccionar máscara manualmente ---
def select_mask(mask_choice_str):
    global selected_mask_idx, current_masks, current_scores
    idx_map = {"Máscara 1": 0, "Máscara 2": 1, "Máscara 3": 2}
    if mask_choice_str in idx_map:
        selected_mask_idx = idx_map[mask_choice_str]
    return update_preview_with_selected_mask()
# --- Eliminar último punto ---
def remove_last_point():
    global points, labels, is_processing
    if is_processing:
        return None, "⏳ Por favor, espera...", None, None, None
    if points:
        removed = points.pop()
        labels.pop()
        return update_preview_with_selected_mask()
    else:
        return None, "⚠️ No hay puntos para eliminar.", None, None, None
# --- Actualizar color con Hue ---
def update_mask_color(hue, alpha):
    global mask_color_hue, mask_alpha
    mask_color_hue = hue
    mask_alpha = alpha
    if current_image_np is not None:
        return update_preview_with_selected_mask()
    return None, "Esperando imagen...", None, None, None
# --- Control unificado de tamaño de puntos ---
def update_point_size(radius):
    global point_radius, inner_radius
    point_radius = int(max(0, min(100, radius)))
    inner_radius = max(0, point_radius // 2)
    if current_image_np is not None:
        return update_preview_with_selected_mask()
    return None, "Esperando imagen...", None, None, None
# --- Invertir máscara ---
def invert_mask():
    global current_masks, selected_mask_idx, current_image_np, mask_color_hue, mask_alpha
    if current_masks is None:
        return None, "❌ No hay máscara para invertir.", None, None, None
    mask = current_masks[selected_mask_idx].astype(bool)
    inverted_mask = ~mask  # Inversión booleana
    # Guardar como nueva máscara temporal
    current_masks = np.array([inverted_mask.astype(np.float32)])
    current_scores = np.array([0.9])  # Valor dummy
    selected_mask_idx = 0
    return update_preview_with_selected_mask()
# --- Guardar objeto (múltiples formatos) ---
def save_current_object(object_name, format_choice, smooth_toggle, clean_toggle, min_area_slider, smooth_sigma):
    global current_image_np, points, labels, brush_strokes, mask_color_hue, mask_alpha, is_processing
    global original_image_pil, original_width, original_height, current_image, point_radius, inner_radius
    global current_masks, selected_mask_idx
    if is_processing:
        return None, "⏳ Por favor, espera...", None
    if current_image_np is None:
        return current_image, "❌ No hay imagen para guardar.", None
    try:
        if not predictor._is_image_set or predictor._features is None:
             predictor.set_image(current_image_np)
        if current_masks is None:
            # Forzar predicción si no hay máscaras
            point_coords = np.array(points) if points else None
            point_labels = np.array(labels) if labels else None
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            current_masks = masks
            selected_mask_idx = int(np.argmax(scores))
        mask = current_masks[selected_mask_idx].astype(bool)
        # Aplicar post-procesamiento al guardar
        mask = postprocess_mask(
            mask,
            smooth=smooth_toggle,
            clean=clean_toggle,
            min_area=min_area_slider,
            sigma=smooth_sigma
        )
        lowres_mask_uint8 = (mask * 255).astype(np.uint8)
        lowres_mask_pil = Image.fromarray(lowres_mask_uint8)
        if original_image_pil is not None:
            original_width_to_use, original_height_to_use = original_image_pil.size
        elif current_image is not None:
             original_width_to_use, original_height_to_use = current_image.size
        else:
             original_height_to_use, original_width_to_use = current_image_np.shape[:2]
        highres_mask_pil = lowres_mask_pil.resize((original_width_to_use, original_height_to_use), Image.NEAREST)
        highres_mask_np = np.array(highres_mask_pil) if highres_mask_pil.mode == 'L' else np.array(highres_mask_pil.convert('L'))
        final_mask_for_saving = highres_mask_np > 127
        timestamp = datetime.now().strftime("%H%M%S")
        safe_name = "".join(c if c.isalnum() else "_" for c in object_name) if object_name.strip() else "mask"
        final_name = f"{safe_name}_{timestamp}"
        # --- OPCIÓN 1: MÁSCARA SÓLIDA (PNG) ---
        if "Máscara sólida" in format_choice:
            mask_path = os.path.join(MASKS_DIR, f"{final_name}_solid.png")
            mask_rgba = np.dstack((
                final_mask_for_saving * 255,
                final_mask_for_saving * 255,
                final_mask_for_saving * 255,
                final_mask_for_saving * 255
            )).astype(np.uint8)
            mask_pil = Image.fromarray(mask_rgba)
            mask_pil.save(mask_path, format='PNG', compress_level=6)
            message = f"✅ Máscara sólida guardada: {os.path.basename(mask_path)}"
        # --- OPCIÓN 2: IMAGEN RECORTADA (PNG con fondo transparente) ---
        elif "Imagen recortada" in format_choice:
            if original_image_pil is None:
                return current_image, "❌ No se puede recortar: imagen original no disponible.", None
            original_rgba = original_image_pil.convert("RGBA")
            original_np = np.array(original_rgba)
            alpha_channel = final_mask_for_saving.astype(np.uint8) * 255
            original_np[:, :, 3] = alpha_channel
            cutout_path = os.path.join(MASKS_DIR, f"{final_name}_cutout.png")
            cutout_pil = Image.fromarray(original_np, mode='RGBA')
            cutout_pil.save(cutout_path, format='PNG', compress_level=6)
            message = f"✅ Imagen recortada guardada: {os.path.basename(cutout_path)}"
        # --- OPCIÓN 3: CONTORNOS (JSON) ---
        elif "Contornos (JSON)" in format_choice:
            if original_image_pil is None:
                return current_image, "❌ Imagen original no disponible.", None
            contours, _ = cv2.findContours(final_mask_for_saving.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_serializable = []
            for contour in contours:
                if len(contour) >= 3:  # Filtrar contornos válidos
                    contours_serializable.append(contour.squeeze().tolist())
            json_path = os.path.join(MASKS_DIR, f"{final_name}_contours.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "object_name": object_name,
                    "contours": contours_serializable,
                    "image_size": [original_width_to_use, original_height_to_use],
                    "timestamp": timestamp
                }, f, indent=2, ensure_ascii=False)
            message = f"✅ Contornos guardados: {os.path.basename(json_path)}"
        else:
            return current_image, "⚠️ Formato no implementado aún.", None
        # Limpiar puntos
        points.clear()
        labels.clear()
        brush_strokes.clear()
        box_points.clear()
        current_masks = None
        current_scores = None
        selected_mask_idx = 0
        return current_image, message
    except Exception as e:
        return current_image, f"❌ Error al guardar: {str(e)}"
# --- Reiniciar todo ---
def reset_all():
    global points, labels, brush_strokes, is_processing, image_loaded
    global original_image_pil, original_width, original_height
    global box_points, current_masks, current_scores, selected_mask_idx
    if is_processing:
        return None, "⏳ Por favor, espera...", None, None, None
    # Limpiar TODO lo relacionado con segmentación
    points.clear()
    labels.clear()
    brush_strokes.clear()
    box_points.clear()
    current_masks = None
    current_scores = None
    selected_mask_idx = 0
    # Forzar actualización visual: si hay imagen, mostrarla SIN máscara
    if original_image_pil is not None:
        # Mostrar la imagen original sin superposición
        output_img = resize_output_image(original_image_pil)
        return output_img, "🔄 Máscara y puntos reiniciados.", gr.update(value=False), gr.update(value="Máscara 1"), gr.update(value="Esperando segmentación...")
    elif current_image is not None:
        output_img = resize_output_image(current_image)
        return output_img, "🔄 Máscara y puntos reiniciados.", gr.update(value=False), gr.update(value="Máscara 1"), gr.update(value="Esperando segmentación...")
    else:
        return None, "🔄 Todo reiniciado.", gr.update(value=False), gr.update(value="Máscara 1"), gr.update(value="Esperando segmentación...")
# --- Abrir carpeta ---
def open_masks_folder():
    try:
        if platform.system() == "Windows":
            os.startfile(MASKS_DIR)
        elif platform.system() == "Darwin":
            subprocess.call(["open", MASKS_DIR])
        else:
            subprocess.call(["xdg-open", MASKS_DIR])
        return "✅ Carpeta de máscaras abierta."
    except Exception as e:
        return f"❌ Error al abrir carpeta: {str(e)}"
# --- CSS PERSONALIZADO ---
css = """
.image-container.svelte-12ioyct img {
    user-drag: none;
    user-select: none;
    -moz-user-select: none;
    -webkit-user-drag: none;
    -webkit-user-select: none;
    -ms-user-select: none;
}
.output-image .svelte-12ioyct .placeholder {
    display: none !important;
}
.output-image .svelte-12ioyct .upload-container {
    display: none !important;
}
#input_image .source-selection.svelte-snayfm {
    display: none !important;
}
#output_image .image-actions.svelte-v1x63mz {
    display: none !important;
}

/* --- ESTILOS RESPONSIVOS PARA EL ENCABEZADO --- */
.header-container {
    display: flex;
    align-items: center;
    gap: 1em;
    padding: 1em 0;
}

.logo-svg {
    flex-shrink: 0;
}

.text-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.main-title {
    font-size: 1.8em;
    font-weight: 600;
    line-height: 1.2;
    color: #333;
    margin: 0;
}

.subtitle {
    font-size: 1.1em;
    color: #666;
    font-weight: 400;
    line-height: 1.4;
    margin: 0;
}

/* Media Queries: Ajustar tamaños en pantallas pequeñas */
@media (max-width: 1200px) {
    .main-title { font-size: 1.6em; }
    .subtitle { font-size: 1.05em; }
}

@media (max-width: 992px) {
    .main-title { font-size: 1.4em; }
    .subtitle { font-size: 1em; }
}

@media (max-width: 768px) {
    .main-title { font-size: 1.3em; }
    .subtitle { font-size: 0.95em; }
}

@media (max-width: 576px) {
    .main-title { font-size: 1.2em; }
    .subtitle { font-size: 0.9em; }
    .header-container { gap: 0.75em; }
}

@media (max-width: 480px) {
    .main-title { font-size: 1.1em; }
    .subtitle { font-size: 0.85em; }
}

/* --- FULLSCREEN FIX --- */
.gradio-container {
    max-width: 100% !important;
    padding: 0 2vw;
}
#input_image, #output_image {
    width: 100% !important;
}
footer {
    display: none !important;
}
#_ui_row_ {
    display: none !important;
}
.gradio-container > div:last-child {
    display: none !important;
}
.custom-footer {
    text-align: center;
    padding: 20px 0;
    margin-top: 20px;
    font-size: 1em;
    color: #666;
    border-top: 1px solid #e0e0e0;
    width: 100%;
    box-sizing: border-box;
}
.custom-footer a {
    color: #4a90e2;
    text-decoration: none;
}
.custom-footer a:hover {
    text-decoration: underline;
}

"""

# --- TEMA PERSONALIZADO ---
theme = gr.themes.Monochrome(
    primary_hue="stone",
    neutral_hue="stone",
    radius_size="lg",
    font=[
        gr.themes.GoogleFont('DM Sans'),
        gr.themes.GoogleFont('Rubik'),
        gr.themes.GoogleFont('Nunito'),
        gr.themes.GoogleFont('Exo 2')
    ],
    font_mono=[
        gr.themes.GoogleFont('IBM Plex Mono'),
        'ui-monospace',
        'Consolas',
        'monospace'
    ],
).set(
    embed_radius='*radius_xl',
    button_border_width='3px',
    button_border_width_dark='3px',
    button_transition='all 0.5s ease',
    button_large_padding='*spacing_md',
    button_large_text_size='*text_md',
    button_small_radius='*radius_lg'
)

# --- Interfaz ---
with gr.Blocks(theme=theme, css=css, title="CropOn!") as demo:
    try:
        with open("logo_cropon.svg", "r", encoding="utf-8") as svg_file:
            svg_content = svg_file.read()
            svg_content = svg_content.replace('<?xml version="1.0" encoding="UTF-8"?>', '')
            svg_content = svg_content.replace('<defs>', '').replace('</defs>', '')
            svg_content = svg_content.replace('<style>', '').replace('</style>', '')
            svg_content = svg_content.replace('<rect class="st1" width="48" height="48"/>', '')
            svg_content = svg_content.replace('class="st0"', 'fill="#3d3d3d"')
    except:
        svg_content = ""
    header_html = f"""
    <div class="header-container">
        <svg xmlns="http://www.w3.org/2000/svg" width="70" height="70" viewBox="0 0 40 40" class="logo-svg">
            {svg_content}
        </svg>
        <div class="text-container">
            <div class="main-title">
                CropOn! - (SAM2.1 Model)
            </div>
            <div class="subtitle">
                Segmenta, ajusta y guarda en el formato que quieras
            </div>
        </div>
    </div>
    """
    gr.HTML(header_html, elem_id="title_container")

    
    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(
                label="Entrada",
                type="pil",
                interactive=True,
                height=500,
                show_label=True,
                placeholder="Arrastre la imagen aquí - o - Haga clic para cargar",
                elem_id="input_image"
            )
            output_image = gr.Image(
                label="Salida",
                interactive=False,
                height=500,
                show_label=True,
                show_download_button=False,
                show_share_button=False,
                type="pil",
                container=True,
                elem_classes="output-image",
                elem_id="output_image"
            )
        with gr.Column(scale=1):
            # 1 - Estado de la imagen
            status_text = gr.Textbox(
                label="Estado",
                value="Esperando imagen...",
                interactive=False,
                lines=2
            )
            # 2 - Instrucciones de uso
            with gr.Accordion("Instrucciones de uso", open=False):
                gr.Markdown("""

                ### ⚠️ **ADVERTENCIA:**
                Si estás trabajando con imágenes de **muy alta resolución**, es posible que experimentes un error interno del tipo **"ASGI application"** o que la imagen no se cargue correctamente (carga parcial, sin visualización, etc.).

                **Esto NO es un error del modelo SAM2 ni de CropOn!**, sino un límite técnico en la comunicación entre el servidor y tu navegador. Para solucionarlo:

                1. Haz clic en el botón **"Limpiar Caché"**.
                2. Espera a que el sistema se reinicie.
                3. Vuelve a arrastrar o seleccionar tu imagen.

                Este error suele aparecer detallado en la consola de comandos (Git Bash, Terminal, CMD), pero en la interfaz puede manifestarse de forma silenciosa. ¡El botón "Limpiar Caché" es tu mejor aliado en estos casos!

                ---

                ### **Cómo interactuar con la imagen:**
                - **Clic izquierdo**: Agrega un punto **foreground** (verde) para indicar qué quieres **INCLUIR** en la máscara.
                - **Shift + Clic izquierdo**: Agrega un punto **background** (rojo) para indicar qué quieres **EXCLUIR** de la máscara.
                - **Modo Rectángulo (2 Clicks)**: Activa este checkbox y haz **dos clics** en la imagen para dibujar un cuadro delimitador alrededor del objeto que deseas segmentar.

                ### **Controles Básicos:**
                - **Tono (Hue)**: Cambia el color de la superposición de la máscara (por ejemplo, de verde a azul o rojo).
                - **Opacidad (Alpha)**: Ajusta cuán transparente o sólida se ve la máscara sobre la imagen original.
                - **Tamaño de Puntos**: Controla el radio visual de los puntos verdes y rojos que marcas en la imagen.
                - **Forzar modo background**: Cuando está activado, **TODOS** tus clics se convierten en puntos rojos (background), sin importar si presionas Shift o no.

                ### **Gestión de Máscaras:**
                - **Seleccionar máscara (Máscara 1, 2 o 3)**: SAM2 genera 3 máscaras posibles. Elige la que mejor se ajuste a tu objeto. El "score" indica la confianza del modelo (más alto = mejor).
                - **Eliminar Último Punto**: Borra el último punto (verde o rojo) que agregaste, por si te equivocaste.
                - **Reiniciar Puntos**: Limpia **todos** los puntos y la máscara actual, pero mantiene la imagen cargada. Ideal para empezar de cero con la misma imagen.
                - **Invertir Máscara**: Convierte lo que estaba seleccionado (enmascarado) en no seleccionado, y viceversa. Útil para segmentar el fondo en lugar del objeto.

                ### **Controles Avanzados (Post-Procesamiento):**
                - **Suavizar bordes**: Aplica un filtro para que los bordes de la máscara sean menos pixelados y más suaves.
                - **Eliminar ruido pequeño**: Elimina pequeñas áreas aisladas dentro o fuera de la máscara que probablemente sean errores.
                - **Área mínima (px)**: Define el tamaño mínimo (en píxeles) que debe tener una región para no ser considerada "ruido" y eliminada.
                - **Intensidad suavizado**: Controla cuán fuerte es el efecto de suavizado aplicado a los bordes.

                ### **Exportación y Guardado:**
                - **Formato de exportación**:
                    - *Máscara sólida (PNG)*: Guarda solo la máscara en blanco y negro (blanco = objeto, negro = fondo).
                    - *Imagen recortada (PNG)*: Guarda la imagen original recortada con fondo transparente, listo para usar en diseños.
                    - *Contornos (JSON)*: Guarda las coordenadas del borde del objeto en un archivo de texto para usar en otros programas.
                - **Nombre de la máscara**: Asigna un nombre personalizado a tu archivo (ej: "logo_empresa", "persona_01"). Si lo dejas vacío, se usará "mask".
                - **Guardar Máscara**: Ejecuta la acción de guardar en el formato y nombre especificados.
                - **Abrir Carpeta**: Abre la carpeta `masks` en tu explorador de archivos para que veas los archivos que has guardado.
                - **Limpiar Caché**: Reinicia completamente la aplicación. Útil si la imagen no carga correctamente o si hay errores persistentes. **¡Recuerda volver a arrastrar tu imagen después de usarlo!**

                """)
            # 3 - Menú de Control
            gr.Markdown("### Menú de Control")
            # Controles Básicos
            with gr.Accordion("Controles Básicos", open=False):
                hue_slider = gr.Slider(0, 360, 125, step=1, label="Tono (Hue)")  
                alpha_slider = gr.Slider(0.00, 1.00, 0.50, step=0.01, label="Opacidad (Alpha)")
                unified_radius_slider = gr.Slider(0, 100, 20, step=1, label="Tamaño de Puntos") 
                background_toggle = gr.Checkbox(label="Forzar modo background", value=False)
                box_mode_toggle = gr.Checkbox(label="Modo Rectángulo (2 Clicks)", value=False) 
            # Controles Avanzados (Post-Procesamiento)
            with gr.Accordion("Controles Avanzados (Post-Procesamiento)", open=False):
                smooth_toggle = gr.Checkbox(label="Suavizar bordes", value=False)
                clean_toggle = gr.Checkbox(label="Eliminar ruido pequeño", value=False)
                min_area_slider = gr.Slider(0, 5000, 300, step=50, label="Área mínima (px)")
                smooth_sigma = gr.Slider(0, 10, 1, step=0.5, label="Intensidad suavizado")
            # 4 - Botón de limpiar caché.
            reload_image_btn = gr.Button("Limpiar Caché", variant="secondary")
            # 5 - Botones de gestión de puntos
            with gr.Row():
                remove_point_btn = gr.Button("Eliminar Último Punto", variant="secondary", scale=1)
                reset_all_btn = gr.Button("Reiniciar Puntos", variant="secondary", scale=1)
                invert_mask_btn = gr.Button("Invertir Máscara", variant="secondary", scale=1)
            # 6 - Seleccionar máscara
            gr.Markdown("### Seleccionar máscara")
            mask_choice = gr.Radio(
                choices=["Máscara 1", "Máscara 2", "Máscara 3"],
                value="Máscara 1",
                label="Elegir máscara",
                interactive=True
            )
            mask_scores_display = gr.Markdown("Esperando segmentación...")
            # 7 - Formato de exportación
            format_choice = gr.Dropdown(
                choices=["Máscara sólida (PNG)", "Imagen recortada (PNG)", "Contornos (JSON)"],
                value="Máscara sólida (PNG)",
                label="Formato de exportación",
                interactive=True
            )
            # 8 - Textbox de nombre y botones de acción
            object_name_input = gr.Textbox(
                label="Nombre de la máscara",
                placeholder="Ej: mask_01...",
                elem_classes="name-input"
            )
            with gr.Row():
                save_btn = gr.Button("Guardar Máscara", variant="primary", scale=1)
                open_folder_btn = gr.Button("Abrir Carpeta", variant="secondary", scale=1)

    gr.HTML("""
    <div class="custom-footer">
        CropOn! - Designed by Felipe Dorado | Powered by <a href="https://github.com/facebookresearch/segment-anything-2" target="_blank">SAM 2.1</a> & <a href="https://www.gradio.app/" target="_blank">Gradio</a>
    </div>
    """)
    # Eventos
    input_image.select(
        fn=segment_image,
        inputs=[input_image, background_toggle, box_mode_toggle],
        outputs=[output_image, status_text, box_mode_toggle, mask_choice, mask_scores_display]
    )
    mask_choice.change(
        fn=select_mask,
        inputs=mask_choice,
        outputs=[output_image, status_text, box_mode_toggle, mask_choice, mask_scores_display]
    )
    remove_point_btn.click(
        fn=remove_last_point,
        outputs=[output_image, status_text, box_mode_toggle, mask_choice, mask_scores_display]
    )
    reset_all_btn.click(
        fn=reset_all,
        outputs=[output_image, status_text, box_mode_toggle, mask_choice, mask_scores_display]
    )
    invert_mask_btn.click(
        fn=invert_mask,
        outputs=[output_image, status_text, box_mode_toggle, mask_choice, mask_scores_display]
    )
    save_btn.click(
        fn=save_current_object,
        inputs=[object_name_input, format_choice, smooth_toggle, clean_toggle, min_area_slider, smooth_sigma],
        outputs=[output_image, status_text]
    )
    open_folder_btn.click(fn=open_masks_folder, outputs=status_text)
    # Actualizar vista al cambiar post-procesamiento
    smooth_toggle.change(
        fn=lambda s, c, a, sig: update_preview_with_selected_mask(smooth_toggle=s, clean_toggle=c, min_area_slider=a, smooth_sigma=sig),
        inputs=[smooth_toggle, clean_toggle, min_area_slider, smooth_sigma],
        outputs=[output_image, status_text, box_mode_toggle, mask_choice, mask_scores_display]
    )
    clean_toggle.change(
        fn=lambda s, c, a, sig: update_preview_with_selected_mask(smooth_toggle=s, clean_toggle=c, min_area_slider=a, smooth_sigma=sig),
        inputs=[smooth_toggle, clean_toggle, min_area_slider, smooth_sigma],
        outputs=[output_image, status_text, box_mode_toggle, mask_choice, mask_scores_display]
    )
    min_area_slider.release(
        fn=lambda s, c, a, sig: update_preview_with_selected_mask(smooth_toggle=s, clean_toggle=c, min_area_slider=a, smooth_sigma=sig),
        inputs=[smooth_toggle, clean_toggle, min_area_slider, smooth_sigma],
        outputs=[output_image, status_text, box_mode_toggle, mask_choice, mask_scores_display]
    )
    smooth_sigma.release(
        fn=lambda s, c, a, sig: update_preview_with_selected_mask(smooth_toggle=s, clean_toggle=c, min_area_slider=a, smooth_sigma=sig),
        inputs=[smooth_toggle, clean_toggle, min_area_slider, smooth_sigma],
        outputs=[output_image, status_text, box_mode_toggle, mask_choice, mask_scores_display]
    )
    unified_radius_slider.release(fn=update_point_size, inputs=unified_radius_slider, outputs=[output_image, status_text, box_mode_toggle, mask_choice, mask_scores_display])
    unified_radius_slider.change(fn=update_point_size, inputs=unified_radius_slider, outputs=[output_image, status_text, box_mode_toggle, mask_choice, mask_scores_display])
    hue_slider.change(fn=update_mask_color, inputs=[hue_slider, alpha_slider], outputs=[output_image, status_text, box_mode_toggle, mask_choice, mask_scores_display])
    alpha_slider.change(fn=update_mask_color, inputs=[hue_slider, alpha_slider], outputs=[output_image, status_text, box_mode_toggle, mask_choice, mask_scores_display])
    input_image.change(fn=handle_image_load, inputs=input_image, outputs=[output_image, status_text, box_mode_toggle, mask_choice, mask_scores_display])
    reload_image_btn.click(fn=reload_image, outputs=[input_image, status_text])
# ✅ ¡LANZAMIENTO ESTABLE!
if __name__ == "__main__":
    import os
    favicon_path = "./favicon_base.ico"
    if not os.path.exists(favicon_path):
        print(f"❌ Error: No se encontró {favicon_path}.")
        exit(1)
    file_size = os.path.getsize(favicon_path)
    print(f"📁 Favicon cargado: {favicon_path} ({file_size} bytes)")
    if file_size < 1000:
        print("⚠️ ¡Alerta! El favicon parece demasiado pequeño.")
    else:
        print("🎉 ¡Favicon listo para usar!")
    demo.queue().launch(
        inbrowser=True,
        favicon_path=favicon_path
    )