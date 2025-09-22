import cv2
import numpy as np
import torch
from PIL import Image
import os

# --- Cargar modelo ---
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = "F:/IA/SAM2/checkpoints/sam2.1_hiera_large.pt"
CONFIG_PATH = "F:/IA/SAM2/segment-anything-2/sam2/configs/sam2.1_hiera_l.yaml"

print("Cargando modelo SAM 2.1...")
sam2_model = build_sam2(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)
print("✅ ¡Modelo cargado!")

# --- Pedir imagen ---
print("\n➡️  Arrastra tu imagen aquí o escribe la ruta completa:")
image_path = input().strip().strip('"')

if not os.path.exists(image_path):
    print("❌ No se encontró la imagen. Verifica la ruta.")
    exit()

image = cv2.imread(image_path)
if image is None:
    print("❌ No se pudo cargar la imagen. ¿Es un formato válido (JPG, PNG)?")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
points = []
labels = []

def click_event(event, x, y, flags, param):
    global points, labels, image_rgb
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        labels.append(1)  # 1 = foreground
        
        predictor.set_image(image_rgb)
        input_points = np.array(points)
        input_labels = np.array(labels)
        
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,
        )
        
        mask = masks[0]
        overlay = image_rgb.copy()
        overlay[mask] = [255, 100, 100]  # Rojo suave
        blended = (image_rgb * 0.7 + overlay * 0.3).astype(np.uint8)
        
        display = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        cv2.imshow("SAM 2.1 - OpenCV UI (Presiona S para guardar, Q para salir)", display)

# --- Mostrar ventana ---
cv2.namedWindow("SAM 2.1 - OpenCV UI (Presiona S para guardar, Q para salir)")
cv2.setMouseCallback("SAM 2.1 - OpenCV UI (Presiona S para guardar, Q para salir)", click_event)
cv2.imshow("SAM 2.1 - OpenCV UI (Presiona S para guardar, Q para salir)", image)

print("\n✅ Listo para usar:")
print("🖱️  Haz clic en la imagen para segmentar.")
print("💾 Presiona 'S' para guardar la máscara como 'mask_output.png'.")
print("🚪 Presiona 'Q' para salir.")

# --- Loop principal ---
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if 'mask' in locals():
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil.save("mask_output.png")
            print("✅ Máscara guardada como 'mask_output.png'")
        else:
            print("⚠️  Primero haz clic en la imagen para generar una máscara.")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()