import cv2
import numpy as np
import os

def overlay_glare_on_sign(base_img_path, glare_img_path, bbox,
                          glare_scale=2.0, brightness=1.3, offset=(0, -0.3)):
    # Load base and glare image
    base = cv2.imread(base_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    glare = cv2.imread(glare_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    h, w = base.shape[:2]

    # Unpack bbox
    x_c, y_c, bw, bh = bbox
    sign_w, sign_h = int(bw * w), int(bh * h)
    sign_x, sign_y = int(x_c * w), int(y_c * h)

    # Glare center (slightly offset)
    glare_center = (
        int(sign_x + offset[0] * sign_w),
        int(sign_y + offset[1] * sign_h)
    )

    # Resize glare based on sign size
    glare_h, glare_w = glare.shape[:2]
    new_w, new_h = int(sign_w * glare_scale), int(sign_h * glare_scale)
    glare = cv2.resize(glare, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Compute overlay region
    x0 = max(0, glare_center[0] - new_w // 2)
    y0 = max(0, glare_center[1] - new_h // 2)
    x1 = min(w, x0 + new_w)
    y1 = min(h, y0 + new_h)
    glare = glare[0:(y1 - y0), 0:(x1 - x0)]

    # Split channels
    if glare.shape[2] == 4:
        glare_rgb = glare[..., :3] * brightness
        alpha = glare[..., 3:4]
    else:
        glare_rgb = glare * brightness
        alpha = np.ones_like(glare_rgb[..., :1])

    # Blend additively
    roi = base[y0:y1, x0:x1]
    blended = np.clip(roi + glare_rgb * alpha, 0, 1)
    base[y0:y1, x0:x1] = blended

    return (base * 255).astype(np.uint8)


# -------------------------------
# Apply glare to multiple images
# -------------------------------
glare_png = "./1_Datasets/sun_glare.png"

# Example list of images with bbox (normalized)
image_info = ""

output_dir = "./1_Datasets/car/train/glare_results"
os.makedirs(output_dir, exist_ok=True)

for info in image_info:
    img_path = info["path"]
    bbox = info["bbox"]
    filename = os.path.basename(img_path)
    output_path = os.path.join(output_dir, f"{filename}")

    result = overlay_glare_on_sign(
        img_path, glare_png, bbox,
        glare_scale=2.2, brightness=1.6, offset=(0.3, -0.5)
    )

    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}")

print("Glare applied to all images successfully!")
