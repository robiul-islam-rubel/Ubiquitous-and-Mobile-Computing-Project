from ultralytics import YOLO
import cv2

# Load your custom YOLO model trained on traffic signs
model = YOLO("./model/traffic_signs.pth")   # replace with your model

# Load your image
image_path = "./1_Datasets/speedlimit.jpg"
results = model(image_path)

# Read image once (not inside the loop)
img = cv2.imread(image_path)

# Loop over detections
for result in results:
    for box in result.boxes:
        # Bounding box in xyxy format
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        # Only keep speed limit detections
        if "speed limit" in label.lower():
            print(f"Detected {label} with {conf:.2f} confidence")
            print(f"BBox: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

            # Draw box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
            cv2.putText(img, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# Save result
cv2.imwrite("speedlimit_detected.jpg", img)
print("Result saved to speedlimit_detected.jpg")
