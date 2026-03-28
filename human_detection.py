image_path = take_photo()
print("Image captured:", image_path)

from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

img = cv2.imread(image_path)

results = model(img)

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        
        if cls == 0:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

cv2.imwrite("output.jpg", img)

print("✅ Detection Done")

from IPython.display import Image
Image("output.jpg")
