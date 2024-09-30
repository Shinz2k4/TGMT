import pytesseract
import cv2
import numpy as np

# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString

# Đường dẫn ảnh
img_path = "test.jpg"
Ivehicle = cv2.imread(img_path)

# Load YOLO model
net = cv2.dnn.readNet("yolo-plate.weights", "yolo-plate.cfg")

# Load class names for YOLO (if needed)
classes = ["license_plate"]

# Prepare the image for YOLO
height, width = Ivehicle.shape[:2]
blob = cv2.dnn.blobFromImage(Ivehicle, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Run forward pass to get the detections
detections = net.forward(output_layers)

# Initialize variables for storing license plate region
license_plate = None
conf_threshold = 0.5
nms_threshold = 0.4

# Loop over each detection
for output in detections:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Extract the license plate region
            license_plate = Ivehicle[y:y+h, x:x+w]

            # Draw the rectangle around the detected license plate
            cv2.rectangle(Ivehicle, (x, y), (x + w, y + h), (0, 255, 0), 2)

# If a license plate was detected, process it further
if license_plate is not None:
    # Chuyen doi anh bien so
    license_plate = cv2.convertScaleAbs(license_plate, alpha=(255.0))
    gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Nhan dien bien so bang pytesseract
    text = pytesseract.image_to_string(binary, lang="eng", config="--psm 7")

    # Fine-tune the result
    text = fine_tune(text)

    # Viet bien so len anh
    cv2.putText(Ivehicle, text, (50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)

cv2.imwrite("output.png", Ivehicle)
