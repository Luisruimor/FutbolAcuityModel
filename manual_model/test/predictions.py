import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import os

# --- Configuración ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (1024, 1024)  # Tamaño al que se redimensionarán las imágenes
NUM_CLASSES = 3  # Número de clases (Referee, Ball, Player)
MODEL_PATH = "../soccer_object_detector.pth"  # Ruta a tu modelo guardado
TEST_IMAGE_PATH = "../../data/frames/ray.v.ath.full.laliga.01.12.2024/frame_13605.jpg"  # Ruta de la imagen de prueba
CONFIDENCE_THRESHOLD = 0.75  # Umbral de confianza para filtrar las predicciones

# Mapeo de IDs a nombres de clases
CLASS_NAMES = {
    0: "Ball",
    1: "Referee",
    2: "Player"
}
# --- Fin de la configuración ---

# Clase del modelo (debe ser la misma que usaste para entrenar)
class SoccerObjectDetector(nn.Module):
    def __init__(self, num_classes):
        super(SoccerObjectDetector, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * (IMAGE_SIZE[0] // 8) * (IMAGE_SIZE[1] // 8), 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc_bbox = nn.Linear(512, 24 * 4)
        self.fc_class = nn.Linear(512, 24 * num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu4(self.fc1(x)))
        bbox_output = self.fc_bbox(x)
        class_output = self.fc_class(x)
        return bbox_output, class_output

# Función de inferencia
def inference(model, image_path, image_size, device, confidence_threshold=0.5):
    model.eval()

    try:
        original_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: No se encontró la imagen en la ruta: {image_path}")
        return [], [], [], None
    except Exception as e:
        print(f"Error al abrir la imagen: {e}")
        return [], [], [], None

    image = original_image.resize(image_size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        bbox_output, class_output = model(image)

    bbox_output = bbox_output.view(-1, 24, 4)
    class_output = class_output.view(-1, 24, NUM_CLASSES)
    probabilities = torch.softmax(class_output, dim=2)
    scores, predicted_classes = torch.max(probabilities, dim=2)

    mask = scores > confidence_threshold
    filtered_boxes = bbox_output[mask]
    filtered_classes = predicted_classes[mask]
    filtered_scores = scores[mask]

    # Escalar las coordenadas de las bounding boxes al tamaño original de la imagen
    width_scale = original_image.width / image_size[0]
    height_scale = original_image.height / image_size[1]

    scaled_boxes = []
    for box in filtered_boxes:
        x_min, y_min, x_max, y_max = box.tolist()
        scaled_boxes.append([
            x_min * width_scale,
            y_min * height_scale,
            x_max * width_scale,
            y_max * height_scale
        ])

    return scaled_boxes, filtered_classes, filtered_scores, original_image

# Cargar el modelo
model = SoccerObjectDetector(NUM_CLASSES)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except FileNotFoundError:
    print(f"Error: No se encontró el modelo en la ruta: {MODEL_PATH}")
    exit()
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

model.to(DEVICE)

# Realizar la inferencia
boxes, classes, scores, original_image = inference(model, TEST_IMAGE_PATH, IMAGE_SIZE, DEVICE, CONFIDENCE_THRESHOLD)

# Dibujar las predicciones sobre la imagen
if original_image:
    draw = ImageDraw.Draw(original_image)
    for box, class_id, score in zip(boxes, classes, scores):
        # Asegurarse de que las coordenadas sean válidas
        x0, y0, x1, y1 = box
        if x0 < x1 and y0 < y1:
            draw.rectangle(box, outline="red", width=3)
            draw.text((x0, y0), f"{CLASS_NAMES[class_id.item()]}: {score.item():.2f}", fill="red")
        else:
            print(f"Advertencia: Se detectó una bounding box inválida: {box}. No se dibujará.")

    # Mostrar la imagen
    original_image.show()

    # Opcional: Guardar la imagen con las predicciones
    original_image.save("img_predict2.jpg")