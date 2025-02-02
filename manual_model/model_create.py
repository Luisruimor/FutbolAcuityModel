import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os

# --- Configuración ---
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (1024, 1024)  # Tamaño de las imágenes de entrada
NUM_CLASSES = 3  # Número de clases (Referee, Ball, Player)
DATA_DIR = "../data/Acuity11m.v2i.yolov8" # Directorio principal de los datos
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR = os.path.join(DATA_DIR, "test")


class SoccerDataset(Dataset):
    def __init__(self, root_dir, image_size, transform=None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.image_files = sorted(os.listdir(self.image_dir))
        self.label_files = sorted(os.listdir(self.label_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.image_size)

        boxes = []
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())

                # Convertir de YOLO a formato (x_min, y_min, x_max, y_max)
                x_min = (x_center - width / 2) * self.image_size[0]
                y_min = (y_center - height / 2) * self.image_size[1]
                x_max = (x_center + width / 2) * self.image_size[0]
                y_max = (y_center + height / 2) * self.image_size[1]

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(class_id)) # Ya están en el rango 0, 1, 2

        # Convertir a tensores
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Aplicar transformaciones si se especifican
        if self.transform:
            image = self.transform(image)

        # Crear el target para el modelo
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return image, target

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

        # Capa de salida para las coordenadas de las bounding boxes (4 valores por caja)
        # Suponiendo un máximo de 24 objetos (1 árbitro, 1 balón, 22 jugadores)
        self.fc_bbox = nn.Linear(512, 24 * 4)

        # Capa de salida para las clases (probabilidades)
        self.fc_class = nn.Linear(512, 24 * num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Aplanar la salida de las capas convolucionales

        x = self.dropout(self.relu4(self.fc1(x)))

        bbox_output = self.fc_bbox(x)
        class_output = self.fc_class(x)

        return bbox_output, class_output

def smooth_l1_loss(pred, target):
    diff = torch.abs(pred - target)
    less_than_one = (diff < 1.0).float()
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)

    return loss.mean()

def calculate_loss(bbox_pred, class_pred, target):
    bbox_target = target["boxes"]
    labels_target = target["labels"]

    # Ajustar las dimensiones de las predicciones
    bbox_pred = bbox_pred.view(-1, 24, 4)
    class_pred = class_pred.view(-1, 24, NUM_CLASSES)

    total_bbox_loss = 0
    total_class_loss = 0

    for i in range(bbox_target.size(0)):  # Iterar sobre cada imagen en el batch
        # Encontrar las etiquetas que no son -1 (las que no son padding)
        valid_indices = labels_target[i] != -1
        num_valid_boxes = valid_indices.sum().item()

        # Calcular la pérdida de bounding box solo para las predicciones válidas
        if num_valid_boxes > 0:
            # Seleccionar las predicciones y targets válidas
            bbox_pred_i_filtered = bbox_pred[i][:num_valid_boxes]
            bbox_target_i_filtered = bbox_target[i][:num_valid_boxes]

            bbox_loss = smooth_l1_loss(bbox_pred_i_filtered, bbox_target_i_filtered)
            total_bbox_loss += bbox_loss

        # Calcular la pérdida de clasificación solo para las predicciones válidas
        if num_valid_boxes > 0:
            class_pred_i_filtered = class_pred[i][:num_valid_boxes]
            labels_target_i_filtered = labels_target[i][:num_valid_boxes]

            class_loss = nn.CrossEntropyLoss()(class_pred_i_filtered, labels_target_i_filtered)
            total_class_loss += class_loss

    # Promediar las pérdidas sobre el número de imágenes en el batch
    return total_bbox_loss / bbox_target.size(0), total_class_loss / bbox_target.size(0)

def custom_collate(batch):
    """
    Función de collate personalizada para manejar tensores de diferentes tamaños.
    """
    # Separar imágenes y targets
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Manejar imágenes con default_collate
    images = torch.utils.data.default_collate(images)

    # Encontrar el número máximo de objetos en el batch
    max_num_boxes = max(target["boxes"].shape[0] for target in targets)

    # Rellenar las bounding boxes y las etiquetas
    padded_boxes = []
    padded_labels = []
    for target in targets:
        num_boxes = target["boxes"].shape[0]
        pad_size = max_num_boxes - num_boxes

        # Rellenar con ceros para las bounding boxes
        padded_box = torch.cat([target["boxes"], torch.zeros((pad_size, 4))], dim=0)
        padded_boxes.append(padded_box)

        # Rellenar con -1 para las etiquetas (se ignorarán en la función de pérdida)
        padded_label = torch.cat([target["labels"], torch.full((pad_size,), -1, dtype=torch.int64)], dim=0)
        padded_labels.append(padded_label)

    # Convertir a tensores
    padded_boxes = torch.stack(padded_boxes)
    padded_labels = torch.stack(padded_labels)

    # Crear el nuevo target
    new_targets = {"boxes": padded_boxes, "labels": padded_labels}

    return images, new_targets

# Transformaciones
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Datasets
train_dataset = SoccerDataset(TRAIN_DIR, IMAGE_SIZE, transform=transform)
val_dataset = SoccerDataset(VAL_DIR, IMAGE_SIZE, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)

# Modelo
model = SoccerObjectDetector(NUM_CLASSES).to(DEVICE)

# Optimizador
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target_boxes, target_labels = data.to(DEVICE), target["boxes"].to(DEVICE), target["labels"].to(DEVICE)
        target = {"boxes": target_boxes, "labels": target_labels}

        optimizer.zero_grad()
        bbox_output, class_output = model(data)
        bbox_loss, class_loss = calculate_loss(bbox_output, class_output, target)
        total_loss = bbox_loss + class_loss

        total_loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch+1}/{EPOCHS}, Batch: {batch_idx}/{len(train_loader)}, "
                  f"Total Loss: {total_loss.item():.4f}, BBox Loss: {bbox_loss.item():.4f}, "
                  f"Class Loss: {class_loss.item():.4f}")

    # Validación
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for data, target in val_loader:
            data, target_boxes, target_labels = data.to(DEVICE), target["boxes"].to(DEVICE), target["labels"].to(DEVICE)
            target = {"boxes": target_boxes, "labels": target_labels}

            bbox_output, class_output = model(data)
            bbox_loss, class_loss = calculate_loss(bbox_output, class_output, target)
            val_loss += bbox_loss.item() + class_loss.item()

        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
torch.save(model.state_dict(), "soccer_object_detector.pth")

def inference(model, image_path, image_size, device, confidence_threshold=0.5):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        bbox_output, class_output = model(image)

    # Procesar las salidas
    bbox_output = bbox_output.view(-1, 24, 4)
    class_output = class_output.view(-1, 24, NUM_CLASSES)
    probabilities = torch.softmax(class_output, dim=2)
    scores, predicted_classes = torch.max(probabilities, dim=2)

    # Filtrar por confianza
    mask = scores > confidence_threshold
    filtered_boxes = bbox_output[mask]
    filtered_classes = predicted_classes[mask]
    filtered_scores = scores[mask]

    # Convertir las coordenadas de las bounding boxes a píxeles
    filtered_boxes[:, 0] *= image_size[0]  # x_min
    filtered_boxes[:, 1] *= image_size[1]  # y_min
    filtered_boxes[:, 2] *= image_size[0]  # x_max
    filtered_boxes[:, 3] *= image_size[1]  # y_max

    return filtered_boxes, filtered_classes, filtered_scores

# Ejemplo de uso
model = SoccerObjectDetector(NUM_CLASSES)
model.load_state_dict(torch.load("soccer_object_detector.pth"))
model.to(DEVICE)

# Obtener una imagen de prueba del directorio de prueba
test_image_files = os.listdir(os.path.join(TEST_DIR, "images"))
test_image_path = os.path.join(TEST_DIR, "images", test_image_files[0]) # Coge la primera imagen de test
boxes, classes, scores = inference(model, test_image_path, IMAGE_SIZE, DEVICE)

# Mapeo de IDs a nombres de clases
class_names = {0: "Ball", 1: "Referee", 2: "Player"}

# Imprimir resultados
for box, class_id, score in zip(boxes, classes, scores):
    print(f"Class: {class_names[class_id.item()]}, Score: {score.item():.2f}, Box: {box.tolist()}")