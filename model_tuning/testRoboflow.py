from roboflow import Roboflow
import cv2
import matplotlib.pyplot as plt

# Configura tu API key y la información de tu proyecto
rf = Roboflow(api_key="API_KEY")  # Reemplaza con tu API key de Roboflow
project = rf.workspace("futbolcv").project("acuity-ctzp4")
model = project.version(5).model


# Define las clases para una mejor visualización
class_names = {
    0: "Referee",
    1: "Ball",
    2: "Player"
}

def annotate_image(image_path, model):
    """
    Predice las detecciones en una imagen, dibuja las anotaciones y muestra la imagen.

    Args:
        image_path (str): Ruta a la imagen que se va a anotar.
        model: Modelo de Roboflow cargado.
    """

    # Realiza la predicción
    predictions = model.predict(image_path, confidence=40, overlap=30).json()

    # Carga la imagen con OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convierte de BGR a RGB para matplotlib

    # Dibuja las anotaciones en la imagen
    for prediction in predictions['predictions']:
        x = prediction['x']
        y = prediction['y']
        width = prediction['width']
        height = prediction['height']
        class_id = prediction['class_id']
        confidence = prediction['confidence']

        # Calcula las coordenadas de las esquinas del cuadro delimitador
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)

        # Define el color del cuadro delimitador (puedes personalizarlo)
        color = (0, 255, 0)  # Verde

        # Dibuja el rectángulo
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Prepara el texto de la etiqueta
        label = f"{class_names.get(class_id, class_id)}: {confidence:.2f}"

        # Calcula el tamaño del texto para posicionarlo correctamente
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        # Dibuja un fondo para el texto (opcional)
        cv2.rectangle(image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)

        # Dibuja el texto de la etiqueta
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Muestra la imagen con matplotlib
    plt.imshow(image)
    plt.title("Imagen con Anotaciones")
    plt.axis('off')  # Oculta los ejes
    plt.show()

# Ejemplo de uso:
image_path = "../data_prueba/frame_3528.jpg"  # Reemplaza con la ruta a tu imagen
annotate_image(image_path, model)