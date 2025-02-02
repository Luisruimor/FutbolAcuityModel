import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def Acuity11modelImg(model_path, image_path, CLASS_ID, CONFIDENCE_THRESHOLD, IMAGE_SIZE, output_path='imagenes/magen_nueva_detectada.jpg'):
    # Carga del Modelo
    print("Cargando el modelo entrenado...")
    model = YOLO(model_path)

    print(f"Realizando detección en la imagen: {image_path}")
    results = model.predict(
        source=image_path,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        save=False,       # No guardar automáticamente las imágenes
        save_txt=False,   # No guardar los resultados en .txt
        show=False        # No mostrar la ventana de detección
    )

    # Filtrar las detecciones de la CLASS_ID

    # Cargar la imagen original usando OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen en la ruta: {image_path}")

    # Iterar sobre las detecciones
    for result in results:
        boxes = result.boxes  # Acceder a las cajas detectadas
        for box in boxes:
            cls = int(box.cls)       # ID de clase
            conf = float(box.conf)   # Confianza de la detección
            if cls == CLASS_ID:
                # Coordenadas de la caja delimitadora
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])

                # Dibujar la caja en la imagen
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)

                # Etiqueta con el nombre de la clase y la confianza
                label = f"Player A {conf:.2f}"
                cv2.putText(
                    image,
                    label,
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    thickness=1
                )

    # Ruta para guardar la imagen con detecciones
    cv2.imwrite(output_path, image)
    print(f"Imagen con detecciones guardada en: {output_path}")

    # Mostrar la imagen usando Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title("Detecciones de Player ")
    plt.show()

def Acuity11modelVideo(model_path, video_path, CLASS_ID, CONFIDENCE_THRESHOLD, IMAGE_SIZE, output_path='video/video_detectado.mp4'):
    print("Cargando el modelo entrenado...")
    model = YOLO(model_path)

    # Captura de Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"No se pudo abrir el video: {video_path}")
        return

    # Propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    # Calcular el número máximo de fotogramas a procesar (5 minutos)
    # Saltar al fotograma inicial del intervalo deseado
    start_frame = int(10 * 60 * fps)  # Por ejemplo, minuto 10
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Calcular el número máximo de fotogramas para el intervalo deseado
    end_frame = int(15 * 60 * fps)  # Por ejemplo, minuto 15
    max_frames = end_frame - start_frame

    #max_frames = int(MAX_DURATION_SECONDS * fps)

    # Configurar VideoWriter para guardar el video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break  # Fin del video

        frame_count += 1

        # Realizar detección en el frame actual
        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            save=False,
            save_txt=False,
            show=False
        )

        # Dibujar las detecciones en el frame
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls)  # ID de clase
                conf = float(box.conf)  # Confianza de la detección
                if cls == CLASS_ID:
                    # Coordenadas de la caja delimitadora
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])

                    # Dibujar la caja
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    # Etiqueta con la clase y la confianza
                    label = f"Player A {conf:.2f}"
                    cv2.putText(frame, label, (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Mostrar en pantalla (opcional)
        cv2.imshow("Detecciones de Player", frame)

        # Guardar el frame procesado en el video de salida
        out.write(frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = 'runs/fine_tuning/acuity11m_v2/weights/best.pt'
    image_path = '../data_prueba/frame_3528.jpg'  # Cambia esta ruta según corresponda

    # Clase a detectar (Referee = 80, Player= 81, Ball = 32)
    CLASS_ID_PLAYER_A = 81
    # Umbral de confianza para las detecciones
    CONFIDENCE_THRESHOLD = 0.55
    # Tamaño de la imagen de entrada
    IMAGE_SIZE = 1024  # Debe coincidir con el tamaño usado durante el entrenamiento

    video_path = '../data/video/rma.v.get.1.laliga.01.12.2024.fullmatchsports.com.mp4'

    Acuity11modelVideo(model_path, video_path, CLASS_ID_PLAYER_A, CONFIDENCE_THRESHOLD, IMAGE_SIZE)
    #Acuity11modelImg(model_path, image_path, CLASS_ID_PLAYER_A, CONFIDENCE_THRESHOLD, IMAGE_SIZE)