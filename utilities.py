import os
import random
import shutil
import cv2

def copy_files(ruta_imagenes, ruta_anotaciones, output_dir, lista_imagenes, conjunto):
    """
    Copia archivos de imágenes y sus correspondientes anotaciones a una carpeta específica dentro de un directorio de salida.

    Args:
        ruta_imagenes (str): Ruta al directorio que contiene las imágenes originales.
        ruta_anotaciones (str): Ruta al directorio que contiene las anotaciones (archivos .txt).
        output_dir (str): Ruta al directorio de salida donde se copiarán los archivos.
        lista_imagenes (list): Lista de nombres de imágenes a copiar.
        conjunto (str): Nombre del subconjunto (train, val, test) al que pertenecen las imágenes.

    Returns:
        None
    """
    for img in lista_imagenes:
        # Crea el directorio del conjunto si no existe
        os.makedirs(f"{output_dir}/{conjunto}", exist_ok=True)

        # Copia la imagen al directorio del conjunto
        shutil.copy(os.path.join(ruta_imagenes, img), f"{output_dir}/{conjunto}/{img}")

        # Busca el archivo de anotación correspondiente y lo copia si existe
        txt_path = os.path.join(ruta_anotaciones, img.replace('.jpg', '.txt'))
        if os.path.exists(txt_path):
            shutil.copy(txt_path, f"{output_dir}/{conjunto}/{img.replace('.jpg', '.txt')}")

def split_dataset(ruta_imagenes, ruta_anotaciones, output_dir):
    """
    Divide un conjunto de datos en tres subconjuntos: entrenamiento, validación y prueba, asegurando que las anotaciones sean incluidas.

    Args:
        ruta_imagenes (str): Ruta al directorio que contiene las imágenes.
        ruta_anotaciones (str): Ruta al directorio que contiene las anotaciones (archivos .txt).
        output_dir (str): Ruta al directorio de salida donde se almacenarán los subconjuntos divididos.

    Returns:
        None: La función no retorna ningún valor, pero crea carpetas con las imágenes y anotaciones divididas.
    """
    os.makedirs(output_dir, exist_ok=True)
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        os.makedirs(f"{output_dir}/{subset}", exist_ok=True)

    # Filtra las imágenes que tienen una anotación correspondiente
    imagenes_utilizables = [
        img for img in os.listdir(ruta_imagenes)
        if os.path.exists(os.path.join(ruta_anotaciones, img.replace('.jpg', '.txt')))
    ]

    imagenes_con_anotaciones = []
    imagenes_sin_anotaciones = []
    for img in imagenes_utilizables:
        txt_path = os.path.join(ruta_anotaciones, img.replace('.jpg', '.txt'))
        # Clasifica imágenes según si tienen anotaciones válidas (no vacías)
        if os.path.getsize(txt_path) > 0:
            imagenes_con_anotaciones.append(img)
        else:
            imagenes_sin_anotaciones.append(img)

    # Mezcla aleatoriamente las imágenes con anotaciones
    random.shuffle(imagenes_con_anotaciones)
    num_train_con = int(len(imagenes_con_anotaciones) * 0.7)
    num_val_con = int(len(imagenes_con_anotaciones) * 0.2)

    # Divide las imágenes con anotaciones en entrenamiento, validación y prueba
    train_con = imagenes_con_anotaciones[:num_train_con]
    val_con = imagenes_con_anotaciones[num_train_con:num_train_con + num_val_con]
    test_con = imagenes_con_anotaciones[num_train_con + num_val_con:]

    # Mezcla aleatoriamente las imágenes sin anotaciones
    random.shuffle(imagenes_sin_anotaciones)
    num_train_sin = int(len(imagenes_sin_anotaciones) * 0.9)

    # Divide las imágenes sin anotaciones en entrenamiento y prueba
    train_sin = imagenes_sin_anotaciones[:num_train_sin]
    test_sin = imagenes_sin_anotaciones[num_train_sin:]

    # Combina imágenes con y sin anotaciones para formar los subconjuntos finales
    train = train_con + train_sin
    val = val_con
    test = test_con + test_sin

    # Copia los archivos a sus respectivos directorios
    copy_files(ruta_imagenes, ruta_anotaciones, output_dir, train, "train")
    copy_files(ruta_imagenes, ruta_anotaciones, output_dir, val, "val")
    copy_files(ruta_imagenes, ruta_anotaciones, output_dir, test, "test")

def mix_frames(directorio_principal, num_imagenes, directorio_salida):
    """
    Selecciona aleatoriamente num_imagenes de entre todas las subcarpetas
    del directorio_principal y copia esos archivos al directorio_salida.

    directorio_principal: ruta al directorio donde están las subcarpetas con los frames
    num_imagenes: número total de imágenes a seleccionar
    directorio_salida: ruta al directorio donde se copiarán las imágenes seleccionadas
    """
    os.makedirs(directorio_salida, exist_ok=True)

    # Recolectaar los frames del directorio principal
    todas_las_imagenes = []
    for ruta_raiz, directorios, ficheros in os.walk(directorio_principal):
        for fichero in ficheros:
            # Verifica que sea un JPG (puedes adaptarlo para PNG, etc.)
            if fichero.lower().endswith('.jpg'):
                ruta_completa = os.path.join(ruta_raiz, fichero)
                todas_las_imagenes.append(ruta_completa)

    # Verificar que haya suficientes imágenes
    if len(todas_las_imagenes) < num_imagenes:
        num_imagenes = len(todas_las_imagenes)

    # Mezclar las imagenes
    random.shuffle(todas_las_imagenes)
    imagenes_seleccionadas = todas_las_imagenes[:num_imagenes]

    # Distribuir las imágenes seleccionadas en el directorio de salida
    for img_path in imagenes_seleccionadas:
        nombre_archivo = os.path.basename(img_path)
        destino = os.path.join(directorio_salida, nombre_archivo)
        shutil.copy2(img_path, destino)

    print(f"Se han copiado {num_imagenes} imágenes en la carpeta: {directorio_salida}")

def actualizar_anotaciones(rutas, id_original, id_nuevo):
    """
    Actualiza los archivos de anotaciones reemplazando un id específico por otro.

    Args:
        rutas (list): Lista de rutas de carpetas donde se encuentran los archivos.
        id_original (str): ID que se desea reemplazar.
        id_nuevo (str): Nuevo ID que reemplazará al original.
    """
    for ruta in rutas:
        if not os.path.exists(ruta):
            print(f"La ruta {ruta} no existe. Se omite.")
            continue

        # Procesar cada archivo en la carpeta
        for file_name in os.listdir(ruta):
            if file_name.endswith('.txt'):
                txt_file_path = os.path.join(ruta, file_name)

                # Leer el contenido del archivo
                with open(txt_file_path, 'r') as f:
                    lines = f.readlines()

                # Actualizar las líneas con el ID cambiado
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:  # Validar que la línea tenga 5 partes
                        if parts[0] == id_original:
                            parts[0] = id_nuevo
                        new_line = ' '.join(parts) + '\n'
                        new_lines.append(new_line)
                    else:
                        new_lines.append(line)

                # Escribir los cambios de vuelta al archivo
                with open(txt_file_path, 'w') as f:
                    f.writelines(new_lines)

    print("Proceso finalizado. Se han actualizado las anotaciones.")

def extract_frames(video_path, output_dir, frame_rate):
    """
    Extrae frames de un video y los guarda en una carpeta.

    Args:
        video_path (str): Ruta al archivo de video.
        output_dir (str): Carpeta donde se guardarán los frames.
        frame_rate (int): Cada cuántos frames extraer (1 = todos los frames, 2 = cada 2 frames, etc.).
    """
    # Crear la carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Cargar el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error al abrir el video: {video_path}")
        return

    frame_count = 0
    extracted_count = 0

    # Iterar sobre los frames del video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: # Si no hay frame para leer, salir del bucle
            break

        # Extraer los frames deseados según el frame_rate
        if frame_count % frame_rate == 0:
            frame_path = os.path.join(output_dir, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_count += 1

        frame_count += 1

    cap.release()
    print(f"Se han extraído {extracted_count} frames y guardado en '{output_dir}'.")

def extract_frames_videoDir(video_input, frames_dir, frame_rate=20):
    """
    Procesa uno o varios videos dependiendo de si video_input es un solo archivo o un directorio.

    Args:
        video_input (str): Ruta a un archivo de video o a un directorio con varios videos.
        frames_dir  (str): Carpeta base donde se guardarán los frames de cada video.
        frame_rate  (int): Extraer cada N frames (por defecto 20).
    """

    # 1. Comprobar si el input es un archivo o un directorio
    if os.path.isfile(video_input):
        # 1.1. Es un solo archivo de video
        video_path = video_input
        video_name, _ = os.path.splitext(os.path.basename(video_path))
        output_dir = os.path.join(frames_dir, video_name)

        print(f"Procesando archivo de video único: {video_path}")
        extract_frames(video_path, output_dir, frame_rate)

    elif os.path.isdir(video_input):
        # 1.2. Es un directorio con varios videos
        video_files = os.listdir(video_input)
        # Crear el directorio frames, si no existe
        os.makedirs(frames_dir, exist_ok=True)

        # Para controlar qué carpetas de frames ya existen
        frames_folders = set(os.listdir(frames_dir))

        for vf in video_files:
            # Ignoramos ficheros que no parezcan videos (ej. extensiones no deseadas)
            # Puedes personalizar la lista de extensiones
            if not (vf.lower().endswith('.mp4') or vf.lower().endswith('.avi') or vf.lower().endswith('.mov')):
                continue

            video_name, _ = os.path.splitext(vf)
            # Si no existe carpeta con ese nombre en frames_dir, procesamos el video
            if video_name not in frames_folders:
                video_path = os.path.join(video_input, vf)
                output_dir = os.path.join(frames_dir, video_name)

                print(f"Procesando video: {vf}")
                extract_frames(video_path, output_dir, frame_rate)
            else:
                print(f"El video '{vf}' ya fue procesado (se encontró la carpeta '{video_name}').")
    else:
        print(f"ERROR: '{video_input}' no es un archivo ni un directorio válido.")

def generar_lista_imagenes(carpeta_imagenes, archivo_salida):
    """
    Genera un archivo de texto con la lista de imágenes en la carpeta especificada.

    :param carpeta_imagenes: Ruta de la carpeta que contiene las imágenes.
    :param archivo_salida: Nombre del archivo de salida donde se guardará la lista.
    """
    # Verifica si la carpeta existe
    if not os.path.isdir(carpeta_imagenes):
        print(f"La carpeta {carpeta_imagenes} no existe.")
        return

    # Extensiones de imágenes que se van a considerar
    extensiones = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

    try:
        # Abre el archivo de salida en modo escritura
        with open(archivo_salida, 'w') as f:
            # Recorre todos los archivos en la carpeta
            for nombre_archivo in os.listdir(carpeta_imagenes):
                # Verifica si el archivo tiene una extensión de imagen
                if nombre_archivo.lower().endswith(extensiones):
                    # Construye la ruta relativa
                    ruta = os.path.join(carpeta_imagenes, nombre_archivo)
                    # Escribe la ruta en el archivo de texto
                    f.write(f"{ruta}\n")

        print(f"Lista de imágenes guardada en {archivo_salida}")
    except Exception as e:
        print(f"Ocurrió un error al escribir el archivo: {e}")