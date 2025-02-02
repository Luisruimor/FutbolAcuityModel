import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import albumentations as A
import cv2

def augment_dataset(input_dir, output_dir, augmentations_per_image):
    """
    Amplía un dataset de imágenes y anotaciones YOLO utilizando Albumentations.

    Args:
        input_dir: Directorio que contiene las imágenes y archivos de anotaciones.
        output_dir: Directorio donde se guardarán las imágenes aumentadas y sus anotaciones.
        augmentations_per_image: Número de imágenes aumentadas a generar por cada imagen original.
    """

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Definir la pipeline de transformaciones
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(shear={'x': (-4, 4), 'y': (-15, 15)}, p=0.5),
        A.ToGray(p=0.15),
        A.ColorJitter(saturation=0.19, p=1),  # Saturation between -19% and +19%
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

    # Recorrer los archivos del directorio de entrada
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(input_dir, filename)
            label_path = os.path.join(input_dir, filename[:-4] + ".txt")

            # Verificar si existe el archivo de anotaciones
            if not os.path.exists(label_path):
                print(f"No se encontró el archivo de anotaciones para {filename}. Se omitirá esta imagen.")
                continue

            # Leer la imagen
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Leer las anotaciones YOLO
            bboxes = []
            category_ids = []
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())
                        bboxes.append([x_center, y_center, width, height])
                        category_ids.append(int(class_id))
            except Exception as e:
                print(f"Error al leer las anotaciones de {label_path}: {e}")
                continue

            # Copiar la imagen original al directorio de salida
            original_image_path = os.path.join(output_dir, filename)
            cv2.imwrite(original_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # Copiar la etiqueta original al directorio de salida
            original_label_path = os.path.join(output_dir, filename[:-4] + ".txt")
            with open(label_path, 'r') as f_in, open(original_label_path, 'w') as f_out:
                f_out.write(f_in.read())

            # Aplicar las transformaciones y guardar las imágenes aumentadas
            for i in range(augmentations_per_image):
                try:
                    augmented = transform(image=image, bboxes=bboxes, category_ids=category_ids)
                    augmented_image = augmented['image']
                    augmented_bboxes = augmented['bboxes']

                    # Guardar la imagen aumentada
                    augmented_filename = f"{filename[:-4]}_aug_{i}.jpg"
                    augmented_image_path = os.path.join(output_dir, augmented_filename)
                    cv2.imwrite(augmented_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

                    # Guardar las anotaciones aumentadas
                    augmented_label_path = os.path.join(output_dir, f"{filename[:-4]}_aug_{i}.txt")
                    with open(augmented_label_path, 'w') as f:
                        for j, bbox in enumerate(augmented_bboxes):
                            x_center, y_center, width, height = bbox
                            class_id = category_ids[j]
                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

                except Exception as e:
                    print(f"Error al aplicar la transformación a {filename} en la iteración {i}: {e}")
                    # Imprimir información útil para la depuración
                    print(f"  Tamaño de la imagen: {image.shape}")
                    print(f"  Número de bounding boxes: {len(bboxes)}")
                    if len(bboxes) > 0:
                        print(f"  Primer bounding box: {bboxes[0]}")
                    continue


if __name__ == '__main__':
    input_directory = "data/annotations/mix.laliga.24.01.2025.first.300 - copia/obj_train_data"
    output_directory = "data/annotations/mix.laliga.24.01.2025.first.300 - copia/prueba2"
    num_augmentations = 10  # Generar 5 imágenes aumentadas por cada imagen original

    augment_dataset(input_directory, output_directory, num_augmentations)