if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    from ultralytics import YOLO

    model = YOLO('yolo11m.pt')
    results = model.train(
        data='data.yaml',   # Indicar la ruta del archivo data.yaml
        epochs=50,
        batch=16,
        patience=3,
        imgsz=648,
        device=0,
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        augment=False,
        save=True,  # Guardar el modelo en cada epoch
        project='runs/fine_tuning', # Directorio donde se guardarán los resultados
        name='acuity11m_v3',    # Nombre del modelo
        workers=16,
        verbose=True    # Mostrar información detallada del entrenamiento
    )

    print(results)
