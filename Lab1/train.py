from ultralytics import YOLO
if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov3-tinyu.yaml")  # build a new model from scratch
    # model = YOLO("yolov8.pt")  # load a pretrained model (recommended for training)

    # Use the model
    results = model.train(data="edu_train.yaml", epochs=30, batch=12, device=0, imgsz=640, patience=5,
                          name='yolov3-tiny', v5loader=True)  # train the model
    results = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # success = model.export(format="onnx")  # export the model to ONNX format
