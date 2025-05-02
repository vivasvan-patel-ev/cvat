import base64
import io
import json
import cv2

import yaml
from evml_inference import ModelStore
from evml_inference.models.arch import Yolov8
import numpy as np
import asyncio

import nest_asyncio
nest_asyncio.apply()

# from model_handler import ModelHandler
from PIL import Image

def init_context(context):
    context.logger.info("Init context...  0%")

    model = Yolov8(config="/opt/nuclio/aoi-config.yaml")
    # CLASS = model.config.input.class_name

    # Read labels
    # with open("/opt/nuclio/function.yaml", 'rb') as function_file:
    #     functionconfig = yaml.safe_load(function_file)

    # labels_spec = functionconfig['metadata']['annotations']['spec']
    # labels = {item['id']: item['name'] for item in json.loads(labels_spec)}

    # # Read the DL model
    # model = ModelHandler(labels)
    context.user_data.model = model

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run YoloV7 ONNX model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    image = Image.open(buf)
    image = np.array(image)

    pre_image = context.user_data.model.preprocess(image)
    preds = asyncio.run(context.user_data.model.inference(pre_image))
    results = context.user_data.model.postprocess(preds)

    output = []
    for idx, mask in enumerate(results.segmap):
        confidence = results.detection.scores[idx]
        class_ = results.detection.labels[idx]
        contours, _ = cv2.findContours(mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        contours = [c.reshape(-1).tolist() for c in contours]
        if len(contours[0]) < 6:
            continue
        output.append({
            'confidence': confidence,
            'label': class_,
            'points' : contours[0],
            'type' : 'polygon'
        })

    # results = context.user_data.model.infer(image, threshold)

    print(output)

    return context.Response(body=json.dumps(output), headers={},
        content_type='application/json', status_code=200)
