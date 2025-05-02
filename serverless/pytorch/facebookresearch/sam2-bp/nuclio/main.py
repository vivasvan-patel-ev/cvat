# Copyright (C) 2023-2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
import base64

from PIL import Image
import io
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.misc import variant_to_config_mapping
from sam2.utils.visualization import show_masks
import numpy as np
import time
import os

def init_context(context):
    print(os.listdir("."))
    print(os.getcwd())
    print(os.listdir("/opt/nuclio"))
    print(os.listdir("/opt/nuclio/sam2"))
    model = build_sam2(
        variant_to_config_mapping["base_plus"],
        "./sam2_hiera_base_plus.pt",
    )
    image_predictor = SAM2ImagePredictor(model)

    context.user_data.model = image_predictor
    context.user_data.image_hash = None
    context.logger.info("Init context...100%")

def calculate_image_hash(img: Image):
    """
    Calculate a perceptual hash for an image.

    This function resizes the input image to 10x10 pixels, converts it to
    grayscale, and then computes a hash based on the average pixel value.
    The hash is returned as a hexadecimal string.

    Args:
        img (Image): The input image to be hashed.

    Returns:
        str: The hexadecimal representation of the image hash.
    """
    img = img.resize((10, 10), Image.LANCZOS)
    img = img.convert("L")
    pixel_data = list(img.getdata())
    avg_pixel = sum(pixel_data)/len(pixel_data)
    bits = "".join(['1' if (px >= avg_pixel) else '0' for px in pixel_data])
    hex_representation = str(hex(int(bits, 2)))[2:][::-1].upper()
    return hex_representation




def handler(context, event):
    try:
        context.logger.info("call handler")
        data = event.body
        image_predictor = context.user_data.model

        buf = io.BytesIO(base64.b64decode(data["image"]))
        image = Image.open(buf)
        image = image.convert("RGB")
        image_hash = calculate_image_hash(image)
        image = np.array(image)
        if(context.user_data.image_hash != image_hash):
            image_predictor.set_image(image)
            context.user_data.image_hash = image_hash

        pos_points = np.array(data["pos_points"])
        neg_points = np.array(data["neg_points"])

        if pos_points.ndim == 1:
            pos_points = pos_points.reshape(-1, 2)  # Assuming 2D points

        if neg_points.ndim == 1:
            neg_points = neg_points.reshape(-1, pos_points.shape[1])

        input_point = np.concatenate((pos_points, neg_points), axis=0)
        input_label = np.concatenate((np.ones(len(pos_points)), np.zeros(len(neg_points))), axis=0)
        masks, scores, logits = image_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=None,
            multimask_output=True,
        )

        print("masks", masks, masks.shape, "scores", scores, scores.shape, "logits", logits, logits.shape)

        return context.Response(
            body=json.dumps({'mask': masks[np.argmax(scores)].tolist()}),
            headers={},
            content_type='application/json',
            status_code=200
        )
    except Exception as e:
        context.logger.error(f"Error in handler: {str(e)}")
        return context.Response(
            body=json.dumps({'error': str(e)}),
            headers={},
            content_type='application/json',
            status_code=500
        )
