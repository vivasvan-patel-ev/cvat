# Copyright (C) 2023-2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
import base64
import boto3
from PIL import Image
import io
import numpy as np
import os
import traceback
from botocore.exceptions import ClientError

# SageMaker endpoint configuration
REGION = 'us-east-2'  # Update with your region
ENDPOINT_NAME = 'sam2-prompted'  # Your endpoint name

def init_context(context):
    try:
        context.user_data.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=REGION)
        context.user_data.image_hash = None
        context.logger.info("Init context...100%")
    except Exception as e:
        context.logger.error(f"Error in init_context: {str(e)}")
        context.logger.error(traceback.format_exc())
        raise

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

def decode_rle_mask(encoded_mask, mask_shape):
    """
    Decode run-length encoded mask.

    Args:
        encoded_mask (list): List of [value, count] pairs
        mask_shape (list): Shape of the mask [height, width]

    Returns:
        numpy.ndarray: Decoded binary mask
    """
    mask = np.zeros(mask_shape[0] * mask_shape[1], dtype=np.uint8)
    idx = 0
    for value, count in encoded_mask:
        mask[idx:idx + count] = value
        idx += count
    return mask.reshape(mask_shape)

def handler(context, event):
    try:
        context.logger.info("Starting handler execution")
        data = event.body
        if isinstance(data, str):
            data = json.loads(data)
        context.logger.info(f"Received data keys: {data.keys()}")

        sagemaker_runtime = context.user_data.sagemaker_runtime

        # Get image from request
        context.logger.info("Processing image")
        buf = io.BytesIO(base64.b64decode(data["image"]))
        image = Image.open(buf)
        image = image.convert("RGB")
        image_hash = calculate_image_hash(image)
        context.logger.info(f"Image processed, hash: {image_hash}")

        # Prepare points
        context.logger.info("Processing points")
        pos_points = np.array(data["pos_points"])
        neg_points = np.array(data["neg_points"])
        context.logger.info(f"Points shapes - pos: {pos_points.shape}, neg: {neg_points.shape}")

        if pos_points.ndim == 1:
            pos_points = pos_points.reshape(-1, 2)

        if neg_points.ndim == 1:
            neg_points = neg_points.reshape(-1, pos_points.shape[1])

        input_point = np.concatenate((pos_points, neg_points), axis=0)
        input_label = np.concatenate((np.ones(len(pos_points)), np.zeros(len(neg_points))), axis=0)
        context.logger.info(f"Final input shapes - points: {input_point.shape}, labels: {input_label.shape}")

        # Prepare payload for SageMaker
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()

        payload = {
            'img': img_base64,
            'input_point': input_point.tolist(),
            'input_label': input_label.tolist()
        }
        context.logger.info("Payload prepared for SageMaker")

        try:
            # Call SageMaker endpoint
            context.logger.info(f"Calling SageMaker endpoint: {ENDPOINT_NAME}")
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            context.logger.info("SageMaker response received")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            context.logger.error(f"AWS Error - Code: {error_code}, Message: {error_message}")
            return context.Response(
                body=json.dumps({
                    'error': f"AWS Error: {error_code} - {error_message}",
                    'details': str(e)
                }),
                headers={},
                content_type='application/json',
                status_code=500
            )

        # Parse response
        context.logger.info("Parsing response")
        result = json.loads(response['Body'].read().decode())

        # Extract encoded masks and shape
        encoded_masks = result['encoded_masks']
        mask_shape = result['mask_shape']
        scores = result['scores']

        context.logger.info(f"Response parsed - mask shape: {mask_shape}, scores: {len(scores)}")

        # Decode the best mask
        best_mask_idx = np.argmax(scores)
        best_encoded_mask = encoded_masks[best_mask_idx]
        best_mask = decode_rle_mask(best_encoded_mask, mask_shape)

        context.logger.info("Returning response")

        return context.Response(
            body=json.dumps({'mask': best_mask.tolist()}),
            headers={},
            content_type='application/json',
            status_code=200
        )
    except Exception as e:
        context.logger.error(f"Error in handler: {str(e)}")
        context.logger.error(traceback.format_exc())
        return context.Response(
            body=json.dumps({
                'error': str(e),
                'traceback': traceback.format_exc()
            }),
            headers={},
            content_type='application/json',
            status_code=500
        )
