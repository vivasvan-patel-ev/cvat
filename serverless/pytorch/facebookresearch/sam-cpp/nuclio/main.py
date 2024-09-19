# Copyright (C) 2023-2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
import base64
from PIL import Image
import io
import subprocess
import requests
import numpy as np
import cv2
import imutils
import time


def init_context(context):
    # model = ModelHandler()
    # context.user_data.model = model
    # Open a log file in write mode
    with open("output.log", "w") as log_file:
        # Start the process and redirect stdout and stderr to the log file
        process = subprocess.Popen(
            ["./build/bin/sam", "-m", "./checkpoints/ggml-model-f16.bin"],
            stdout=log_file,
            stderr=log_file,
            universal_newlines=True,  # Ensure the output is in text mode
        )

    context.logger.info("Init context...100%")


def handler(context, event):
    try:
        start_time_total = time.time()
        context.logger.info("call handler")
        data = event.body

        # Step 1: Decode and load the image
        start_time = time.time()
        image = Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGB")
        original_orientation = image.size  # Save original dimensions (width, height)
        width, height = original_orientation
        elapsed_time = time.time() - start_time
        context.logger.info(f"Time to decode and load image: {elapsed_time} seconds")

        # Step 2: Rotate image to landscape if it's portrait
        rotated = False
        start_time = time.time()
        if image.height > image.width:  # Portrait detected
            image = image.rotate(90, expand=True)  # Rotate to landscape
            rotated = True
            context.logger.info("Rotated image from portrait to landscape.")
            # Update pos_points to reflect the rotation
            pos_points = [(y, width - x) for (x, y) in data["pos_points"]]
        else:
            pos_points = data["pos_points"]  # No rotation needed
        elapsed_time = time.time() - start_time
        context.logger.info(
            f"Time to check and rotate image if needed: {elapsed_time} seconds"
        )

        # Step 3: Save image to a temporary in-memory buffer
        start_time = time.time()
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)
        elapsed_time = time.time() - start_time
        context.logger.info(f"Time to save image in memory: {elapsed_time} seconds")

        # Step 4: Prepare image payload
        files = {"image": img_byte_arr}

        # Step 5: Time and send the request
        start_time = time.time()
        mask_response = requests.post(
            f"http://localhost:42069/generate_mask?x={pos_points[0][0]}&y={pos_points[0][1]}",
            files=files,
        )
        elapsed_time = time.time() - start_time
        context.logger.info(
            f"Time to send request and get response: {elapsed_time} seconds"
        )

        # Step 6: Handle response
        start_time = time.time()
        if mask_response.status_code != 200:
            return context.Response(
                body=json.dumps({"mask": []}),
                content_type="application/json",
                status_code=200,
            )

        elapsed_time = time.time() - start_time
        context.logger.info(f"Time to check response status: {elapsed_time} seconds")

        # Step 7: Convert response content to grayscale image
        start_time = time.time()
        mask_image = Image.open(io.BytesIO(mask_response.content))
        gray_image = cv2.cvtColor(np.array(mask_image), cv2.COLOR_RGB2GRAY)

        # Rotate back to original orientation if the image was rotated
        start_time = time.time()
        if rotated:
            gray_image = np.rot90(gray_image, -1)  # Rotate back to portrait
            context.logger.info(
                "Rotated grayscale image back to original portrait orientation."
            )
        elapsed_time = time.time() - start_time
        context.logger.info(
            f"Time to rotate grayscale image back to original orientation if needed: {elapsed_time} seconds"
        )

        elapsed_time = time.time() - start_time
        context.logger.info(
            f"Time to process response and convert to grayscale: {elapsed_time} seconds"
        )

        total_elapsed_time = time.time() - start_time_total
        context.logger.info(
            f"Total handler execution time: {total_elapsed_time} seconds"
        )

        return context.Response(
            body=json.dumps({"mask": gray_image.tolist()}),
            content_type="application/json",
            status_code=200,
        )

    except Exception as e:
        context.logger.error(f"Error in handler: {str(e)}")
        return context.Response(
            body=json.dumps({"error": str(e)}),
            content_type="application/json",
            status_code=500,
        )
