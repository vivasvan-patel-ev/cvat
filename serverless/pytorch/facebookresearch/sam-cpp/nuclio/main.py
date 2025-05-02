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
import ctypes
import platform
import os


# Define the params structure
class SamParams(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("model", ctypes.c_char * 256),
        ("fname_inp", ctypes.c_char * 256),
        ("fname_out", ctypes.c_char * 256),
    ]

    def __init__(
        self,
        seed=-1,
        n_threads=4,
        model="./checkpoints/ggml-model-f16.bin",
        fname_inp="./img.jpg",
        fname_out="img.png",
    ):
        self.seed = seed
        self.n_threads = n_threads
        self.model = model.encode("utf-8")
        self.fname_inp = fname_inp.encode("utf-8")
        self.fname_out = fname_out.encode("utf-8")


def init_context(context):
    # model = ModelHandler()
    # Load the shared library
    architecture = platform.machine()

    # Set the correct path based on architecture
    if architecture == "x86_64" or architecture == "64bit":
        lib_path = "x64"
    elif architecture == "arm64":
        lib_path = "arm"
    elif architecture == "aarch64":
        lib_path = "aarch"
    else:
        raise Exception(f"Unsupported architecture: {architecture}")
    print(f"Loading shared library from: ./release/{lib_path}/libmask.so")
    print(os.getcwd())
    print(os.listdir("./release"))
    print(os.listdir(f"./release/{lib_path}"))

    lib = ctypes.CDLL(f"./release/{lib_path}/libmask.so")
    context.user_data.model = lib
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

        context.logger.info(f"pos_points before transform: {data['pos_points']} width:{width} height:{height}")
        # Step 2: Rotate image to landscape if it's portrait
        rotated = False
        start_time = time.time()
        if image.height > image.width:  # Portrait detected
            image = image.rotate(90, expand=True)  # Rotate to landscape
            rotated = True
            context.logger.info("Rotated image from portrait to landscape.")
            # Update pos_points to reflect the rotation
            pos_points = [(y, width-x) for (x, y) in data["pos_points"]]
            width, height = image.size  # Update width and height
        else:
            pos_points = data["pos_points"]  # No rotation needed

        context.logger.info(f"pos_points after transform: {data['pos_points']} width:{width} height:{height}")

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

        # Step 4: Convert image to RGB format and flatten to a list
        start_time = time.time()
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert to BGR
        image_data = image_rgb.flatten().tolist()  # Flatten to 1D list
        elapsed_time = time.time() - start_time
        context.logger.info(
            f"Time to convert image to RGB and flatten: {elapsed_time} seconds"
        )

        # Step 5: Create an instance of SamParams and set values
        params = SamParams()

        # Step 6: Prepare output variables
        output_data = ctypes.POINTER(ctypes.c_ubyte)()
        output_size = ctypes.c_int()

        # Step 7: Call the function to generate the mask
        start_time = time.time()

        # Hard code data as correct types
        # image_data = (ctypes.c_ubyte * len(image_data))(*image_data)
        c_width = ctypes.c_int(width)
        c_height = ctypes.c_int(height)
        x = ctypes.c_float(pos_points[0][0])  # Replace with actual value if needed
        y = ctypes.c_float(pos_points[0][1])  # Replace with actual value if needed
        seed = ctypes.c_int(params.seed)
        n_threads = ctypes.c_int(params.n_threads)
        model = ctypes.c_char_p(params.model)
        fname_inp = ctypes.c_char_p(params.fname_inp)
        fname_out = ctypes.c_char_p(params.fname_out)
        output_data = ctypes.POINTER(ctypes.c_ubyte)()
        output_size = ctypes.c_int()
        # Call the function to generate the mask
        context.user_data.model.generate_mask_wrapper(
            (ctypes.c_ubyte * len(image_data))(*image_data),
            c_width,
            c_height,
            x,  # x coordinate (replace with actual value)
            y,  # y coordinate (replace with actual value)
            seed,
            n_threads,
            model,  # Pass model as bytes
            fname_inp,  # Pass fname_inp as bytes
            fname_out,  # Pass fname_out as bytes
            ctypes.byref(output_data),
            ctypes.byref(output_size),
        )

        context.user_data.model.generate_mask_wrapper.restype = None
        elapsed_time = time.time() - start_time
        context.logger.info(f"Time to generate mask: {elapsed_time} seconds")

        if output_size.value == 0:
            context.logger.error("No mask found!")
            return context.Response(
                body=json.dumps({"mask": []}),
                content_type="application/json",
                status_code=200,
            )

        # Step 8: Convert the pointer to bytes
        start_time = time.time()
        bytes_data = ctypes.string_at(output_data, output_size.value)
        elapsed_time = time.time() - start_time
        context.logger.info(
            f"Time to convert output pointer to bytes: {elapsed_time} seconds"
        )

        # Step 9: Convert the output to a numpy array
        start_time = time.time()
        mask_data = np.frombuffer(bytes_data, dtype=np.uint8)
        mask_image = mask_data.reshape(
            -1, width
        )  # Adjust if necessary based on output format
        elapsed_time = time.time() - start_time
        context.logger.info(
            f"Time to convert output to numpy array: {elapsed_time} seconds"
        )

        # Step 10: Rotate back to original orientation if the image was rotated
        start_time = time.time()
        if rotated:
            mask_image = np.rot90(mask_image, -1)  # Rotate back to portrait
            context.logger.info(
                "Rotated mask image back to original portrait orientation."
            )
        elapsed_time = time.time() - start_time
        context.logger.info(
            f"Time to rotate mask image back to original orientation if needed: {elapsed_time} seconds"
        )

        total_elapsed_time = time.time() - start_time_total
        context.logger.info(
            f"Total handler execution time: {total_elapsed_time} seconds"
        )

        return context.Response(
            body=json.dumps({"mask": mask_image.tolist()}),
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
