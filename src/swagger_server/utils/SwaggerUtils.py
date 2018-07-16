import base64
import io
import numpy as np

from PIL import Image


def convert_image_array_to_byte_string(image_array, channels=1, normalize=True):
    """
    converts a numpy array into a PNG byte string
    :param image_array:
    :param channels:
    :param normalize:
    :return:
    """

    image_array = np.array(image_array)

    img_array_min = np.min(image_array)
    img_array_max = np.max(image_array)

    if normalize:
        image_array = (image_array - img_array_min) / (img_array_max - img_array_min) * 255

    if channels == 1:
        # convert array values to uint8 and reshape to 2D
        image_data = image_array.reshape(28, 28).astype('uint8')
    else:
        # only convert to uint8
        image_data = image_array.astype('uint8')

    image = Image.fromarray(image_data)

    # generate byte string
    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format='PNG')
    image_byte_array = image_byte_array.getvalue()

    # return byte string
    return str(base64.b64encode(image_byte_array))
