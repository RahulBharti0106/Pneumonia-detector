import re
import base64
import numpy as np
from PIL import Image
from io import BytesIO


def base64_to_pil(img_base64):
    """Convert base64 image string (from browser) to PIL Image."""
    image_data = re.sub("^data:image/.+;base64,", "", img_base64)
    return Image.open(BytesIO(base64.b64decode(image_data)))


def np_to_base64(img_np):
    """Convert numpy RGB array to base64 PNG string."""
    img = Image.fromarray(img_np.astype("uint8"), "RGB")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode(
        "ascii"
    )
