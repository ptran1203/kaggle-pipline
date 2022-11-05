import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def put_text(image, text, save_path=None, expand_h_ratio=1.2):
    """
    Write text to image, need to use PIL Image since cv2 doesn't handle japanese text
    """
    if isinstance(image, torch.Tensor):
        image = image.numpy()
        image = image.transpose(1, 2, 0)
    
    if image.dtype.kind == 'f':
        # float data type
        # Rescale
        image = image * 255.0
        image = image.astype(np.uint8)

    if image.shape[-1] == 1:
        image = np.concatenate([image, image, image], axis=-1)

    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    height, width = image.shape[:2]
    bg_img = Image.new('RGB', (width, int(height * expand_h_ratio)), 'white')
    pil_img = Image.fromarray(image)
    bg_img.paste(pil_img, (0, 0))

    draw = ImageDraw.Draw(bg_img)
    textwidth, textheight = draw.textsize(text.encode("utf-8"))

    xmargin = 10
    ymargin = 10
    x = xmargin
    y = height + ymargin
    draw.text((x, y), text, (0, 0, 0))

    if save_path:
        # write image
        bg_img.save(save_path, optimize=True, quality=50)
    return bg_img
