def tight_crop(pil_img):
    return pil_img.crop(pil_img.getbbox())
