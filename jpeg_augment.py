import io
from PIL import Image

def compress_img(image: Image, qf: int) -> Image:
    outputIoStream = io.BytesIO()
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)