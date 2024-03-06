import random
import io
import torchvision.transforms as T
from PIL import Image

def compress_img(image, qf):

    outputIoStream = io.BytesIO()
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)


def get_transform(args, train_df=None):
    # input and output have to be PIL Image

    # make value count of training compression-rate
    if args.sample_qf_ai:
        assert train_df is not None
        qf_distribution = train_df[train_df["generator"] == "nature"].compression_rate.value_counts()
        qf_distribution = qf_distribution / qf_distribution.sum()


    def t(img, label):
        if label == "nature":
            if args.jpeg_qf is not None and args.compress_natural:
                img = compress_img(img, args.jpeg_qf)
        else:
            if args.jpeg_qf is not None:
                img = compress_img(img, args.jpeg_qf)
            elif args.sample_qf_ai:
                qf = int(random.choices(qf_distribution.index, qf_distribution, k=1)[0])
                img = compress_img(img, qf)

        if args.resize is not None:
            img = img.resize((args.resize, args.resize))

        if args.cropsize is not None:
            if args.cropmethod.lower() == "center": # DEFAULT
                img = T.CenterCrop(args.cropsize)(img)
            elif args.cropmethod.lower() == "random":
                img = T.RandomCrop(args.cropsize)(img)
            else:
                raise NotImplementedError("cropmethod must be one of [center, random]")

        return img


    return t