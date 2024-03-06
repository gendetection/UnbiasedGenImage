import sys
import torchjpeg.codec
from torchjpeg.quantization.ijg import get_coefficients_for_qualities
import torch
from torch.nn.functional import mse_loss

def compute_quality(path):
    """
    compute the qf factor of a JPEG image given the path
    """
    dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(path)
    quantization = quantization[0].float()  # Only compute quality of the Y channel (they match anyway)
    d_best = sys.maxsize
    q_best = 0
    for i in range(100, -1, -1):
        q = torch.tensor([i]).float()
        qm = get_coefficients_for_qualities(q).view(8, 8)

        d = mse_loss(qm, quantization)

        if d <= d_best:
            d_best = d
            q_best = i
        else:
            break

    return q_best
