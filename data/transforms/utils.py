import numpy as np
import torch

__all__ = [
    'minmax_normalize',
    'clip_outliers',
]


def minmax_normalize(image: torch.Tensor):
    """Normalization single image by min and max
    Args:
        img (torch.Tensor): Single image.

    Returns:
        img (torch.Tensor): Normalized image.
    """
    amax = image.max()
    amin = image.min()
    image -= amin
    image /= (amax - amin)
    return image

def clip_outliers(
        image: torch.Tensor, min: float = 0 , max: float = 1 
):
    """
    Clip pixel values outside [min,max] back into this interval。

    Args:
        image (torch.Tensor): image
        min (float, optional): min. Defaults to 0.
        max (float, optional): max. Defaults to 1.
    """
    torch.clip_(image, min, max)
    return image



if __name__ == '__main__':
    x = torch.arange(0, 9).reshape(1, 3, 3).float()
    print(id(x))
    print(id(minmax_normalize(x)))
    pass
