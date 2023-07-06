import torch
import kornia.metrics as km


def peak_snr(
    img1: torch.Tensor, img2: torch.Tensor, bandwise: bool = False, band_dim: int = -3
) -> torch.Tensor:
    """
    Compute the peak signal-to-noise ratio between two images

    Parameters
    ----------
    img1 : torch.Tensor
    img2 : torch.Tensor
    bandwise: bool
        If true, do the calculation for each band
        default: False (calculation done for the full image)
    band_dim: int
        the dimension which has the image bands.
        default: -1 (i.e. last dim -> [h, w, bands])

    Returns
    -------
    snr : float
        peak signal-to-noise ration
    """
    if not bandwise:
        return km.psnr(img1, img2, max(img1.max(), img2.max()).item())
    else:
        out = torch.zeros(img1.shape[band_dim], device=img1.device)
        sl = [
            slice(None),
        ] * img1.ndim
        for b in range(img1.shape[band_dim]):
            sl[band_dim] = b
            out[b] = km.psnr(img1[sl], img2[sl], max(img1[sl].max(), img2[sl].max()).item())
        return out


def ssim(
    img1: torch.Tensor, img2: torch.Tensor, win_size=7
) -> torch.Tensor:
    """
    Compute the peak signal-to-noise ratio between two images

    Parameters
    ----------
    img1 : torch.Tensor
    img2 : torch.Tensor
    bandwise: bool
        If true, do the calculation for each band
        default: False (calculation done for the full image)

    Returns
    -------
    ssim : float
        mean structural similarity index
    """
    if len(img1.shape) == 3:
        img1.unsqueeze_(dim=0)
        img2.unsqueeze_(dim=0)
    return torch.mean(km.ssim(img1, img2, window_size=win_size, max_val=max(img1.max(), img2.max()).item()), dtype=torch.float32)


def sam(noise: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Measure spectral similarity using spectral angle mapper. Result is in radians.

    Parameters
    ----------
    noise: torch.Tensor
    reference: torch.Tensor

    Returns
    -------
    SAM: torch.Tensor
        spectral similarity in radians. If inputs are matrices, matrices are returned.
        i.e. a MxN is returned if a MxNxC is given
    """
    if len(noise.shape) == 1:
        # case 1: vectors -> easy
        numer = torch.dot(noise, reference)
        denom = torch.linalg.norm(noise) * torch.linalg.norm(reference)
    else:
        # case 2: matrices -> return a MxN if MxNxC is given
        numer = torch.sum(noise * reference, dim=-1)
        denom = torch.linalg.norm(noise, dim=-1) * torch.linalg.norm(reference, dim=-1)
    eps = torch.finfo(denom.dtype).eps
    return torch.arccos(numer / (denom + eps)).mean()


if __name__ == '__main__':
    x1 = torch.randn(1, 3, 31, 31)
    x2 = torch.randn(1, 3, 31, 31)
    print(sam(x1,x2))