"""Metrics used in visualization."""

from collections.abc import Callable

import numpy  # noqa: ICN001
from PIL import Image

from .metrics_lpips import LPIPS_AVAILABLE


def _common_check(lhs: numpy.typing.NDArray, rhs: numpy.typing.NDArray) -> None:
  if lhs.shape != rhs.shape:
    message = (
      f"Images must have same shape to calculate metrics with each other. Got {lhs.shape} vs {rhs.shape}"
    )
    raise ValueError(message)

  if lhs.dtype != numpy.uint8 or rhs.dtype != numpy.uint8:
    message = (
      f"Only images with uint8 channel values are supported. Got {lhs.dtype.name} and {rhs.dtype.name}"
    )
    raise ValueError(message)


def _conv2d(x: numpy.typing.NDArray, kernel: numpy.typing.NDArray) -> numpy.typing.NDArray:
  pad_length = kernel.shape[0] // 2
  x = numpy.pad(x, pad_width=((pad_length, pad_length), (pad_length, pad_length), (0, 0)))
  x = numpy.lib.stride_tricks.as_strided(
    x,
    shape=(*kernel.shape, x.shape[-1], *tuple(numpy.subtract(x.shape[:-1], kernel.shape) + 1)),
    strides=(x.strides * 2)[:-1],
  )
  return numpy.einsum("ij...,ij...kl->kl...", kernel, x)


def _1d_gaussian_kernel(size: int, sigma: float) -> numpy.typing.NDArray:
  kernel = numpy.array(
    [numpy.exp(-((x - size // 2) ** 2) / (2.0 * sigma**2)) for x in range(size)],
  )
  return kernel / kernel.sum()


def _create_kernel(size: int) -> numpy.typing.NDArray:
  _1d_kernel = _1d_gaussian_kernel(size=size, sigma=1.5).reshape(-1, 1)
  return _1d_kernel @ _1d_kernel.T


def _ssim(
  lhs: numpy.typing.NDArray,
  rhs: numpy.typing.NDArray,
  window: numpy.typing.NDArray,
) -> float:
  mu1 = _conv2d(lhs, window)
  mu2 = _conv2d(rhs, window)

  mu1_sq = mu1**2
  mu2_sq = mu2**2
  mu1_mu2 = mu1 * mu2

  sigma1_sq = _conv2d(lhs * lhs, window) - mu1_sq
  sigma2_sq = _conv2d(rhs * rhs, window) - mu2_sq
  sigma12 = _conv2d(lhs * rhs, window) - mu1_mu2

  c1 = 0.01**2
  c2 = 0.03**2

  ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
    (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
  )

  return ssim_map.mean()


def psnr(lhs: Image.Image, rhs: Image.Image) -> float:
  """Calculate PSNR, peak signal-to-noise ratio."""
  lhs_array, rhs_array = numpy.array(lhs), numpy.array(rhs)
  _common_check(lhs_array, rhs_array)

  mse = ((lhs_array / 255 - rhs_array / 255) ** 2).mean()
  return -10 * numpy.log10(mse + 1e-10).item()


def ssim(lhs: Image.Image, rhs: Image.Image, *, window_size: int = 11) -> float:
  """Calculate SSIM, structural similarity index."""
  lhs_array, rhs_array = numpy.array(lhs), numpy.array(rhs)
  _common_check(lhs_array, rhs_array)

  window = _create_kernel(size=window_size)
  return _ssim(lhs=lhs_array / 255, rhs=rhs_array / 255, window=window)


if LPIPS_AVAILABLE:
  from torch import as_tensor

  from .metrics_lpips import calculate_lpips

  def lpips(lhs: Image.Image, rhs: Image.Image) -> float:
    """Calculate LPIPS, Learned Perceptual Image Patch Similarity."""
    lhs, rhs = lhs.convert("RGB"), rhs.convert("RGB")
    lhs_array, rhs_array = numpy.array(lhs), numpy.array(rhs)
    _common_check(lhs_array, rhs_array)

    lhs_tensor, rhs_tensor = (
      as_tensor(lhs_array / 255).permute(2, 0, 1),
      as_tensor(rhs_array / 255).permute(2, 0, 1),
    )
    return calculate_lpips(lhs=lhs_tensor, rhs=rhs_tensor).item()


class _MetricsAccessHelper:
  def __init__(self) -> None:
    self._metrics_map = {
      "PSNR": psnr,
      "SSIM": ssim,
    }
    if LPIPS_AVAILABLE:
      self._metrics_map["LPIPS"] = lpips

    for key, value in self._metrics_map.items():
      object.__setattr__(self, key, value)
      object.__setattr__(self, key.lower(), value)

  def __contains__(self, item: str) -> bool:
    if not isinstance(item, str):
      return False
    return item.upper() in self._metrics_map

  def __getitem__(self, key: str) -> Callable[[Image.Image, Image.Image], float]:
    if not isinstance(key, str):
      message = f"expected str as key, got {type(key)}"
      raise KeyError(message)
    if key not in self:
      message = f"cannot find metric named {key}"
      raise KeyError(message)
    return self._metrics_map[key.upper()]

  @property
  def methods(self) -> list[str]:
    return list(self._metrics_map.keys())


METRICS = _MetricsAccessHelper()
