"""Visualizers that converts numpy array into an image."""

import numpy
from PIL import Image
from PIL.ImageColor import getrgb
from PIL.ImageDraw import _Ink

_default_negligible_color = numpy.array([0.0, 0.0, 0.0])
_default_significant_color = numpy.array([1.0, 1.0, 0.11])
_default_neutral_color = numpy.array([0.0, 0.0, 0.0])
_default_positive_color = numpy.array([0.94, 0.33, 0.31])
_default_negative_color = numpy.array([0.31, 0.93, 0.94])


def _ink_to_rgb(color: _Ink) -> tuple[int, int, int]:
  """Convert _Ink to RGB tuple."""
  match color:
    case (int(), int(), int()):
      return color
    case float():
      return (int(255 * color), int(255 * color), int(255 * color))
    case int():
      return (color, color, color)
    case str():
      return getrgb(color)[:3]
    case _:
      message = f"Unexpected _Ink specification to be converted into RGB: {color}"
      raise ValueError(message)


def _standardize_color(color: _Ink | numpy.ndarray) -> numpy.ndarray:
  if isinstance(color, numpy.ndarray):
    if color.shape != (3,):
      message = f"shape of color must be (3, ), but got {color.shape}"
      raise ValueError(message)
    return color
  return numpy.array(_ink_to_rgb(color)) / 255


def to_difference_map(
  value: numpy.ndarray,
  *,
  negligible_color: _Ink | numpy.ndarray = _default_negligible_color,
  significant_color: _Ink | numpy.ndarray = _default_significant_color,
  ranges: tuple[float | None, float | None] = (None, None),
) -> Image.Image:
  """Visualize the array as unsigned absolute difference."""
  lower_color = _standardize_color(negligible_color)
  higher_color = _standardize_color(significant_color)

  minimum, maximum = ranges[0] or value.min(), ranges[1] or value.max()
  image_array = ((value - minimum) / (maximum - minimum) * (higher_color - lower_color) + lower_color) * 255
  return Image.fromarray(image_array.round().astype(numpy.uint8), mode="RGB")


def to_compare_map(
  value: numpy.ndarray,
  *,
  positive_color: _Ink | numpy.ndarray = _default_positive_color,
  negative_color: _Ink | numpy.ndarray = _default_negative_color,
  neutral_color: _Ink | numpy.ndarray = _default_neutral_color,
  ranges: tuple[float | None, float | None] = (None, None),
) -> Image.Image:
  """Visualize the array as signed difference.

  Non-negative result values will be shown by linear gradient from neutral_color (for 0) to positive_color.
  Non-positive result values will be shown by linear gradient from neutral_color (for 0) to negative_color.
  """
  positive = _standardize_color(positive_color)
  negative = _standardize_color(negative_color)
  neutral = _standardize_color(neutral_color)

  minimum, maximum = ranges[0] or value.min(), ranges[1] or value.max()
  positive_selector = (value > 0)[..., 0]
  negative_selector = (value < 0)[..., 0]

  result_array = numpy.broadcast_to(neutral, (*value.shape[:2], 3)).copy()

  result_array[positive_selector] = (value[positive_selector] / maximum) * (positive - neutral) + neutral
  result_array[negative_selector] = (value[negative_selector] / minimum) * (negative - neutral) + neutral
  result_array = (result_array * 255).round().astype(numpy.uint8)
  return Image.fromarray(result_array, mode="RGB")
