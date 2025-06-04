"""Tools for visualize differences between two image sets."""

import math
from argparse import ArgumentParser, Namespace
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import mean
from typing import NamedTuple

import numpy  # noqa: ICN001
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageColor import getrgb
from PIL.ImageDraw import _Ink

from .metrics import METRICS
from .utils import find_shape, fit_font, place_lines, tile_images

_TextboxBackground = ("#546E7A", 0.4)
_TextboxForeground = "white"


def _select_images(sources: list[Path]) -> list[str]:
  lists = [sorted([path.name for path in source.iterdir()]) for source in sources]
  return [v for v in lists[0] if v in set(lists[0]).intersection(*lists[1:])]


def _place_metrics(
  image: Image.Image,
  name: str,
  metrics: list[tuple[str, float]],
  font: ImageFont.FreeTypeFont,
) -> None:
  lines = [name, *[f"{name} = {str(value)[:7]}" for name, value in metrics]]
  place_lines(
    image=image,
    lines=lines,
    font=font,
    line_distance=8,
    target_width=min(int(image.width * 0.4), 240),
    background=_TextboxBackground,
  )


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


def render_difference_image(
  lhs: Image.Image,
  rhs: Image.Image,
  *,
  lower_difference: _Ink = "#000000",
  higher_difference: _Ink = "#ffff1c",
  global_range: tuple[float, float] | None = None,
) -> tuple[Image.Image, numpy.typing.NDArray]:
  """Render a difference map between two images.

  The absolute difference between pixels of these two images is calculated and averaged over all channels.
  The greater the difference is, the more close to higher_difference the corresponding pixel on rendered
   image will be.
  Both the rendered image and the raw value of difference is returned ti support various usages.
  """
  lower_color = numpy.array(_ink_to_rgb(lower_difference)) / 255
  higher_color = numpy.array(_ink_to_rgb(higher_difference)) / 255

  lhs_array, rhs_array = numpy.array(lhs) / 255, numpy.array(rhs) / 255
  difference = numpy.abs(lhs_array - rhs_array).mean(axis=-1, keepdims=True)
  if global_range is not None:
    minimum, maximum = global_range
  else:
    minimum, maximum = difference.min(), difference.max()
  image_array = (
    (difference - minimum) / (maximum - minimum) * (higher_color - lower_color) + lower_color
  ) * 255
  return Image.fromarray(image_array.round().astype(numpy.uint8), mode="RGB"), difference


def render_compare_map(  # noqa: PLR0913
  lhs: numpy.typing.NDArray,
  rhs: numpy.typing.NDArray,
  *,
  positive_color: _Ink = "#ef5350",
  negative_color: _Ink = "#50ecef",
  neutral_color: _Ink = "#000000",
  global_range: tuple[float, float] | None = None,
) -> Image.Image:
  """Visualize the signed difference.

  Value of rhs is subtracted from value of lhs, each resulting value will be represented by a single pixel
   on the image rendered.
  Non-negative result values will be shown by linear gradient from neutral_color (for 0) to positive_color.
  Non-positive result values will be shown by linear gradient from neutral_color (for 0) to negative_color.
  """
  positive = numpy.array(_ink_to_rgb(positive_color)) / 255
  negative = numpy.array(_ink_to_rgb(negative_color)) / 255
  neutral = numpy.array(_ink_to_rgb(neutral_color)) / 255

  difference = lhs - rhs
  if global_range is not None:
    minimum, maximum = global_range
  else:
    minimum, maximum = difference.min(), difference.max()
  positive_selector = (difference > 0)[..., 0]
  negative_selector = (difference < 0)[..., 0]

  result_array = numpy.broadcast_to(neutral, (*difference.shape[:2], 3)).copy()

  result_array[positive_selector] = (difference[positive_selector] / maximum) * (positive - neutral) + neutral
  result_array[negative_selector] = (difference[negative_selector] / minimum) * (negative - neutral) + neutral
  result_array = (result_array * 255).round().astype(numpy.uint8)
  return Image.fromarray(result_array, mode="RGB")


def render_single2(
  lhs: tuple[Image.Image, str],
  rhs: tuple[Image.Image, str],
  font: ImageFont.FreeTypeFont,
  metrics: Iterable[str],
  global_range: tuple[float, float] | None = None,
) -> tuple[Image.Image, list[tuple[str, float]]]:
  """Visualize a single pair of images.

  Both image should have size w x h, and this will result in an image whose size is 2w x 2h.
  If metrics are specified, they are measured against each other, marked on the visualized image and returned.
  """
  lhs_image, lhs_name = lhs
  rhs_image, rhs_name = rhs

  calculated_metrics = [(metric, METRICS[metric](lhs_image, rhs_image)) for metric in metrics]
  difference_map = render_difference_image(lhs=lhs_image, rhs=rhs_image, global_range=global_range)[0]
  local_difference_map = render_difference_image(lhs=lhs_image, rhs=rhs_image, global_range=None)[0]

  result = Image.new(mode=lhs_image.mode, size=(lhs_image.width * 2, lhs_image.height * 2))
  _place_metrics(image=lhs_image, name=lhs_name, metrics=calculated_metrics, font=font)
  _place_metrics(image=rhs_image, name=rhs_name, metrics=calculated_metrics, font=font)

  place_lines(
    image=difference_map,
    lines=[f"{lhs[1]} vs {rhs[1]}"],
    font=font,
    target_width=min(int(difference_map.width * 0.4), 240),
    background=_TextboxBackground,
  )

  result.paste(difference_map, box=(0, 0))
  result.paste(lhs_image, box=(0, lhs_image.height))
  result.paste(rhs_image, box=(lhs_image.width, 0))
  result.paste(local_difference_map, box=(lhs_image.width, lhs_image.height))
  return result, calculated_metrics


def render_single3(  # noqa: PLR0913
  lhs: tuple[Image.Image, str],
  rhs: tuple[Image.Image, str],
  reference: Image.Image,
  font: ImageFont.FreeTypeFont,
  metrics: Iterable[str],
  global_ranges: list[tuple[float, float]] | None = None,
) -> tuple[Image.Image, list[tuple[str, float]], list[tuple[str, float]]]:
  """Visualize a single pair of images with a reference.

  Both image should have size w x h, and this will result in an image whose size is 3w x 3h.
  If metrics are specified, they are measured for both lhs and rhs image against the reference image,
   marked on the visualized image and returned separately for each image.
  """
  lhs_image, lhs_name = lhs
  rhs_image, rhs_name = rhs

  calculated_metrics = [
    (metric, METRICS[metric](lhs_image, reference), METRICS[metric](rhs_image, reference))
    for metric in metrics
  ]
  lhs_metrics = [(name, value) for name, value, _ in calculated_metrics]
  rhs_metrics = [(name, value) for name, _, value in calculated_metrics]

  lhs_difference_map, lhs_difference = render_difference_image(
    lhs=lhs_image,
    rhs=reference,
    global_range=None if global_ranges is None else global_ranges[0],
  )
  rhs_difference_map, rhs_difference = render_difference_image(
    lhs=rhs_image,
    rhs=reference,
    global_range=None if global_ranges is None else global_ranges[0],
  )
  compare_map = render_compare_map(
    lhs=lhs_difference,
    rhs=rhs_difference,
    global_range=None if global_ranges is None else global_ranges[1],
  )
  local_compare_map = render_compare_map(
    lhs=lhs_difference,
    rhs=rhs_difference,
    global_range=None,
  )
  local_lhs_difference_map = render_difference_image(lhs=lhs_image, rhs=reference, global_range=None)[0]
  local_rhs_difference_map = render_difference_image(lhs=rhs_image, rhs=reference, global_range=None)[0]

  result = Image.new(mode=lhs_image.mode, size=(lhs_image.width * 3, lhs_image.height * 3))
  _place_metrics(image=lhs_image, name=lhs_name, metrics=lhs_metrics, font=font)
  _place_metrics(image=rhs_image, name=rhs_name, metrics=rhs_metrics, font=font)

  place_lines(
    image=compare_map,
    lines=[f"{lhs[1]} vs {rhs[1]}"],
    font=font,
    target_width=min(int(compare_map.width * 0.4), 240),
    background=_TextboxBackground,
  )
  place_lines(
    image=lhs_difference_map,
    lines=[f"{lhs[1]} vs GT"],
    font=font,
    target_width=min(int(lhs_difference_map.width * 0.4), 240),
    background=_TextboxBackground,
  )
  place_lines(
    image=rhs_difference_map,
    lines=[f"{rhs[1]} vs GT"],
    font=font,
    target_width=min(int(rhs_difference_map.width * 0.4), 240),
    background=_TextboxBackground,
  )

  result.paste(lhs_image, box=(0, lhs_image.height))
  result.paste(rhs_image, box=(lhs_image.width, 0))
  result.paste(reference, box=(lhs_image.width, lhs_image.height))

  result.paste(compare_map, box=(0, 0))
  result.paste(lhs_difference_map, box=(0, 2 * lhs_image.height))
  result.paste(rhs_difference_map, box=(2 * lhs_image.width, 0))
  result.paste(local_lhs_difference_map, box=(lhs_image.width, 2 * lhs_image.height))
  result.paste(local_rhs_difference_map, box=(2 * lhs_image.width, lhs_image.height))
  result.paste(local_compare_map, box=(2 * lhs_image.width, 2 * lhs_image.height))
  return result, lhs_metrics, rhs_metrics


def render_single(  # noqa: PLR0913
  lhs: tuple[Image.Image, str],
  rhs: tuple[Image.Image, str] | None,
  reference: Image.Image | None,
  font: ImageFont.FreeTypeFont,
  metrics: Iterable[str],
  global_ranges: list[tuple[float, float]] | None = None,
) -> tuple[Image.Image, list[list[tuple[str, float]]]]:
  """Visualize a single pair of images.

  This will choose render_single2 or render_single3 automatically
  """
  if rhs is None and reference is not None:
    image, calculated = render_single2(
      lhs=lhs,
      rhs=(reference, "GT"),
      font=font,
      metrics=metrics,
      global_range=None if global_ranges is None else global_ranges[0],
    )
    return (image, [calculated])
  if rhs is not None and reference is None:
    image, calculated = render_single2(
      lhs=lhs,
      rhs=rhs,
      font=font,
      metrics=metrics,
      global_range=None if global_ranges is None else global_ranges[0],
    )
    return (image, [calculated])
  if rhs is not None and reference is not None:
    image, lhs_calculated, rhs_calculated = render_single3(
      lhs=lhs,
      rhs=rhs,
      reference=reference,
      font=font,
      metrics=metrics,
      global_ranges=global_ranges,
    )
    return (image, [lhs_calculated, rhs_calculated])
  raise ValueError


def _place_description(  # noqa: PLR0913
  image: Image.Image,
  description: str,
  font: ImageFont.FreeTypeFont,
  *,
  line_distance: int = 0,
  foreground: _Ink = "white",
  background: tuple[_Ink, float] | None = None,
) -> None:
  drawer = ImageDraw.Draw(image)
  lines = description.split("\n")

  font, width, height = fit_font(
    texts=lines,
    font=font,
    width=int(image.width * 0.8),
    height=int((image.height * 0.6 - (len(lines) - 1) * line_distance) / len(lines)),
  )
  total_height = height * len(lines) + (len(lines) - 1) * line_distance

  if background is not None:
    background_box = Image.new(
      mode=image.mode,
      size=(width, total_height),
      color=background[0],
    )
    position_box = (
      (image.width - width) // 2,
      (image.height - total_height) // 2,
      (image.width + width) // 2,
      (image.height + total_height) // 2,
    )
    image.paste(
      Image.blend(
        image.crop(position_box),
        background_box,
        alpha=background[1],
      ),
      box=position_box,
    )

  start_height = (image.height - total_height) // 2
  for line in lines:
    drawer.text(xy=(image.width // 2, start_height), text=line, font=font, anchor="mt", fill=foreground)
    start_height += height + line_distance


def _render_cover2(
  image_size: tuple[int, int],
  lhs: str,
  rhs: str,
  font: ImageFont.FreeTypeFont,
) -> Image.Image:
  width, height = image_size
  cover = Image.new(size=(width * 2, height * 2), mode="RGB", color="black")
  difference_sample = render_difference_image(
    lhs=Image.fromarray(numpy.zeros(shape=(height, width, 3), dtype=numpy.uint8), mode="RGB"),
    rhs=Image.fromarray(
      numpy.linspace(start=0, stop=255, num=width, dtype=numpy.uint8)[None, :, None]
      .repeat(height, axis=0)
      .repeat(3, axis=-1),
      mode="RGB",
    ),
  )[0]
  lb = Image.new(size=(width, height), mode="RGB", color="black")
  rt = Image.new(size=(width, height), mode="RGB", color="black")
  rb = difference_sample.copy()
  _place_description(
    image=difference_sample,
    description=f"difference between {lhs} and {rhs}\n<-- lower     \n    higher -->",
    font=font,
    line_distance=12,
    background=_TextboxBackground,
  )
  _place_description(
    image=lb,
    description=lhs,
    font=font,
    line_distance=12,
    background=_TextboxBackground,
  )
  _place_description(
    image=rt,
    description=rhs,
    font=font,
    line_distance=12,
    background=_TextboxBackground,
  )
  _place_description(
    image=rb,
    description=f"difference between {lhs} and {rhs}\n<-- lower     \n    higher -->\n[unaligned]",
    font=font,
    line_distance=12,
    background=_TextboxBackground,
  )

  cover.paste(difference_sample)
  cover.paste(lb, box=(0, height))
  cover.paste(rt, box=(width, 0))
  cover.paste(rb, box=(width, height))
  return cover


def _render_cover3(
  image_size: tuple[int, int],
  lhs: str,
  rhs: str,
  font: ImageFont.FreeTypeFont,
) -> Image.Image:
  width, height = image_size

  compare_sample = render_compare_map(
    lhs=numpy.linspace(start=0.0, stop=1.0, num=width)[None, :, None]
    .repeat(height, axis=0)
    .repeat(3, axis=-1),
    rhs=numpy.full(shape=(height, width, 3), fill_value=0.5),
  )
  difference_sample = render_difference_image(
    lhs=Image.fromarray(numpy.zeros(shape=(height, width, 3), dtype=numpy.uint8), mode="RGB"),
    rhs=Image.fromarray(
      numpy.linspace(start=0, stop=255, num=width, dtype=numpy.uint8)[None, :, None]
      .repeat(height, axis=0)
      .repeat(3, axis=-1),
      mode="RGB",
    ),
  )[0]
  empty_sample = Image.new(size=(width, height), mode="RGB", color="black")

  unaligned_compare_sample = compare_sample.copy()

  _place_description(
    image=compare_sample,
    description=f"{lhs} vs {rhs}\n<-- {lhs} closer to GT     \n    {rhs} closer to GT -->",
    font=font,
    line_distance=12,
    background=_TextboxBackground,
  )
  _place_description(
    image=unaligned_compare_sample,
    description=f"{lhs} vs {rhs}\n<-- {lhs} closer to GT     \n    {rhs} closer to GT -->\n[unaligned]",
    font=font,
    line_distance=12,
    background=_TextboxBackground,
  )

  rhs_sample = empty_sample.copy()
  _place_description(
    image=rhs_sample,
    description=rhs,
    font=font,
    line_distance=12,
    background=_TextboxBackground,
  )

  rhs_difference = difference_sample.copy()
  _place_description(
    image=rhs_difference,
    description=f"difference between {rhs} and GT\n<-- lower     \n    higher -->",
    font=font,
    line_distance=12,
    background=_TextboxBackground,
  )

  lhs_sample = empty_sample.copy()
  _place_description(
    image=lhs_sample,
    description=rhs,
    font=font,
    line_distance=12,
    background=_TextboxBackground,
  )

  reference_sample = empty_sample.copy()
  _place_description(
    image=reference_sample,
    description="GT",
    font=font,
    line_distance=12,
    background=_TextboxBackground,
  )

  rhs_unaligned_difference_sample = difference_sample.copy()
  _place_description(
    image=rhs_unaligned_difference_sample,
    description=f"difference between {rhs} and GT\n<-- lower     \n    higher -->\n[unaligned]",
    font=font,
    line_distance=12,
    background=_TextboxBackground,
  )

  lhs_difference = difference_sample.copy()
  _place_description(
    image=lhs_difference,
    description=f"difference between {lhs} and GT\n<-- lower     \n    higher -->",
    font=font,
    line_distance=12,
    background=_TextboxBackground,
  )

  lhs_unaligned_difference_sample = difference_sample.copy()
  _place_description(
    image=lhs_unaligned_difference_sample,
    description=f"difference between {lhs} and GT\n<-- lower     \n    higher -->\n[unaligned]",
    font=font,
    line_distance=12,
    background=_TextboxBackground,
  )

  return tile_images(
    images=(
      compare_sample,
      rhs_sample,
      rhs_difference,
      lhs_sample,
      reference_sample,
      rhs_unaligned_difference_sample,
      lhs_difference,
      lhs_unaligned_difference_sample,
      unaligned_compare_sample,
    ),
    rows=3,
    columns=3,
  )


def _render_cover(
  image_size: tuple[int, int],
  names: Sequence[str],
  font: ImageFont.FreeTypeFont,
) -> Image.Image:
  if len(names) == 2:  # noqa: PLR2004
    return _render_cover2(image_size=image_size, lhs=names[0], rhs=names[1], font=font)
  return _render_cover3(image_size=image_size, lhs=names[0], rhs=names[1], font=font)


def _render_bar(
  drawer: ImageDraw.ImageDraw,
  xy: tuple[int, int],
  size: tuple[int, int],
  color: _Ink = "#80cbc3",
) -> None:
  drawer.rectangle(
    xy=(xy[0], xy[1], xy[0] + size[0], xy[1] + size[1]),
    fill=color,
    outline=None,
    width=0,
  )


def _render_difference_bar(  # noqa: PLR0913
  drawer: ImageDraw.ImageDraw,
  xy: tuple[int, int],
  lhs_size: tuple[int, int],
  rhs_size: tuple[int, int],
  *,
  base_color: _Ink = "#80cbc3",
  advantage_color: _Ink = "#8088cb",
  disadvantage_color: _Ink = "#c480cb",
) -> None:
  # base part: within range of both lhs and rhs
  _render_bar(
    drawer=drawer,
    xy=xy,
    size=(min(lhs_size[0], rhs_size[0]), min(lhs_size[1], rhs_size[1])),
    color=base_color,
  )

  # extra parts
  size_with_larger_width = lhs_size if lhs_size[0] > rhs_size[0] else rhs_size
  size_with_smaller_width = lhs_size if lhs_size[0] <= rhs_size[0] else rhs_size
  size_with_larger_height = lhs_size if lhs_size[1] > rhs_size[1] else rhs_size
  size_with_smaller_height = lhs_size if lhs_size[1] <= rhs_size[1] else rhs_size

  if size_with_larger_width[0] != size_with_smaller_width[0]:
    _render_bar(
      drawer=drawer,
      xy=(xy[0] + size_with_smaller_width[0] + 1, xy[1]),
      size=(size_with_larger_width[0] - size_with_smaller_width[0], size_with_larger_width[1]),
      color=advantage_color if lhs_size[0] >= rhs_size[0] else disadvantage_color,
    )

  if size_with_larger_height[1] != size_with_smaller_height[1]:
    _render_bar(
      drawer=drawer,
      xy=(xy[0], xy[1] + size_with_smaller_height[1] + 1),
      size=(size_with_larger_height[0], size_with_larger_height[1] - size_with_smaller_height[1]),
      color=advantage_color if lhs_size[1] >= rhs_size[1] else disadvantage_color,
    )


class _MetricsSummaryConstants(NamedTuple):
  padding: tuple[int, int, int, int] = 64, 32, 48, 32  # top, right, bottom, left
  title_line_spacing: int = 24
  title_bar_height: int = 36
  title_bar_width: float = 0.7
  title_spacing: int = 48
  title_height_limit: float = 0.2
  title_width_multiplier: float = 0.5
  item_margin: tuple[int, int] = 16, 16  # vertical, horizontal
  minimum_bar_ratio: float = 0.4
  metrics_spacing: int = 8
  minimum_per_character_width: float = 12


_metrics_summary_constants = _MetricsSummaryConstants()


class _SummarySizing(NamedTuple):
  image_size: tuple[int, int]
  title_font: tuple[ImageFont.FreeTypeFont, tuple[int, int]]
  item_font: tuple[ImageFont.FreeTypeFont, tuple[int, int]]
  item_size: tuple[int, int]
  shape: tuple[int, int]


def _decide_summary_size(  # noqa: PLR0913
  base_size: tuple[int, int],
  title_lines: Sequence[str],
  title_bars: int,
  item_lines: Sequence[str],
  items: int,
  font: ImageFont.FreeTypeFont,
) -> _SummarySizing:
  base_width, base_height = base_size
  padding_top, padding_right, padding_bottom, padding_left = _metrics_summary_constants.padding

  title_font, title_width, title_height = fit_font(
    texts=title_lines,
    font=font,
    width=int(
      (base_width - padding_left - padding_right) * _metrics_summary_constants.title_width_multiplier,
    ),
    height=int(base_height * _metrics_summary_constants.title_height_limit),
  )

  final_width, final_height = base_width, base_height
  columns, rows = find_shape(items)
  margin_vertical, margin_horizontal = _metrics_summary_constants.item_margin

  while True:
    width_available = final_width - padding_left - padding_right
    height_available = (
      final_height
      - padding_top
      - padding_bottom
      - title_height * len(title_lines)
      - _metrics_summary_constants.title_line_spacing * (len(title_lines) - 1)
      - _metrics_summary_constants.title_spacing
      - _metrics_summary_constants.title_bar_height * title_bars
    )

    item_width = (width_available - (columns - 1) * margin_horizontal) // columns
    item_height = (height_available - (rows - 1) * margin_vertical) // rows
    line_per_item = len(item_lines) // items

    item_font, item_text_width, item_text_height = fit_font(
      texts=item_lines,
      font=font,
      height=(item_height - _metrics_summary_constants.metrics_spacing * (line_per_item - 1))
      // (line_per_item * 3 - 2),
      width=item_width,
    )

    if (
      item_text_width / max([len(line) for line in item_lines])
      < _metrics_summary_constants.minimum_per_character_width
    ):
      final_width += base_width
      final_height += base_height
    else:
      break

  return _SummarySizing(
    image_size=(final_width, final_height),
    title_font=(title_font, (title_width, title_height)),
    item_font=(item_font, (item_text_width, item_text_height)),
    item_size=(item_width, item_height),
    shape=(columns, rows),
  )


def _render_metrics_summary2(
  image_size: tuple[int, int],
  metrics: Sequence[tuple[str, Sequence[tuple[str, float]]]],
  font: ImageFont.FreeTypeFont,
) -> Image.Image:
  metric_names = [name for name, _ in metrics[0][1]]
  ranges = {}
  for name in metric_names:
    all_values = [value for _, values in metrics for metric_name, value in values if metric_name == name]
    ranges[name] = (
      min(all_values),
      max(all_values),
      mean(all_values),
    )
    ranges[name] = (*ranges[name], ranges[name][1] - ranges[name][0])

  title_line = "    ".join(f"{name}: {str(value[2])[:7]}" for name, value in ranges.items())
  item_lines = [
    *[f"{name}: {str(value)[:7]}" for _, values in metrics for name, value in values],
    *[name for name, _ in metrics],
  ]
  sizing = _decide_summary_size(
    base_size=(image_size[0] * 2, image_size[1] * 2),
    title_lines=(title_line,),
    title_bars=0,
    item_lines=item_lines,
    items=len(metrics),
    font=font,
  )

  width, height = sizing.image_size
  title_font, (_, title_height) = sizing.title_font
  item_font, (_, item_text_height) = sizing.item_font
  item_width, item_height = sizing.item_size
  columns, rows = sizing.shape

  summary = Image.new(size=(width, height), mode="RGB", color="black")
  summary_drawer = ImageDraw.Draw(summary)

  padding_top, padding_right, padding_bottom, padding_left = _metrics_summary_constants.padding
  margin_vertical, margin_horizontal = _metrics_summary_constants.item_margin

  start_height = padding_top
  start_width = padding_left

  summary_drawer.text(
    xy=((width - padding_left - padding_right) // 2 + padding_left, start_height),
    text=title_line,
    font=title_font,
    anchor="mt",
    fill="white",
  )
  start_height += title_height + _metrics_summary_constants.title_spacing

  for i in range(rows):
    start_width = padding_left
    for j in range(columns):
      index = i * columns + j
      if index >= len(metrics):
        break
      name, items = metrics[index]
      current_height = start_height
      summary_drawer.text(
        xy=(start_width, current_height),
        text=name,
        font=item_font,
        anchor="lt",
        fill="white",
      )
      current_height += item_text_height + _metrics_summary_constants.metrics_spacing
      for metric_name, metric_value in items:
        summary_drawer.text(
          xy=(start_width, current_height),
          text=f"{metric_name}: {str(metric_value)[:7]}",
          font=item_font,
          anchor="lt",
          fill="white",
        )
        current_height += item_text_height
        _render_bar(
          drawer=summary_drawer,
          xy=(start_width, current_height),
          size=(
            int(
              (
                (1 - _metrics_summary_constants.minimum_bar_ratio)
                * ((metric_value - ranges[metric_name][0]) / ranges[metric_name][3])
                + _metrics_summary_constants.minimum_bar_ratio
              )
              * item_width,
            ),
            item_text_height * 2,
          ),
        )
        current_height += item_text_height * 2 + _metrics_summary_constants.metrics_spacing
      start_width += item_width + margin_horizontal
    start_height += item_height + margin_vertical
  return summary


def _get_summary_line(name: str, lhs_value: float, rhs_value: float) -> str:
  base = f"{name}: {str(lhs_value)[:7]} vs {str(rhs_value)[:7]} "
  if lhs_value > rhs_value:
    return f"{base} (+{(lhs_value - rhs_value) / rhs_value * 100:.2f}%)"
  if lhs_value < rhs_value:
    return f"{base} (-{(rhs_value - lhs_value) / rhs_value * 100:.2f}%)"
  return f"{base} (+/- 0.00%)"


def _get_bar_length(
  ranges: tuple[float, float, float, float],
  value: float,
  total_width: int,
) -> int:
  return int(
    (
      (value - ranges[0]) / ranges[3] * (1 - _metrics_summary_constants.minimum_bar_ratio)
      + _metrics_summary_constants.minimum_bar_ratio
    )
    * total_width,
  )


def _render_metrics_summary3(
  image_size: tuple[int, int],
  metrics: Sequence[Sequence[tuple[str, Sequence[tuple[str, float]]]]],
  font: ImageFont.FreeTypeFont,
) -> Image.Image:
  metric_names = [name for name, _ in metrics[0][0][1]]
  ranges = {}
  for name in metric_names:
    lhs_values = [value for _, values in metrics[0] for metric_name, value in values if metric_name == name]
    rhs_values = [value for _, values in metrics[1] for metric_name, value in values if metric_name == name]
    ranges[name] = (
      min(*lhs_values, *rhs_values),
      max(*lhs_values, *rhs_values),
      (mean(lhs_values), mean(rhs_values)),
    )
    ranges[name] = (*ranges[name], ranges[name][1] - ranges[name][0])

  title_lines = [
    _get_summary_line(name=name, lhs_value=ranges[name][2][0], rhs_value=ranges[name][2][1])
    for name in metric_names
  ]
  item_lines = []
  for lhs, rhs in zip(metrics[0], metrics[1], strict=True):
    item_lines.append(lhs[0])
    for lhs_metrics, rhs_metrics in zip(lhs[1], rhs[1], strict=True):
      item_lines.append(
        _get_summary_line(name=lhs_metrics[0], lhs_value=lhs_metrics[1], rhs_value=rhs_metrics[1]),
      )

  sizing = _decide_summary_size(
    base_size=(image_size[0] * 3, image_size[1] * 3),
    title_lines=title_lines,
    title_bars=len(metric_names),
    item_lines=item_lines,
    items=len(metrics[0]),
    font=font,
  )

  width, height = sizing.image_size
  title_font, (_, title_height) = sizing.title_font
  item_font, (_, item_text_height) = sizing.item_font
  item_width, item_height = sizing.item_size
  columns, rows = sizing.shape

  summary = Image.new(size=(width, height), mode="RGB", color="black")
  summary_drawer = ImageDraw.Draw(summary)

  padding_top, padding_right, padding_bottom, padding_left = _metrics_summary_constants.padding
  margin_vertical, margin_horizontal = _metrics_summary_constants.item_margin

  start_height = padding_top
  start_width = padding_left

  for index, name in enumerate(metric_names):
    middle_width = (width - padding_left - padding_right) // 2 + padding_left

    if index != 0:
      start_height += _metrics_summary_constants.title_line_spacing

    summary_drawer.text(
      xy=(middle_width, start_height),
      text=title_lines[index],
      font=title_font,
      anchor="mt",
      fill="white",
    )
    start_height += title_height

    total_bar_length = int(
      _metrics_summary_constants.title_bar_width * (image_size[0] * 3 - padding_left - padding_right),
    )
    _render_difference_bar(
      drawer=summary_drawer,
      xy=(middle_width - total_bar_length // 2, start_height),
      lhs_size=(
        _get_bar_length(
          ranges=ranges[name],
          value=ranges[name][2][0],
          total_width=total_bar_length,
        ),
        _metrics_summary_constants.title_bar_height,
      ),
      rhs_size=(
        _get_bar_length(
          ranges=ranges[name],
          value=ranges[name][2][1],
          total_width=total_bar_length,
        ),
        _metrics_summary_constants.title_bar_height,
      ),
    )
    start_height += _metrics_summary_constants.title_bar_height

  start_height += _metrics_summary_constants.title_spacing

  for i in range(rows):
    start_width = padding_left
    for j in range(columns):
      index = i * columns + j
      if index >= len(metrics[0]):
        break
      name, lhs_items = metrics[0][index]
      _, rhs_items = metrics[1][index]
      items = [(name, lhs, rhs) for (name, lhs), (_, rhs) in zip(lhs_items, rhs_items, strict=True)]
      current_height = start_height
      summary_drawer.text(
        xy=(start_width, current_height),
        text=name,
        font=item_font,
        anchor="lt",
        fill="white",
      )
      current_height += item_text_height + _metrics_summary_constants.metrics_spacing
      for metric_name, lhs_value, rhs_value in items:
        summary_drawer.text(
          xy=(start_width, current_height),
          text=_get_summary_line(name=metric_name, lhs_value=lhs_value, rhs_value=rhs_value),
          font=item_font,
          anchor="lt",
          fill="white",
        )
        current_height += item_text_height
        _render_difference_bar(
          drawer=summary_drawer,
          xy=(start_width, current_height),
          lhs_size=(
            _get_bar_length(
              ranges=ranges[metric_name],
              value=lhs_value,
              total_width=item_width,
            ),
            item_text_height * 2,
          ),
          rhs_size=(
            _get_bar_length(
              ranges=ranges[metric_name],
              value=rhs_value,
              total_width=item_width,
            ),
            item_text_height * 2,
          ),
        )
        current_height += item_text_height * 2 + _metrics_summary_constants.metrics_spacing
      start_width += item_width + margin_horizontal
    start_height += item_height + margin_vertical
  return summary


def _render_metrics_summary(
  image_size: tuple[int, int],
  metrics: Sequence[Sequence[tuple[str, Sequence[tuple[str, float]]]]],
  font: ImageFont.FreeTypeFont,
) -> Image.Image:
  if len(metrics) == 1:
    return _render_metrics_summary2(image_size=image_size, metrics=metrics[0], font=font)
  return _render_metrics_summary3(image_size=image_size, metrics=metrics, font=font)


def _get_arguments() -> Namespace:  # noqa: C901
  argument_parser = ArgumentParser(
    description="A tool to visualize the difference between two corresponding image sets. "
    "Specify two or three (one of which is used as reference, or ground truth) image sets "
    "to place them side-by-side with metrics and a heatmap visualizing difference alongside.\n"
    "This functionality can be typically useful when analyzing and/or comparing performance "
    "of image compressing/reconstructing method(s). Images from different set will be compared "
    "if they share the same name.",
  )
  argument_parser.add_argument(
    "-r",
    "--reference",
    help="Path to image set to be used as reference, usually the ground truth images.",
    type=Path,
  )
  argument_parser.add_argument(
    "-o",
    "--output",
    help="Path to store visualized result. If the path specified does not exist on the filesystem, "
    "a directory will be created, otherwise it must refer to an empty directory.\n"
    "Defaults to <first set name>-vs-<second set name> under current working directory",
    type=Path,
  )
  argument_parser.add_argument(
    "-m",
    "--metric",
    help="Metrics to calculate and display.\n"
    "If a reference set is supplied, all other image sets will measure their metrics against it. "
    "Otherwise, they will measure against each other.",
    nargs="+",
    choices=METRICS.methods,
    type=str,
  )
  argument_parser.add_argument(
    "-n",
    "--name",
    help="Names of the image sets specified. This will be used to denote the visualized image. "
    "By default, the last component of path to the image set will be used.",
    nargs="+",
    type=str,
  )
  argument_parser.add_argument(
    "-f",
    "--font",
    help="Font to use when rendering text on the image. Can be specified by the name, filename or path."
    "Locating and loading the font is handled by the ImageDraw module in Pillow library, "
    "TrueType font is recommended and expected.",
    type=str,
  )
  argument_parser.add_argument("images", help="Image set(s) to visualize.", nargs="+", type=Path)
  arguments = argument_parser.parse_args()

  if len(arguments.images) > 2:  # noqa: PLR2004 (It is more than obvious why 2 here)
    message = "Too many image sets specified: no more than two with an extra one as the reference is allowed"
    raise ValueError(message)

  if len(arguments.images) == 1 and arguments.reference is None:
    message = "Comparing a single image set with itself seems meaningless"
    raise ValueError(message)

  if arguments.reference is not None and not arguments.reference.is_dir():
    message = f"Reference set is specified but not refer to a directory. The path is {arguments.reference}"
    raise ValueError(message)

  for path in arguments.images:
    if not path.is_dir():
      message = f"Image set does not refer to a directory. The path is {path}"
      raise ValueError(message)

  if arguments.name is not None and len(arguments.name) != len(arguments.images):
    message = "If name is specified, you must specify it for each image set except the reference set"
    raise ValueError(message)

  if arguments.name is None:
    arguments.name = [path.name for path in arguments.images]

  if arguments.output is None:
    arguments.output = Path.cwd().joinpath(
      f"{arguments.name[0]}-vs-{arguments.name[1] if len(arguments.name) > 1 else 'GT'}",
    )

  if not arguments.output.exists():
    arguments.output.mkdir(parents=True, exist_ok=True)

  if not arguments.output.is_dir() or len(list(arguments.output.iterdir())) != 0:
    message = f"either {arguments.output} is not a directory or is not empty"
    raise ValueError(message)

  if arguments.font is not None:
    arguments.font = ImageFont.truetype(arguments.font)

  return arguments


def _collect_ranges_single(
  sources: Sequence[Path],
  previous_result: list[tuple[float, float]] | None,
) -> list[tuple[float, float]]:
  images = [Image.open(source) for source in sources]
  arrays = [numpy.array(image) / 255 for image in images]
  difference = numpy.abs(arrays[0] - arrays[-1]).mean(axis=-1)
  if len(sources) > 2:  # noqa: PLR2004
    difference2 = numpy.abs(arrays[1] - arrays[-1]).mean(axis=-1)
    compare = difference - difference2
    this_result = [
      (
        min(difference.min().item(), difference2.min().item()),
        max(difference.max().item(), difference2.max().item()),
      ),
      (compare.min().item(), compare.max().item()),
    ]
  else:
    this_result = [(difference.min().item(), difference.max().item())]

  if previous_result is None:
    return this_result
  return [
    (min(previous[0], this[0]), max(previous[1], this[1]))
    for previous, this in zip(previous_result, this_result, strict=True)
  ]


def _collect_ranges(
  sources: Sequence[Path],
  names: Sequence[str],
  prefix_length: int,
) -> list[tuple[float, float]]:
  result = None
  for counter, name in enumerate(names):
    result = _collect_ranges_single(
      sources=[source.joinpath(name) for source in sources],
      previous_result=result,
    )
    print(
      f"listing {str(counter + 1).rjust(prefix_length)} / {str(len(names)).rjust(prefix_length)} images",
      end="\r",
    )
  if result is None:
    raise ValueError

  print()

  return result


def cmdline_main() -> None:
  """Entry of commandline tool."""
  arguments = _get_arguments()
  sources = [(source, name) for source, name in zip(arguments.images, arguments.name, strict=True)]
  if arguments.reference is not None:
    sources.append((arguments.reference, "GT"))
  image_names = _select_images([path for path, _ in sources])
  extra_images = 2 if len(arguments.metric) > 0 else 1
  prefix_length = math.ceil(math.log10(len(image_names) + extra_images))  # add cover and summary
  image_size = Image.open(sources[0][0].joinpath(image_names[0])).size

  _render_cover(image_size=image_size, names=[name for _, name in sources], font=arguments.font).save(
    arguments.output.joinpath(f"{str(0).zfill(prefix_length)}_cover.jpg"),
  )

  global_ranges = _collect_ranges(
    sources=[path for path, _ in sources],
    names=image_names,
    prefix_length=prefix_length,
  )

  index = extra_images
  metrics = [[]] if len(sources) == 2 else [[], []]  # noqa: PLR2004
  for counter, name in enumerate(image_names):
    image, calculated_metrics = render_single(
      lhs=(Image.open(arguments.images[0].joinpath(name)), arguments.name[0]),
      rhs=None
      if len(arguments.images) == 1
      else (Image.open(arguments.images[1].joinpath(name)), arguments.name[1]),
      reference=None if arguments.reference is None else Image.open(arguments.reference.joinpath(name)),
      font=arguments.font,
      metrics=arguments.metric,
      global_ranges=global_ranges,
    )
    image.save(arguments.output.joinpath(f"{str(index).zfill(prefix_length)}_{Path(name).stem}.jpg"))
    index += 1

    for i in range(len(calculated_metrics)):
      metrics[i].append((name, calculated_metrics[i]))

    print(
      f"visualized {str(counter + 1).zfill(prefix_length)} / {str(len(image_names)).zfill(prefix_length)} "
      "images",
      end="\r",
    )

  print()

  if len(arguments.metric) > 0:
    _render_metrics_summary(image_size=image_size, metrics=metrics, font=arguments.font).save(
      arguments.output.joinpath(f"{str(1).zfill(prefix_length)}_summary.jpg"),
    )
