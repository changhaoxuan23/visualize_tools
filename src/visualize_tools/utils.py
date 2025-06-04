"""Utilities used in visualize_tools."""

import math
from collections.abc import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont
from PIL.ImageDraw import _Ink


def find_shape(total: int) -> tuple[int, int]:
  """Find a suitable shape for placing certain amount of blocks as a grid.

  Returns tuple of (columns, rows)
  """
  if total == 0:
    return 0, 0

  if total < 0:
    raise ValueError

  candidate_columns = range(math.ceil(math.sqrt(total)), total + 1)[:4]
  candidates = [
    (columns, math.ceil(total / columns), math.ceil(total / columns) * columns - total)
    for columns in candidate_columns
  ]
  candidates.sort(key=lambda item: item[2], reverse=False)
  return candidates[0][:2]


def tile_images(
  images: Sequence[Image.Image],
  *,
  rows: int | None = None,
  columns: int | None = None,
) -> Image.Image:
  """Tile a list of images into a big picture, from the left to the right, from the top to the bottom."""
  if rows is not None and columns is not None:
    final_rows = rows
    final_columns = columns
  elif rows is not None:
    final_rows = rows
    final_columns = (len(images) + rows - 1) // rows
  elif columns is not None:
    final_columns = columns
    final_rows = (len(images) + columns - 1) // columns
  else:
    final_columns, final_rows = find_shape(len(images))

  if final_columns * final_rows < len(images):
    message = (
      f"Invalid shape configuration: {final_columns} x {final_rows} grid cannot hold {len(images)} images"
    )
    raise ValueError(message)

  result_image = Image.new(
    mode=images[0].mode,
    size=(final_columns * images[0].width, final_rows * images[0].height),
  )
  for index, image in enumerate(images):
    row = index // final_columns
    column = index % final_columns
    result_image.paste(image, box=(column * images[0].width, row * images[0].height))
  return result_image


def predict_render_size(
  *args,  # noqa: ANN002
  texts: Iterable[str],
  font: ImageFont.FreeTypeFont,
  **kwargs,  # noqa: ANN003
) -> tuple[int, int]:
  """Calculate the maximum render size of texts with given arguments.

  Returns a tuple of (width, height).
  """
  boxes = [font.getbbox(*args, text=text, **kwargs) for text in texts]
  return (int(max(box[2] - box[0] for box in boxes)), int(max(box[3] - box[1] for box in boxes)))


def fit_font(
  *args,  # noqa: ANN002
  texts: Sequence[str],
  font: ImageFont.FreeTypeFont,
  height: int | None = None,
  width: int | None = None,
  **kwargs,  # noqa: ANN003
) -> tuple[ImageFont.FreeTypeFont, int, int]:
  """Find the maximum font size that fits into the bounding box.

  To be specific, this function returns a variant of the font specified whose size may be modified.
  When rendering texts with the font returned with exactly the same arguments specified to this function,
   the space required to display the result will never exceed the specified height/width.

  This function returns a three-tuple: the font, the maximum width and the maximum height when rendering
   with the font
  """
  if height is None and width is None:
    return (font, *predict_render_size(*args, texts=texts, font=font, **kwargs))

  candidate_fonts = (
    font.font_variant(size=size) for size in range(max([v for v in [height, width] if v is not None]), 0, -1)
  )
  rendered_sizes = (predict_render_size(*args, texts=texts, font=font, **kwargs) for font in candidate_fonts)

  return next(
    (font, render_width, render_height)
    for (render_width, render_height), font in zip(rendered_sizes, candidate_fonts, strict=True)
    if (width is None or render_width <= width) and (height is None or render_height <= height)
  )


def place_lines(  # noqa: PLR0913
  image: Image.Image,
  lines: Sequence[str],
  font: ImageFont.FreeTypeFont,
  *,
  xy: tuple[int, int] = (0, 0),
  line_distance: int = 0,
  target_width: int | None = None,
  foreground: _Ink = "white",
  background: tuple[_Ink, float] | None = None,
  padding: int = 0,
) -> None:
  """Render lines to the image.

  Use xy to set top-left corner of the background box, default to the top-left corner of the image
  """
  drawer = ImageDraw.Draw(image)
  font, width, height = fit_font(
    texts=lines,
    font=font,
    width=target_width or int(image.width * 0.4),
  )

  if background is not None:
    textbox_size = (width + padding * 2, height * len(lines) + line_distance * (len(lines) - 1) + padding * 2)
    background_box = Image.new(
      mode=image.mode,
      size=textbox_size,
      color=background[0],
    )
    image.paste(
      Image.blend(
        image.crop((*xy, xy[0] + textbox_size[0], xy[1] + textbox_size[1])),
        background_box,
        alpha=background[1],
      ),
      box=xy,
    )

  start_height = padding + xy[1]
  for line in lines:
    drawer.text(xy=(padding + xy[0], start_height), text=line, font=font, anchor="lt", fill=foreground)
    start_height += height + line_distance
