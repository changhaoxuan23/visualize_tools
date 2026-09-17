from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Iterator
from typing import NamedTuple, override

from PIL import Image

from .type_helpers import value_type


class Configuration(NamedTuple):
  images: tuple[Image.Image, ...] = ()
  image_size: tuple[int, int] = (0, 0)
  image_filename: str = ""
  image_stem: str = ""
  image_set_names: tuple[str, ...] = ()
  metrics: tuple[str, ...] = ()
  pre_collection: bool = False


class CalculateTreeNode(metaclass=ABCMeta):
  def __init__(self) -> None:
    self._result: value_type | None = None
    self._data_sources = _DataSource()

  def clear(self) -> None:
    self._result = None
    for data_source in self._data_sources.sources():
      data_source.clear()

  @abstractmethod
  def prepare_result(self) -> None:
    """Calculate result and store to self._result."""

  @property
  @abstractmethod
  def result_type(self) -> type:
    pass

  @abstractmethod
  def configure(self, configuration: Configuration) -> None:
    pass

  def config(self, configuration: Configuration) -> None:
    self.configure(configuration)
    for data_source in self._data_sources.sources():
      data_source.config(configuration)

  @property
  def result(self) -> value_type:
    if self._result is None:
      self.prepare_result()
    if self._result is None:
      raise ValueError
    return self._result


class _DataSource:
  def __init__(self) -> None:
    self._data: list[CalculateTreeNode] = []
    self._name: dict[str, int] = {}

  def register(self, source: CalculateTreeNode, *, name: str | None = None) -> int:
    if name is not None and name in self._name:
      raise KeyError
    index = len(self._data)
    self._data.append(source)
    if name is not None:
      self._name[name] = index
      object.__setattr__(self, name, self._data[index])
    return index

  def __contains__(self, key: str | int) -> bool:
    if isinstance(key, str):
      return key in self._name
    if isinstance(key, int):
      return key < len(self._data)
    return False

  def __getitem__(self, key: str | int) -> CalculateTreeNode:
    if key not in self:
      raise KeyError
    if isinstance(key, str):
      return self._data[self._name[key]]
    if isinstance(key, int):
      return self._data[key]
    raise KeyError

  def __iter__(self) -> Iterator[str]:
    return self.names()

  def names(self) -> Iterator[str]:
    return iter(self._name)

  def sources(self) -> Iterator[CalculateTreeNode]:
    return iter(self._data)


class VisualizedImageBuilder(CalculateTreeNode):
  def __init__(self, grid: tuple[int, int]) -> None:
    """Tile chunks from different sources into the final visualized image.

    grid: (rows, columns), chunk grid specification
    """
    super().__init__()
    self._grid = grid

    self._source_mapping: list[tuple[int, int]] = []

  @override
  def configure(self, configuration: Configuration) -> None:
    self._size = configuration.image_size

  def add_chunk(self, source: CalculateTreeNode, position: tuple[int, int]) -> None:
    self._data_sources.register(source)
    self._source_mapping.append(position)

  @property
  @override
  def result_type(self) -> type:
    return Image.Image

  @override
  def prepare_result(self) -> None:
    chunk_width, chunk_height = self._size
    chunk_rows, chunk_columns = self._grid
    self._result = Image.new(
      mode="RGB",
      size=(chunk_width * chunk_columns, chunk_height * chunk_rows),
      color="black",
    )

    for source, (row, column) in zip(self._data_sources.sources(), self._source_mapping, strict=True):
      image = source.result
      if not isinstance(image, Image.Image):
        raise TypeError
      self._result.paste(image, box=(column * chunk_width, row * chunk_height))


class ConstantGenerator(CalculateTreeNode):
  def __init__(self, *, constant_value: value_type, constant_type: type) -> None:
    if not isinstance(constant_value, constant_type):
      raise TypeError

    super().__init__()
    self._type = constant_type
    self._value = constant_value

  @override
  def configure(self, configuration: Configuration) -> None:
    pass

  @property
  @override
  def result_type(self) -> type:
    return self._type

  @override
  def prepare_result(self) -> None:
    self._result = self._value


class StringGenerator(CalculateTreeNode):
  @staticmethod
  def _find_plain_text(string: str, start: int) -> tuple[str, int]:
    _replacers = {"%%": "%", "%I": "|", "%J": '"'}
    buffer = ""

    while True:
      escape_sequence_index = string.find("%", start)
      if escape_sequence_index == -1:
        return buffer + string[start:], len(string)
      buffer += string[start:escape_sequence_index]

      if string[escape_sequence_index : escape_sequence_index + 2] in _replacers:
        buffer += _replacers[string[escape_sequence_index : escape_sequence_index + 2]]
        start += 2
      else:
        return buffer, escape_sequence_index

  @staticmethod
  def _build_set_name_replacer(sequence: str) -> Callable[[Configuration], str]:
    index = int(sequence[1:])
    return lambda configuration: configuration.image_set_names[index]

  @staticmethod
  def build(template: str) -> CalculateTreeNode:
    if template[0] != '"' or template[-1] != '"':
      raise ValueError
    original_template = template
    template = template[1:-1]

    start = 0
    parts: list[Callable[[Configuration], str]] = []
    special = False
    while start < len(template):
      plain, start = StringGenerator._find_plain_text(string=template, start=start)
      if plain != "":
        parts.append(lambda _, plain=plain: plain)

      match template[start : start + 2]:
        case "%n":
          special = True
          parts.append(lambda configuration: configuration.image_stem)
          start += 2
        case "%N":
          special = True
          parts.append(lambda configuration: configuration.image_filename)
          start += 2
        case "%":
          message = f"Invalid string with partial tailing escape sequence: {original_template}"
          raise SyntaxError(message)
        case _:
          if not template[start + 1].isdigit():
            message = f"Invalid escape sequence in string: {template[start:]}"
            raise SyntaxError(message)
          i = 2
          while start + i <= len(template) and template[start + 1 : start + i].isdigit():
            i += 1
          parts.append(StringGenerator._build_set_name_replacer(template[start : start + i]))
          special = True
          start += i

    if len(parts) == 0:
      parts = [lambda _: ""]

    return (
      StringGenerator(parts=parts)
      if special
      else ConstantGenerator(
        constant_value=parts[0](Configuration()),
        constant_type=str,
      )
    )

  def __init__(self, parts: list[Callable[[Configuration], str]]) -> None:
    super().__init__()

    self._parts = parts

  @override
  def configure(self, configuration: Configuration) -> None:
    self._configuration = configuration

  @property
  @override
  def result_type(self) -> type:
    return str

  @override
  def prepare_result(self) -> None:
    self._result = "".join([part(self._configuration) for part in self._parts])


class NegativeLayer(CalculateTreeNode):
  def __init__(self, other: CalculateTreeNode) -> None:
    super().__init__()

    if other.result_type not in (int, float):
      raise TypeError

    self._data_sources.register(source=other)

  @override
  def configure(self, configuration: Configuration) -> None:
    pass

  @property
  @override
  def result_type(self) -> type:
    return self._data_sources[0].result_type

  @override
  def prepare_result(self) -> None:
    if not isinstance(self._data_sources[0].result, int) and not isinstance(
      self._data_sources[0].result,
      float,
    ):
      raise TypeError
    self._result = -self._data_sources[0].result


class ImageAccessor(CalculateTreeNode):
  """Accessor to input images."""

  def __init__(self, index: int) -> None:
    super().__init__()

    self._index = index

  @override
  def configure(self, configuration: Configuration) -> None:
    self._configuration = configuration

  @property
  @override
  def result_type(self) -> type:
    return Image.Image

  @override
  def prepare_result(self) -> None:
    if self._index >= len(self._configuration.images):
      message = f"Index {self._index} out of bound: only {len(self._configuration.images)} images available"
      raise IndexError(message)
    self._result = self._configuration.images[self._index]


class FileAccessor(CalculateTreeNode):
  """Accessor to image file."""

  def __init__(self, path: CalculateTreeNode) -> None:
    super().__init__()

    if not issubclass(path.result_type, str):
      raise TypeError
    self._path = path

  @override
  def configure(self, configuration: Configuration) -> None:
    pass

  @property
  @override
  def result_type(self) -> type:
    return Image.Image

  @override
  def prepare_result(self) -> None:
    path = self._path.result
    if not isinstance(path, str):
      raise TypeError
    self._result = Image.open(path)
