from typing import NamedTuple

from numpy import ndarray
from PIL.Image import Image

integer_type = int
floating_type = float
string_type = str
number_type = integer_type | floating_type
image_type = Image
array_type = ndarray


class _Metric(NamedTuple):
  metric_name: str
  value: float


class _Metrics(NamedTuple):
  image_name: str
  metrics: tuple[_Metric, ...]


metric_type = _Metric
metrics_type = _Metrics

value_type = int | float | str | Image | array_type | metrics_type

naming = {
  integer_type: "integer",
  floating_type: "float",
  string_type: "string",
  number_type: "number",
  image_type: "image",
  array_type: "array",
  metrics_type: "metric_t",
}
