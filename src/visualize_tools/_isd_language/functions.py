from collections.abc import Callable, Sequence
from functools import partial
from typing import NamedTuple, override

import numpy
from PIL import Image, ImageFont, ImageDraw

from visualize_tools.metrics import METRICS

from . import type_helpers
from .calculate_tree import CalculateTreeNode, Configuration, ConstantGenerator


class FunctionInvoker(CalculateTreeNode):
  def __init__(
    self,
    function: Callable,
    arguments: list[CalculateTreeNode | type_helpers.value_type],
    invoke_result: type,
  ) -> None:
    super().__init__()

    self._function = function
    self._arguments = arguments
    self._invoke_result = invoke_result

    for argument in self._arguments:
      if isinstance(argument, CalculateTreeNode):
        self._data_sources.register(argument)

  @override
  def configure(self, configuration: Configuration) -> None:
    pass

  @property
  @override
  def result_type(self) -> type:
    return self._invoke_result

  @override
  def prepare_result(self) -> None:
    result = self._function(
      *[
        argument.result if isinstance(argument, CalculateTreeNode) else argument
        for argument in self._arguments
      ],
    )
    if not isinstance(result, self._invoke_result):
      raise TypeError
    self._result = result


def _calculate_metrics(
  lhs: type_helpers.image_type,
  rhs: type_helpers.image_type,
  _: str,
  /,
  *,
  metrics: Sequence[str],
  image_name: str,
) -> type_helpers.metrics_type:
  return type_helpers.metrics_type(
    image_name=image_name,
    metrics=tuple(
      type_helpers.metric_type(metric_name=metric, value=METRICS[metric](lhs, rhs)) for metric in metrics
    ),
  )


class MetricsCalculator(FunctionInvoker):
  def __init__(
    self,
    function: Callable,
    arguments: list[CalculateTreeNode | type_helpers.value_type],
    invoke_result: type,
  ) -> None:
    super().__init__(function=function, arguments=arguments, invoke_result=invoke_result)

    _name = arguments[-1]
    if isinstance(_name, ConstantGenerator):
      _name = _name.result
      if not isinstance(_name, str):
        raise TypeError
    self._name = _name

    self._metrics_list: list[type_helpers.metrics_type] | None = None if self._name == "" else []

  @override
  def configure(self, configuration: Configuration) -> None:
    self._configuration = configuration

  @property
  @override
  def result_type(self) -> type:
    return type_helpers.metrics_type

  @override
  def prepare_result(self) -> None:
    self._function = partial(
      _calculate_metrics,
      metrics=self._configuration.metrics,
      image_name=self._configuration.image_filename,
    )

    super().prepare_result()

    if not isinstance(self._result, type_helpers.metrics_type):
      raise TypeError
    if self._metrics_list is not None:
      self._metrics_list.append(self._result)


class RangeCollectionLayer(FunctionInvoker):
  def __init__(
    self,
    function: Callable,
    arguments: list[CalculateTreeNode | type_helpers.value_type],
    invoke_result: type,
  ) -> None:
    super().__init__(function=function, arguments=arguments, invoke_result=invoke_result)

    self._input = arguments[0]
    self._saved_range: tuple[float | None, float | None] = None, None
    self._configured: bool = False

    if not isinstance(self._input, CalculateTreeNode):
      raise TypeError

  @override
  def configure(self, configuration: Configuration) -> None:
    if not configuration.pre_collection and not self._configured:
      self._function = partial(self._function, ranges=self._saved_range)
      self._configured = True

  @property
  @override
  def result_type(self) -> type:
    return type_helpers.image_type

  @override
  def prepare_result(self) -> None:
    if self._configured:
      super().prepare_result()
      return

    if not isinstance(self._input, CalculateTreeNode):
      raise TypeError
    _input = self._input.result
    if not isinstance(_input, type_helpers.array_type):
      raise TypeError
    minimum, maximum = _input.min().item(), _input.max().item()
    self._saved_range = (
      minimum if self._saved_range[0] is None else min(minimum, self._saved_range[0]),
      maximum if self._saved_range[1] is None else min(maximum, self._saved_range[1]),
    )


def _metrics_to_string(metrics: type_helpers.metrics_type) -> str:
  return "\n".join([metrics.image_name, *[f"{name} = {str(value)[:7]}" for name, value in metrics]])

def _line_to_image(
  lines: str,
  /,
  *,
  font: ImageFont.FreeTypeFont,
  target_width: int,
) -> type_helpers.image_type:
  _lines = lines.split("\n")


class _FunctionArgumentSpecification(NamedTuple):
  name: str
  argument_type: type
  have_default_value: bool
  default_value: type_helpers.value_type | None


class _FunctionOverloadSpecification(NamedTuple):
  parameters: tuple[_FunctionArgumentSpecification, ...]
  implementation: Callable
  node: type[FunctionInvoker]


class _FunctionSpecification(NamedTuple):
  overloads: tuple[_FunctionOverloadSpecification, ...]
  return_type: type


_function_table: dict[str, _FunctionSpecification] = {
  "abs": _FunctionSpecification(
    overloads=(
      _FunctionOverloadSpecification(
        parameters=(
          _FunctionArgumentSpecification(
            name="value",
            argument_type=type_helpers.array_type,
            have_default_value=False,
            default_value=None,
          ),
        ),
        implementation=numpy.absolute,
        node=FunctionInvoker,
      ),
    ),
    return_type=type_helpers.array_type,
  ),
  "metrics": _FunctionSpecification(
    overloads=(
      _FunctionOverloadSpecification(
        parameters=(
          _FunctionArgumentSpecification(
            name="lhs",
            argument_type=type_helpers.array_type,
            have_default_value=False,
            default_value=None,
          ),
          _FunctionArgumentSpecification(
            name="rhs",
            argument_type=type_helpers.array_type,
            have_default_value=False,
            default_value=None,
          ),
          _FunctionArgumentSpecification(
            name="name",
            argument_type=type_helpers.string_type,
            have_default_value=True,
            default_value="",
          ),
        ),
        implementation=_calculate_metrics,
        node=MetricsCalculator,
      ),
    ),
    return_type=type_helpers.metrics_type,
  ),
  "to_string": _FunctionSpecification(
    overloads=(
      _FunctionOverloadSpecification(
        parameters=(
          _FunctionArgumentSpecification(
            name="value",
            argument_type=type_helpers.metrics_type,
            have_default_value=False,
            default_value=None,
          ),
        ),
        implementation=_metrics_to_string,
        node=FunctionInvoker,
      ),
    ),
    return_type=type_helpers.string_type,
  ),
  "to_image": _FunctionSpecification(
    overloads=(
      _FunctionOverloadSpecification(
        parameters=(
          _FunctionArgumentSpecification(
            name="string",
            argument_type=type_helpers.string_type,
            have_default_value=False,
            default_value=None,
          ),
        ),
        implementation=,
        node=,
      ),
    ),
    return_type=type_helpers.image_type,
  ),
  "to_difference_image": _FunctionSpecification(
    overloads=(
      (
        _FunctionArgumentSpecification(
          name="value",
          argument_type=type_helpers.array_type,
          have_default_value=False,
          default_value=None,
        ),
      ),
    ),
    return_type=type_helpers.image_type,
  ),
  "to_aligned_difference_image": _FunctionSpecification(
    overloads=(
      (
        _FunctionArgumentSpecification(
          name="value",
          argument_type=type_helpers.array_type,
          have_default_value=False,
          default_value=None,
        ),
      ),
    ),
    return_type=type_helpers.image_type,
  ),
  "to_comparison_image": _FunctionSpecification(
    overloads=(
      (
        _FunctionArgumentSpecification(
          name="value",
          argument_type=type_helpers.array_type,
          have_default_value=False,
          default_value=None,
        ),
      ),
    ),
    return_type=type_helpers.image_type,
  ),
  "to_aligned_comparison_image": _FunctionSpecification(
    overloads=(
      (
        _FunctionArgumentSpecification(
          name="value",
          argument_type=type_helpers.array_type,
          have_default_value=False,
          default_value=None,
        ),
      ),
    ),
    return_type=type_helpers.image_type,
  ),
  "blend": _FunctionSpecification(
    overloads=(
      (
        _FunctionArgumentSpecification(
          name="lhs",
          argument_type=type_helpers.image_type,
          have_default_value=False,
          default_value=None,
        ),
        _FunctionArgumentSpecification(
          name="rhs",
          argument_type=type_helpers.image_type,
          have_default_value=False,
          default_value=None,
        ),
        _FunctionArgumentSpecification(
          name="alpha",
          argument_type=type_helpers.floating_type,
          have_default_value=True,
          default_value=0.4,
        ),
      ),
    ),
    return_type=type_helpers.image_type,
  ),
}
