"""Parse the expression mini-language.

Language syntax summary:
expression ::= sub-expression { ';' sub-expression } [ ';' ]
sub-expression ::= data-source { '|' function-call } '|' destination
data-source ::= function-call | path-reference | variable-reference | immediate | binary-exp
function-call ::= symbol '(' [ data-source { ',' data-source } ] ')'
path-reference ::= 'image@:' string
variable-reference ::= ( 'image:' | 'array:' ) symbol
immediate ::= string | number
binary-exp ::= data-source ( '+' | '-' ) data=-source
destination ::= location-specification | variable-specification
location-specification ::= '[' integer ',' integer ']'
variable-specification ::= 'to:' symbol
string ::= '"' <what the content of a string should look like> '"'
symbol ::= <any valid symbol in C>
"""

from collections.abc import Callable
from io import StringIO
from typing import NamedTuple

from PIL import Image

from . import calculate_tree, type_helpers


class _StringHelper:
  def __init__(self, expression: str) -> None:
    self._string = expression
    self._peeked_token: str | None = None

  def _find_first_compiling(self, *, skip: int = 0, predict: Callable[[str], bool]) -> int:
    for i in range(skip, len(self._string), 1):
      if predict(self._string[i]):
        return i
    return len(self._string)

  def _find_longest_compiling(self, *, skip: int = 0, predict: Callable[[str], bool]) -> int:
    for i in range(skip, len(self._string), 1):
      if not predict(self._string[skip : i + 1]):
        return i
    return len(self._string)

  def _get_string(self) -> str:
    terminating = self._string.find('"', 1)
    if terminating == -1:
      message = f"string starting with {self._string[:10]} is not properly enclosed."
      raise SyntaxError(message)
    result = self._string[: terminating + 1]
    self._string = self._string[terminating + 1 :]
    return result

  def _get_number(self) -> str:
    terminating = self._find_first_compiling(predict=lambda x: not x.isdigit() and x != ".")
    result = self._string[:terminating]
    self._string = self._string[terminating:]
    return result

  def _get_symbol(self) -> str:
    terminating = self._find_longest_compiling(predict=lambda x: x.isidentifier())
    if terminating == 0:
      message = f"Unexpected token starting with {self._string[:10]}"
      raise SyntaxError(message)

    pending_symbol = self._string[:terminating]
    self._string = self._string[terminating:]
    if pending_symbol in ["image", "array", "to"] and len(self._string) > 0 and self._string[0] == ":":
      self._string = self._string[1:]
      return pending_symbol + ":"

    if pending_symbol == "image" and len(self._string) > 1 and self._string[:2] == "@:":
      self._string = self._string[2:]
      return "image@:"

    return pending_symbol

  def get_token(self) -> str | None:
    if self._peeked_token is not None:
      token = self._peeked_token
      self._peeked_token = None
      return token

    self._string = self._string.lstrip()
    if len(self._string) == 0:
      return None

    token = None

    match self._string[0]:
      case '"':
        token = self._get_string()
      case ";" | "|" | "(" | ")" | "," | "[" | "]":
        result = self._string[0]
        self._string = self._string[1:]
        token = result
      case "+" | "-":
        token = self._string[0]
        self._string = self._string[1:]

    if token is not None:
      return token

    return self._get_number() if self._string[0].isdigit() or self._string[0] == "." else self._get_symbol()

  def peek_token(self) -> str | None:
    if self._peeked_token is not None:
      return self._peeked_token

    self._peeked_token = self.get_token()
    return self._peeked_token


def _parse_variable_reference(
  reader: _StringHelper,
  symbol_table: dict[str, calculate_tree.CalculateTreeNode],
  prefix: str,
) -> calculate_tree.CalculateTreeNode:
  variable_name = reader.get_token()
  if variable_name is None:
    message = "unexpected EOF when reading variable name"
    raise SyntaxError(message)

  if variable_name.isdigit():
    return calculate_tree.ImageAccessor(index=int(variable_name))

  if variable_name not in symbol_table:
    message = f"Reference to undefined variable {variable_name}"
    raise SyntaxError(message)

  _typing_match = {"image:": type_helpers.image_type, "array:": type_helpers.array_type}
  if not issubclass(_typing_match[prefix], symbol_table[variable_name].result_type):
    message = "Invalid variable access: type mismatch"
    raise SyntaxError(message)  # noqa: TRY004 : This is actually a syntax error

  return symbol_table[variable_name]


def _parse_file_reference(
  reader: _StringHelper,
) -> calculate_tree.CalculateTreeNode:
  path_template = reader.get_token()
  if path_template is None:
    message = "unexpected EOF when reading file path"
    raise SyntaxError(message)

  path = calculate_tree.StringGenerator.build(template=path_template)
  if isinstance(path, calculate_tree.ConstantGenerator):
    if not isinstance(path.result, str):
      raise ValueError
    image = Image.open(path.result)
    return calculate_tree.ConstantGenerator(constant_value=image, constant_type=type_helpers.image_type)

  return calculate_tree.FileAccessor(path=path)


def list_functions() -> str:
  buffer = StringIO()
  for name, specification in _function_table.items():
    print(f"{name} -> {type_helpers.naming[specification.return_type]}:", file=buffer)
    for overload in specification.overloads:
      parameter_list = ", ".join(
        f"{parameter.name}: {type_helpers.naming[parameter.argument_type]}"
        f"{' = ' + parameter.default_value.__repr__() if parameter.have_default_value else ''}"
        for parameter in overload
      )
      print(f"  -- ({parameter_list})", file=buffer)
  return buffer.getvalue()


def _collect_argument_list(
  reader: _StringHelper,
  symbol_table: dict[str, calculate_tree.CalculateTreeNode],
) -> list[calculate_tree.CalculateTreeNode]:
  if reader.peek_token() != "(":
    message = f"Expected '(' but got {reader.peek_token()}"
    raise SyntaxError(message)

  reader.get_token()

  result: list[calculate_tree.CalculateTreeNode] = []
  while reader.peek_token() != ")":
    if reader.peek_token() is None:
      message = "Unexpected EOF when collecting argument list"
      raise SyntaxError(message)
    result.append(_parse_expression(reader=reader, symbol_table=symbol_table))
    if reader.peek_token() == ",":
      reader.get_token()
  return result


def _parse_function_call(
  reader: _StringHelper,
  symbol_table: dict[str, calculate_tree.CalculateTreeNode],
  *,
  function_name: str | None = None,
  previous_result: calculate_tree.CalculateTreeNode | None = None,
) -> calculate_tree.CalculateTreeNode:
  if function_name is None:
    function_name = reader.get_token()
  if function_name is None:
    message = "expect function name, got EOF"
    raise SyntaxError(message)
  if function_name not in _function_table:
    message = f"reference to undefined function {function_name}"
    raise SyntaxError(message)

  argument_list: list[calculate_tree.CalculateTreeNode] = []
  if previous_result is not None:
    argument_list.append(previous_result)
  if reader.peek_token() == "(":
    argument_list.extend(_collect_argument_list(reader=reader, symbol_table=symbol_table))

  specification = _function_table[function_name]


def _parse_unary_expression(
  reader: _StringHelper,
  symbol_table: dict[str, calculate_tree.CalculateTreeNode],
) -> calculate_tree.CalculateTreeNode:
  token = reader.get_token()
  if token is None:
    message = "expected a data source but got nothing"
    raise SyntaxError(message)

  if token == "-":
    number = _parse_unary_expression(reader=reader, symbol_table=symbol_table)
    return calculate_tree.NegativeLayer(other=number)

  match token[0]:
    case '"':
      result = calculate_tree.StringGenerator.build(template=token)
    case ".":
      result = calculate_tree.ConstantGenerator(constant_value=float(token), constant_type=float)

  match token:
    case "image:" | "array:":
      result = _parse_variable_reference(reader=reader, symbol_table=symbol_table, prefix=token)
    case "image@:":
      result = _parse_file_reference(reader=reader)

  if token[0].isdigit():
    number_class = float if "." in token else int
    result = calculate_tree.ConstantGenerator(constant_value=number_class(token), constant_type=number_class)
  else:
    result = _parse_function_call(reader=reader, symbol_table=symbol_table)

  return result


def _parse_expression(
  reader: _StringHelper,
  symbol_table: dict[str, calculate_tree.CalculateTreeNode],
) -> calculate_tree.CalculateTreeNode:
  """Expression here has generally the same meaning as one in C syntax."""
  lhs = _parse_unary_expression(reader=reader, symbol_table=symbol_table)
  token = reader.peek_token()
  if token is None:
    message = "expected a data source but got nothing"
    raise SyntaxError(message)

  match token[0]:
    case '"':
      result = calculate_tree.StringGenerator.build(template=token)
    case ".":
      result = calculate_tree.ConstantGenerator(constant_value=float(token), constant_type=float)


def _parse_sub_expression(
  reader: _StringHelper,
  symbol_table: dict[str, calculate_tree.CalculateTreeNode],
) -> tuple[calculate_tree.CalculateTreeNode, tuple[int, int]] | None:
  pass


def parse(expression: str) -> calculate_tree.VisualizedImageBuilder:
  reader = _StringHelper(expression)
  symbol_table: dict[str, calculate_tree.CalculateTreeNode] = {}

  maximum_row, maximum_column = 0, 0
  chunks: list[tuple[calculate_tree.CalculateTreeNode, tuple[int, int]]] = []
  while True:
    result = _parse_sub_expression(reader=reader, symbol_table=symbol_table)
    if result is not None:
      _, (row, column) = result
      maximum_row = max(maximum_row, row)
      maximum_column = max(maximum_column, column)
      chunks.append(result)
    separator = reader.get_token()
    if separator is None:
      break
    if separator != ";":
      message = f"expected ';' but got '{separator}'"
      raise SyntaxError(message)

  result = calculate_tree.VisualizedImageBuilder(grid=(maximum_row + 1, maximum_column + 1))
  for source, position in chunks:
    result.add_chunk(source=source, position=position)

  return result


if __name__ == "__main__":
  from sys import stdin

  helper = _StringHelper(stdin.read())
  while True:
    token = helper.get_token()
    if token is None:
      break
    print(token)
