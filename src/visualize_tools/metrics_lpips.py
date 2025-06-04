"""Metrics used in visualization: LPIPS."""

try:
  from collections import OrderedDict
  from itertools import chain

  import torch
  from torchvision.models import vgg16

  _target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  _channels_list = [64, 128, 256, 512, 512]

  class _VGG16(torch.nn.Module):
    def __init__(self) -> None:
      super().__init__()

      # register buffer
      self.register_buffer("mean", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
      self.register_buffer("std", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

      # load network layers
      self.layers = vgg16(weights="DEFAULT").features
      self.target_layers = [4, 9, 16, 23, 30]

      for parameter in chain(self.parameters(), self.buffers()):
        parameter.requires_grad = False

    def _z_score(self, x: torch.Tensor) -> torch.Tensor:
      self.mean: torch.Tensor
      self.std: torch.Tensor
      return (x - self.mean) / self.std

    @staticmethod
    def _normalize(x: torch.Tensor, *, eps: float = 1e-10) -> torch.Tensor:
      factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
      return x / (factor + eps)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
      x = self._z_score(x)

      output = []
      for i, (_, layer) in enumerate(self.layers._modules.items(), 1):  # noqa: SLF001
        if layer is None:
          continue
        x = layer(x)
        if i in self.target_layers:
          output.append(self._normalize(x))
        if len(output) == len(self.target_layers):
          break
      return output

  class _InternalState:
    def __init__(self) -> None:
      self.net: _VGG16
      self.layers: torch.nn.ModuleList
      self._initialized: bool = False

    def prepare(self) -> None:
      if self._initialized:
        return
      self.net = _VGG16().to(_target_device)
      self.layers = torch.nn.ModuleList(
        modules=(
          torch.nn.Sequential(
            torch.nn.Identity(),
            torch.nn.Conv2d(channels, 1, 1, 1, 0, bias=False),
          )
          for channels in _channels_list
        ),
      ).to(_target_device)
      for parameter in self.layers.parameters():
        parameter.requires_grad = False
      self.layers.load_state_dict(
        OrderedDict(
          {
            key.replace("lin", "").replace("model.", ""): value
            for key, value in torch.hub.load_state_dict_from_url(
              "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth",
              progress=True,
              map_location=_target_device,
            ).items()
          },
        ),
      )

      self._initialized = True

  _internal_state = _InternalState()

  def calculate_lpips(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Calculate LPIPS, Learned Perceptual Image Patch Similarity."""
    _internal_state.prepare()

    lhs = lhs.to(_target_device).float()
    rhs = rhs.to(_target_device).float()
    lhs, rhs = _internal_state.net(lhs), _internal_state.net(rhs)
    deltas = [(left - right) ** 2 for left, right in zip(lhs, rhs, strict=True)]
    result = [
      layer(delta).mean(dim=(2, 3), keepdim=True)
      for delta, layer in zip(deltas, _internal_state.layers, strict=True)
    ]
    return torch.sum(torch.concatenate(result, dim=0), dim=0, keepdim=True)

  LPIPS_AVAILABLE = True
except ModuleNotFoundError:
  LPIPS_AVAILABLE = False
