r"""Private plumbing so ``layer.weight = ...`` works on wrapper modules whose
weight/bias are exposed as properties.

``nn.Module.__setattr__`` intercepts :class:`~torch.nn.Parameter` values
*before* the property descriptor is consulted and calls
``register_parameter``, which raises ``KeyError`` when the class already
exposes a ``weight``/``bias`` property. These mixins route those two names to
the property setter when one exists.
"""

import torch.nn as nn

__all__ = ["_RoutePropertySetattrMixin", "_WrappedWeightBiasMixin"]

_ROUTED_NAMES = ("weight", "bias")


class _RoutePropertySetattrMixin:
    r"""Route ``weight``/``bias`` assignment to the class property's setter."""

    def __setattr__(self, name, value):
        if name in _ROUTED_NAMES:
            prop = getattr(type(self), name, None)
            if isinstance(prop, property) and prop.fset is not None:
                prop.fset(self, value)
                return
        super().__setattr__(name, value)


class _WrappedWeightBiasMixin(_RoutePropertySetattrMixin):
    r"""Live ``weight``/``bias`` passthrough for native-wrapper layers.

    Subclasses set ``_wrapped_attr`` to the name of the wrapped ``torch.nn``
    module. The getters return the wrapped module's parameter directly
    (storage-shared, so ``nn.init.*_(layer.weight)`` and
    ``layer.weight.data.copy_(...)`` work); the setters delegate to the
    wrapped module, so ``layer.weight = nn.Parameter(w)`` rebinds exactly as
    raw torch does and a plain-tensor assignment raises torch's own
    ``TypeError``.
    """

    _wrapped_attr: str

    @property
    def weight(self) -> nn.Parameter:
        return getattr(self, self._wrapped_attr).weight

    @weight.setter
    def weight(self, value) -> None:
        getattr(self, self._wrapped_attr).weight = value

    @property
    def bias(self) -> nn.Parameter | None:
        return getattr(self, self._wrapped_attr).bias

    @bias.setter
    def bias(self, value) -> None:
        getattr(self, self._wrapped_attr).bias = value
