import torch
from torch.overrides import has_torch_function_variadic, handle_torch_function
from torch.nn import _reduction as _Reduction
from typing import Optional

Tensor = torch.Tensor

def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    pos_weight: Optional[Tensor] = None,
) -> Tensor:
    r"""Function that measures Binary Cross Entropy between target and input
    logits.
    See :class:`~torch.nn.BCEWithLogitsLoss` for details.
    Args:
        input: Tensor of arbitrary shape as unnormalized scores (often referred to as logits).
        target: Tensor of the same shape as input with values between 0 and 1
        weight (Tensor, optional): a manual rescaling weight
            if provided it's repeated to match input tensor shape
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        pos_weight (Tensor, optional): a weight of positive examples.
                Must be a vector with length equal to the number of classes.
    Examples::
         >>> input = torch.randn(3, requires_grad=True)
         >>> target = torch.empty(3).random_(2)
         >>> loss = F.binary_cross_entropy_with_logits(input, target)
         >>> loss.backward()
    """
    if has_torch_function_variadic(input, target, weight, pos_weight):
        return handle_torch_function(
            binary_cross_entropy_with_logits,
            (input, target, weight, pos_weight),
            input,
            target,
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            pos_weight=pos_weight,
        )
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)

    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    return torch.binary_cross_entropy_with_logits(input, target, weight, pos_weight, reduction_enum)