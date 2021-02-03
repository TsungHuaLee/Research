# +
import torch.nn as nn
import torch 
import torch.nn.functional as F
import numpy as np

def range_to_anchors_and_delta(precision_range, num_anchors):
    """Calculates anchor points from precision range.
    Args:
        precision_range: an interval (a, b), where 0.0 <= a <= b <= 1.0
        num_anchors: int, number of equally spaced anchor points.
    Returns:
        precision_values: A `Tensor` of [num_anchors] equally spaced values
            in the interval precision_range.
        delta: The spacing between the values in precision_values.
    Raises:
        ValueError: If precision_range is invalid.
    """
    # Validate precision_range.
    if len(precision_range) != 2:
        raise ValueError(
            "length of precision_range (%d) must be 2" % len(precision_range)
        )
    if not 0 <= precision_range[0] <= precision_range[1] <= 1:
        raise ValueError(
            "precision values must follow 0 <= %f <= %f <= 1"
            % (precision_range[0], precision_range[1])
        )

    # Sets precision_values uniformly between min_precision and max_precision.
    precision_values = np.linspace(
        start=precision_range[0], stop=precision_range[1], num=num_anchors + 1
    )[1:]

    delta = (precision_range[1] - precision_range[0]) / num_anchors
    return torch.cuda.FloatTensor(precision_values), delta

def build_class_priors(
    labels,
    class_priors=None,
    weights=None,
    positive_pseudocount=1.0,
    negative_pseudocount=1.0,
):
    """build class priors, if necessary.
    For each class, the class priors are estimated as
    (P + sum_i *w_i *y_i) / (P + N + sum_i *w_i),
    where y_i is the ith label, w_i is the ith weight, P is a pseudo-count of
    positive labels, and N is a pseudo-count of negative labels.
    Args:
        labels: A `Tensor` with shape [batch_size, num_classes].
            Entries should be in [0, 1].
        class_priors: None, or a floating point `Tensor` of shape [C]
            containing the prior probability of each class (i.e. the fraction of the
            training data consisting of positive examples). If None, the class
            priors are computed from `targets` with a moving average.
        weights: `Tensor` of shape broadcastable to labels, [N, 1] or [N, C],
            where `N = batch_size`, C = num_classes`
        positive_pseudocount: Number of positive labels used to initialize the class
            priors.
        negative_pseudocount: Number of negative labels used to initialize the class
            priors.
    Returns:
        class_priors: A Tensor of shape [num_classes] consisting of the
          weighted class priors, after updating with moving average ops if created.
    """
    if class_priors is not None:
        return class_priors

    N, C = labels.size()

    weighted_label_counts = (weights * labels).sum(0)

    weight_sum = weights.sum(0)

    class_priors = torch.div(
        weighted_label_counts + positive_pseudocount,
        weight_sum + positive_pseudocount + negative_pseudocount,
    )

    return class_priors

class LagrangeMultiplier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def lagrange_multiplier(x):
    return LagrangeMultiplier.apply(x)

def weighted_hinge_loss(labels, logits, positive_weights=1.0, negative_weights=1.0):
    """
    Args:
        labels: one-hot representation `Tensor` of shape broadcastable to logits
        logits: A `Tensor` of shape [N, C] or [N, C, K]
        positive_weights: Scalar or Tensor
        negative_weights: same shape as positive_weights
    Returns:
        3D Tensor of shape [N, C, K], where K is length of positive weights
        or 2D Tensor of shape [N, C]
    """
    positive_weights_is_tensor = torch.is_tensor(positive_weights)
    negative_weights_is_tensor = torch.is_tensor(negative_weights)

    # Validate positive_weights and negative_weights
    if positive_weights_is_tensor ^ negative_weights_is_tensor:
        raise ValueError(
            "positive_weights and negative_weights must be same shape Tensor "
            "or both be scalars. But positive_weight_is_tensor: %r, while "
            "negative_weight_is_tensor: %r"
            % (positive_weights_is_tensor, negative_weights_is_tensor)
        )

    if positive_weights_is_tensor and (
        positive_weights.size() != negative_weights.size()
    ):
        raise ValueError(
            "shape of positive_weights and negative_weights "
            "must be the same! "
            "shape of positive_weights is {0}, "
            "but shape of negative_weights is {1}"
            % (positive_weights.size(), negative_weights.size())
        )

    # positive_term: Tensor [N, C] or [N, C, K]
    positive_term = (1 - logits).clamp(min=0) * labels
    negative_term = (1 + logits).clamp(min=0) * (1 - labels)

    if positive_weights_is_tensor and positive_term.dim() == 2:
        return (
            positive_term.unsqueeze(-1) * positive_weights
            + negative_term.unsqueeze(-1) * negative_weights
        )
    else:
        return positive_term * positive_weights + negative_term * negative_weights


class AUCPRHingeLoss(nn.Module):
    """area under the precision-recall curve loss,
    Reference: "Scalable Learning of Non-Decomposable Objectives", Section 5 \
    TensorFlow Implementation: \
    https://github.com/tensorflow/models/tree/master/research/global_objectives\
    """

    def __init__(self, weights=None, precision_range_lower = 0.0, precision_range_upper = 1.0, 
                 num_classes = 2, num_anchors = 20):
        """Args:
        config: Config containing `precision_range_lower`, `precision_range_upper`,
            `num_classes`, `num_anchors`
        """
        nn.Module.__init__(self)
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.precision_range = (
            precision_range_lower,
            precision_range_upper,
        )

        # Create precision anchor values and distance between anchors.
        # coresponding to [alpha_t] and [delta_t] in the paper.
        # precision_values: 1D `Tensor` of shape [K], where `K = num_anchors`
        # delta: Scalar (since we use equal distance between anchors)
        self.precision_values, self.delta = range_to_anchors_and_delta(
            self.precision_range, self.num_anchors
        )

        # notation is [b_k] in paper, Parameter of shape [C, K]
        # where `C = number of classes` `K = num_anchors`
        self.biases = nn.Parameter(
            torch.cuda.FloatTensor(num_classes, num_anchors).zero_()
        )
        self.lambdas = nn.Parameter(
            torch.cuda.FloatTensor(num_classes, num_anchors).data.fill_(
                1.0
            )
        )
    
    @property
    def __name__(self):
        return "AUPRLoss"
    
    def forward(self, logits, targets, reduce=True, size_average=True, weights=None):
        """
        Args:
            logits: Variable :math:`(N, C)` where `C = number of classes`
            targets: Variable :math:`(N)` where each value is
                `0 <= targets[i] <= C-1`
            weights: Coefficients for the loss. Must be a `Tensor` of shape
                [N] or [N, C], where `N = batch_size`, `C = number of classes`.
            size_average (bool, optional): By default, the losses are averaged
                    over observations for each minibatch. However, if the field
                    sizeAverage is set to False, the losses are instead summed
                    for each minibatch. Default: ``True``
            reduce (bool, optional): By default, the losses are averaged or summed over
                observations for each minibatch depending on size_average. When reduce
                is False, returns a loss per input/target element instead and ignores
                size_average. Default: True
        """
        C = 1 if logits.dim() == 1 else logits.size(1)

        # softmax with temperature, large temperater get smooth result
        # logits = torch.softmax(logits/5, 1)
        
        if self.num_classes != C:
            raise ValueError(
                "num classes is %d while logits width is %d" % (self.num_classes, C)
            )

        labels, weights = AUCPRHingeLoss._prepare_labels_weights(
            logits, targets, weights=weights
        )

        # Lagrange multipliers
        # Lagrange multipliers are required to be nonnegative.
        # Their gradient is reversed so that they are maximized
        # (rather than minimized) by the optimizer.
        # 1D `Tensor` of shape [K], where `K = num_anchors`
        lambdas = lagrange_multiplier(self.lambdas)
        # print("lambdas: {}".format(lambdas))

        # A `Tensor` of Shape [N, C, K]
        hinge_loss = weighted_hinge_loss(
            labels.unsqueeze(-1),
            logits.unsqueeze(-1) - self.biases,
            positive_weights=1.0 + lambdas * (1.0 - self.precision_values),
            negative_weights=lambdas * self.precision_values,
        )

        # 1D tensor of shape [C]
        class_priors = build_class_priors(labels, weights=weights)

        # lambda_term: Tensor[C, K]
        # according to paper, lambda_term = lambda * (1 - precision) * |Y^+|
        # where |Y^+| is number of postive examples = N * class_priors
        lambda_term = class_priors.unsqueeze(-1) * (
            lambdas * (1.0 - self.precision_values)
        )

        per_anchor_loss = weights.unsqueeze(-1) * hinge_loss - lambda_term

        # Riemann sum over anchors, and normalized by precision range
        # loss: Tensor[N, C]
        loss = per_anchor_loss.sum(2) * self.delta
        loss /= self.precision_range[1] - self.precision_range[0]

        if not reduce:
            return loss
        elif size_average:
            return loss.mean()
        else:
            return loss.sum()

    @staticmethod
    def _prepare_labels_weights(logits, targets, weights=None):
        """
        Args:
            logits: A tensor of shape [N, C] where `C = number of classes`
            targets: A tensor of shape [N]
            weights: Either `None` or a `Tensor` with shape broadcastable to `logits`
        Returns:
            labels: Tensor of shape [N, C], one-hot representation
            weights: Tensor of shape broadcastable to labels
        """
        N, C = logits.size()
        # Converts targets to one-hot representation. Dim: [N, C]
        labels = torch.cuda.FloatTensor(N, C).zero_().scatter(1, targets.unsqueeze(1).data, 1)
        if weights is None:
            weights = torch.cuda.FloatTensor(N, 1).data.fill_(1.0)

        return labels, weights
    
def weighted_sigmoid_cross_entropy_with_logits(labels,
                                               logits,
                                               positive_weights=1.0,
                                               negative_weights=1.0,
                                               name=None):
    """Computes a weighting of sigmoid cross entropy given `logits`.
    Measures the weighted probability error in discrete classification tasks in
    which classes are independent and not mutually exclusive.  For instance, one
    could perform multilabel classification where a picture can contain both an
    elephant and a dog at the same time. The class weight multiplies the
    different types of errors.
    For brevity, let `x = logits`, `z = labels`, `c = positive_weights`,
    `d = negative_weights`  The
    weighed logistic loss is
    ```
    c * z * -log(sigmoid(x)) + d * (1 - z) * -log(1 - sigmoid(x))
    = c * z * -log(1 / (1 + exp(-x))) - d * (1 - z) * log(exp(-x) / (1 + exp(-x)))
    = c * z * log(1 + exp(-x)) + d * (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
    = c * z * log(1 + exp(-x)) + d * (1 - z) * (x + log(1 + exp(-x)))
    = (1 - z) * x * d + (1 - z + c * z ) * log(1 + exp(-x))
    =  - d * x * z + d * x + (d - d * z + c * z ) * log(1 + exp(-x))
    ```
    To ensure stability and avoid overflow, the implementation uses the identity
      log(1 + exp(-x)) = max(0,-x) + log(1 + exp(-abs(x)))
    and the result is computed as
    ```
    = -d * x * z + d * x
      + (d - d * z + c * z ) * (max(0,-x) + log(1 + exp(-abs(x))))
    ```
    Note that the loss is NOT an upper bound on the 0-1 loss, unless it is divided
    by log(2).
    Args:
    labels: A `Tensor` of type `float32` or `float64`. `labels` can be a 2D
      tensor with shape [batch_size, num_labels] or a 3D tensor with shape
      [batch_size, num_labels, K].
    logits: A `Tensor` of the same type and shape as `labels`. If `logits` has
      shape [batch_size, num_labels, K], the loss is computed separately on each
      slice [:, :, k] of `logits`.
    positive_weights: A `Tensor` that holds positive weights and has the
      following semantics according to its shape:
        scalar - A global positive weight.
        1D tensor - must be of size K, a weight for each 'attempt'
        2D tensor - of size [num_labels, K'] where K' is either K or 1.
      The `positive_weights` will be expanded to the left to match the
      dimensions of logits and labels.
    negative_weights: A `Tensor` that holds positive weight and has the
      semantics identical to positive_weights.
    name: A name for the operation (optional).
    Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
      weighted logistic losses.
    """

    softplus_term = torch.add(torch.max(-logits, torch.zeros_like(logits)),
                           torch.log(1.0 + torch.exp(-torch.abs(logits))))
    weight_dependent_factor = (
        negative_weights + (positive_weights - negative_weights) * labels)
    return (negative_weights * (logits - labels * logits) +
            weight_dependent_factor * softplus_term)

def _prepare_labels_weights(logits, targets, weights=None):
    """
    Args:
        logits: A tensor of shape [N, C] where `C = number of classes`
        targets: A tensor of shape [N]
        weights: Either `None` or a `Tensor` with shape broadcastable to `logits`
    Returns:
        labels: Tensor of shape [N, C], one-hot representation
        weights: Tensor of shape broadcastable to labels
    """
    N, C = logits.size()
    # Converts targets to one-hot representation. Dim: [N, C]
    labels = torch.cuda.FloatTensor(N, C).zero_().scatter(1, targets.unsqueeze(1).data, 1)
    if weights is None:
        # Weights has shape [batch_size]. Reshape to [batch_size, 1].
        weights = torch.cuda.FloatTensor(N, 1).data.fill_(1.0)
    elif weights.dim() == 0:
        # Weights is a scalar. Change shape of weights to match logits.
        weights *= torch.ones_like(logits)
    return labels, weights

class AUROCLoss(nn.Module):
    def __init__(self, num_classes = 1):
        """ Computes ROC AUC loss == area above AUC.
        The area under the ROC curve is the probability p that a randomly chosen
        positive example will be scored higher than a randomly chosen negative
        example.
        This loss approximates 1-p by using a surrogate (either hinge loss or
        cross entropy) for the indicator function. Specifically, the loss is:
        sum_i sum_j w_i*w_j*loss(logit_i - logit_j)
        """
        nn.Module.__init__(self)
        self.num_classes = num_classes
 
    @property
    def __name__(self):
        return "AUROCLoss"
    
    def forward(self, logits, targets, weights=None, surrogate_type="xent"):
        """
        Args:
            logits: A tensor of shape [N, C] where `C = number of classes`
            targets: A tensor of shape [N] where each value is `0 <= targets[i] <= C-1`
            weights: Coefficients for the loss. Must be a `Tensor` of shape
                [N] or [N, C], where `N = batch_size`, `C = number of classes`.
            surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
                should be used for the indicator function.
            
        """
        C = 1 if logits.dim() == 1 else logits.size(1)

        if self.num_classes != C:
            raise ValueError(
                "num classes is %d while logits width is %d" % (self.num_classes, C)
            )
        labels, weights = _prepare_labels_weights(
            logits, targets, weights=weights
        )
        # Create tensors of pairwise differences for logits and labels, and
        # pairwise products of weights. These have shape [N, N, C].
        logits_difference = torch.unsqueeze(logits, 0) - torch.unsqueeze(logits, 1)
        labels_difference = torch.unsqueeze(labels, 0) - torch.unsqueeze(labels, 1)
        weights_product = torch.unsqueeze(weights, 0) * torch.unsqueeze(weights, 1) 
        
        signed_logits_difference = labels_difference * logits_difference
        if surrogate_type == "hinge":
            raw_loss = weighted_hinge_loss(
                labels=torch.ones_like(signed_logits_difference),
                logits=signed_logits_difference
            )
        elif surrogate_type == "xent":
            raw_loss = weighted_sigmoid_cross_entropy_with_logits(
                labels=torch.ones_like(signed_logits_difference),
                logits=signed_logits_difference
            )
        weighted_loss = weights_product * raw_loss
        # Zero out entries of the loss where labels_difference zero (so loss is only
        # computed on pairs with different labels).
        loss = (torch.abs(labels_difference) * weighted_loss).mean() * 0.5
        return loss
