import torch
import torch.nn as nn
import torch.optim


def get_loss():
    """
    Get an instance of the CrossEntropyLoss (useful for classification),
    optionally moving it to the GPU if use_cuda is set to True
    """

    loss  = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss = loss.cuda()

    return loss

def get_optimizer(
    model: nn.Module,
    optimizer: str = "SGD",
    learning_rate: float = 0.01,
    momentum: float = 0.5,
    weight_decay: float = 0,
):
    """
    Returns an optimizer instance

    :param model: the model to optimize
    :param optimizer: one of 'SGD' or 'Adam'
    :param learning_rate: the learning rate
    :param momentum: the momentum (if the optimizer uses it)
    :param weight_decay: regularization coefficient
    """
    if optimizer.lower() == "sgd":
        
        # optimizer. Use the input parameters learning_rate, momentum
        # and weight_decay
        opt = torch.optim.SGD(
            model.parameters(),
            lr= learning_rate,
            momentum= momentum,
            weight_decay= weight_decay
        )

    elif optimizer.lower() == "adam":
        # optimizer. Use the input parameters learning_rate, momentum
        # and weight_decay
        opt = torch.optim.Adam(
            model.parameters(),
            lr= learning_rate,
            betas= (momentum, 0.999),
            weight_decay= weight_decay
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    return opt


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def fake_model():
    return nn.Linear(16, 256)


def test_get_loss():

    loss = get_loss()

    assert isinstance(
        loss, nn.CrossEntropyLoss
    ), f"Expected cross entropy loss, found {type(loss)}"


def test_get_optimizer_type(fake_model):

    opt = get_optimizer(fake_model)

    assert isinstance(opt, torch.optim.SGD), f"Expected SGD optimizer, got {type(opt)}"


def test_get_optimizer_is_linked_with_model(fake_model):

    opt = get_optimizer(fake_model)

    assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])


def test_get_optimizer_returns_adam(fake_model):

    opt = get_optimizer(fake_model, optimizer="adam")

    assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])
    assert isinstance(opt, torch.optim.Adam), f"Expected SGD optimizer, got {type(opt)}"


def test_get_optimizer_sets_learning_rate(fake_model):

    opt = get_optimizer(fake_model, optimizer="adam", learning_rate=0.123)

    assert (
        opt.param_groups[0]["lr"] == 0.123
    ), "get_optimizer is not setting the learning rate appropriately. Check your code."


def test_get_optimizer_sets_momentum(fake_model):

    opt = get_optimizer(fake_model, optimizer="SGD", momentum=0.123)

    assert (
        opt.param_groups[0]["momentum"] == 0.123
    ), "get_optimizer is not setting the momentum appropriately. Check your code."


def test_get_optimizer_sets_weight_decat(fake_model):

    opt = get_optimizer(fake_model, optimizer="SGD", weight_decay=0.123)

    assert (
        opt.param_groups[0]["weight_decay"] == 0.123
    ), "get_optimizer is not setting the weight_decay appropriately. Check your code."
