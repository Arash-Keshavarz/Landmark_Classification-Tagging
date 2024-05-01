import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        self.model = nn.Sequential(
            # First Conv
            nn.Conv2d(3, 16, 3, padding=1),    #(3*224*224)
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),                #(16*112*112)
            nn.ReLU(),
            
            # Second Conv
            nn.Conv2d(16, 32, 3, padding=1),   #(16*112*112)
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),                #(32*56*56)
            nn.ReLU(),
            
            #Third Conv
            nn.Conv2d(32, 64, 3, padding=1),  #(32*56*56)
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),               #(64*28*28)
            nn.ReLU(),
            
            #Fourth Conv
            nn.Conv2d(64, 128, 3, padding=1),  #(64*28*28)
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),               #(128*14*14)
            nn.ReLU(),
            
            #Fifth Conv
            nn.Conv2d(128, 256, 3, padding=1),  #(128*14*14)
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),               #(256*7*7)
            nn.ReLU(),
            
            nn.Flatten(),
            
            nn.Linear(256*7*7, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # feature extractor, the pooling and the final linear
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from src.data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
