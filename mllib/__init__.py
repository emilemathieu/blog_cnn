from .svm import (
    binary_classification_qp,
    binary_classification_smo,
    #multiclass_ova,
    multiclass_ovo,
    Kernel
)

from .optim import (
    SGD,
    Adam
)

from .nn import (
    Module,
    ReLU,
    MaxPool2d,
    Conv2d,
    Linear,
    Sequential
)

from .loss import (
    CrossEntropyLoss
)

from .kflearn import (
    Kmeans,
    extract_features
    )

from .tools import (
    whiten,
    standard
    )