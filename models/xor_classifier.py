from flax import linen as nn
from typing import Any


class SimpleClassifier(nn.Module):
    num_hidden: int  # Number of hidden neurons
    num_outputs: int  # Number of output neurons

    def setup(self):
        # Create the modules we need to build the network
        # nn.Dense is a linear layer
        # import pdb

        # pdb.set_trace()
        self.linear1 = nn.Dense(features=self.num_hidden)
        self.linear2 = nn.Dense(features=self.num_outputs)

    def __call__(self, x):
        # Perform the calculation of the model to determine the prediction
        # import pdb

        # pdb.set_trace()
        x = self.linear1(x)
        x = nn.tanh(x)
        x = self.linear2(x)

        return x
