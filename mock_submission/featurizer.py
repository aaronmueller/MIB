"""
Copy of the existing SubspaceFeaturizer implementation for submission.
This file provides the same SubspaceFeaturizer functionality in a self-contained format.
"""

import torch
import torch.nn as nn
import pyvene as pv
from CausalAbstraction.model_units.model_units import Featurizer


class SubspaceFeaturizerModuleCopy(torch.nn.Module):
    def __init__(self, rotate_layer):
        super().__init__()
        self.rotate = rotate_layer
        
    def forward(self, x):
        r = self.rotate.weight.T
        f = x.to(r.dtype) @ r.T
        error = x - (f @ r).to(x.dtype)
        return f, error


class SubspaceInverseFeaturizerModuleCopy(torch.nn.Module):
    def __init__(self, rotate_layer):
        super().__init__()
        self.rotate = rotate_layer
        
    def forward(self, f, error):
        r = self.rotate.weight.T
        return (f.to(r.dtype) @ r).to(f.dtype) + error.to(f.dtype)


class SubspaceFeaturizerCopy(Featurizer):
    def __init__(self, shape=None, rotation_subspace=None, trainable=True, id="subspace"):
        assert shape is not None or rotation_subspace is not None, "Either shape or rotation_subspace must be provided."
        if shape is not None:
            self.rotate = pv.models.layers.LowRankRotateLayer(*shape, init_orth=True)
        elif rotation_subspace is not None:
            shape = rotation_subspace.shape
            self.rotate = pv.models.layers.LowRankRotateLayer(*shape, init_orth=False)
            self.rotate.weight.data.copy_(rotation_subspace)
        self.rotate = torch.nn.utils.parametrizations.orthogonal(self.rotate)

        if not trainable:
            self.rotate.requires_grad_(False)

        # Create module-based featurizer and inverse_featurizer
        featurizer = SubspaceFeaturizerModuleCopy(self.rotate)
        inverse_featurizer = SubspaceInverseFeaturizerModuleCopy(self.rotate)
            
        super().__init__(featurizer, inverse_featurizer, n_features=self.rotate.weight.shape[1], id=id)