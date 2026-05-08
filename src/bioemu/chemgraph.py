# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import copy
from typing import Any

import torch
from torch_geometric.data import Data


class ChemGraph(Data):
    node_orientations: torch.Tensor  # [num_nodes, 3, 3] or [num_nodes, 3] when it's a score (since the scores are given as rotation vectors)
    pos: torch.Tensor  # [num_nodes, 3] score model expects this to be in nanometers.
    likelihood: torch.Tensor  # [num_nodes,] computed log likelihood for each node. This is the integral of the likelihood ODE when using flow sampling. The sum is the total likelihood of the sample.
    noise_likelihood: torch.Tensor  # [num_nodes,] harmonic noise energy for each node, used for likelihood computation
    edge_index: torch.Tensor  # [2, num_edges]
    single_embeds: torch.Tensor  # [num_nodes, EVOFORMER_NODE_DIM]
    pair_embeds: torch.Tensor  # [num_nodes**2, EVOFORMER_EDGE_DIM]
    sequence: str  # amino acid sequence of the protein
    system_id: str  # optional identifier for the protein sequence, e.g. a PDB ID.
    node_labels: torch.LongTensor  # [num_nodes,],  indicates residue type. Ignored unless `extra_residue_embeds` is True.

    def replace(self, **kwargs: Any) -> ChemGraph:
        """Returns a shallow copy of the ChemGraph with updated fields."""
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__["_store"] = copy.copy(self._store)
        for key, value in kwargs.items():
            out._store[key] = value
        out._store._parent = out
        return out
