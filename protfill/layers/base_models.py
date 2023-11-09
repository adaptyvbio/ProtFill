from abc import abstractmethod

from torch import nn


class Encoder(nn.Module):
    @abstractmethod
    def forward(
        self, h_V, h_E, E_idx, mask, X, residue_idx, chain_encoding_all, global_context
    ):
        """
        Returns
        -------
        h_V : torch.Tensor
            the node features `(B, N, F1)`
        h_E : torch.Tensor
            the edge features `(B, N, E, F2)`
        X : torch.Tensor
            the coordinates `(B, N, 4, 3)`
        E_idx : torch.Tensor
            the neighbors `(B, N, E)`
        """

        ...


class Decoder_AR(nn.Module):
    @abstractmethod
    def forward(self, h_V, h_E, h_S, E_idx, mask, X, chain_M):
        """
        Returns
        -------
        h_V : torch.Tensor
            the node features `(B, N, F1)`
        h_E : torch.Tensor
            the edge features `(B, N, E, F2)`
        X : torch.Tensor
            the coordinates `(B, N, 4, 3)`
        E_idx : torch.Tensor
            the neighbors `(B, N, E)`
        """

        ...


class Decoder(nn.Module):
    @abstractmethod
    def forward(
        self, h_V, h_E, E_idx, mask, X, residue_idx, chain_encoding_all, global_context
    ):
        """
        Returns
        -------
        h_V : torch.Tensor
            the node features `(B, N, F1)`
        h_E : torch.Tensor
            the edge features `(B, N, E, F2)`
        X : torch.Tensor
            the coordinates `(B, N, 4, 3)`
        E_idx : torch.Tensor
            the neighbors `(B, N, E)`
        """

        ...
