import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from transformers import AutoModel

from utils import _MCMEDResBlock1D

def _init_linear_near_identity_(layer: nn.Linear, noise_scale: float = 1e-3) -> None:
    """
    Initialize a Linear layer as the closest identity-like map allowed by its
    shape, plus tiny noise.
    """
    if not isinstance(layer, nn.Linear):
        raise TypeError("Expected nn.Linear in _init_linear_near_identity_.")

    with torch.no_grad():
        layer.weight.zero_()
        k = min(layer.out_features, layer.in_features)
        layer.weight[:k, :k] = torch.eye(
            k, device=layer.weight.device, dtype=layer.weight.dtype
        )
        if noise_scale > 0.0:
            layer.weight.add_(noise_scale * torch.randn_like(layer.weight))
        if layer.bias is not None:
            layer.bias.zero_()


"""
Symile-MIMIC 
"""
class CXREncoder(nn.Module):
    def __init__(
        self,
        resnet_params: dict = {
            "weights": None,  # "IMAGENET1K_V2"
            "norm_type": "batchnorm",
        },
        emb_dim: int = 8192,
        geometry_preserving: bool = False,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.geometry_preserving = bool(geometry_preserving)

        self.resnet = models.resnet50(weights=resnet_params["weights"])
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, emb_dim, bias=True)

        nn.init.kaiming_normal_(self.resnet.fc.weight, mode="fan_out")
        nn.init.zeros_(self.resnet.fc.bias)
        #if self.geometry_preserving:
        #    self.init_near_identity_()

    def init_near_identity_(self, noise_scale: float = 1e-3) -> None:
        """
        Make the final projection head of the ResNet start as a near-identity
        padded/truncated map.
        """
        _init_linear_near_identity_(self.resnet.fc, noise_scale=noise_scale)

    def forward(self, x):
        x = self.resnet(x)
        return x

class ECGEncoder(nn.Module):
    def __init__(
        self, 
        resnet_params: dict = {
            "weights": None,  # "IMAGENET1K_V1"
            "norm_type": "batchnorm",
        },
        emb_dim: int = 8192,
        geometry_preserving: bool = False,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.geometry_preserving = bool(geometry_preserving)

        self.resnet = models.resnet18(weights=resnet_params["weights"])
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, emb_dim, bias=True)

        nn.init.kaiming_normal_(self.resnet.fc.weight, mode="fan_out")
        nn.init.zeros_(self.resnet.fc.bias)
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode="fan_out")
        #if self.geometry_preserving:
        #    self.init_near_identity_()

    def init_near_identity_(self, noise_scale: float = 1e-3) -> None:
        """
        Make the final projection head of the ResNet start as a near-identity
        padded/truncated map.
        """
        _init_linear_near_identity_(self.resnet.fc, noise_scale=noise_scale)

    def forward(self, x):
        x = self.resnet(x)
        return x

class LabsEncoder(nn.Module):
    def __init__(
        self,
        emb_dim: int = 8192,
        geometry_preserving: bool = False,
        leaky_relu_negative_slope: float = 0.0,
    ):
        super().__init__()
        self.input_dim = 100
        self.emb_dim = emb_dim
        self.geometry_preserving = bool(geometry_preserving)
        
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, emb_dim)
        #self.act = nn.LeakyRelu(negative_slope=leaky_relu_negative_slope)
        self.act = nn.GELU()

        if self.geometry_preserving:
            self.residual_proj = nn.Linear(self.input_dim, emb_dim)
            self.residual_norm = nn.LayerNorm(emb_dim)
            self.residual_drop = nn.Dropout(0.0)
        else:
            self.residual_proj = None
            self.residual_norm = None
            self.residual_drop = None

        self.apply(self._init_weights)
        if self.geometry_preserving:
            self.init_residual_identity_()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.kaiming_normal_(m.weight, mode="fan_out")
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_residual_identity_(self, branch_scale: float = 1e-3) -> None:
        if not isinstance(self.residual_proj, nn.Linear):
            raise TypeError("LabsEncoder.residual_proj is expected to be a Linear layer.")

        with torch.no_grad():
            proj = self.residual_proj
            proj.weight.zero_()
            k = min(proj.out_features, proj.in_features)
            proj.weight[:k, :k] = branch_scale * torch.eye(
                k, device=proj.weight.device, dtype=proj.weight.dtype
            )
            if proj.bias is not None:
                proj.bias.zero_()

    def forward(self, x):
        x_in = x
        x = self.fc1(x_in)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        if self.geometry_preserving:
            residual = self.residual_drop(self.residual_norm(self.residual_proj(x_in)))
            return residual + x
        return x

class CXREncoder_EF(nn.Module):
    def __init__(
        self,
        resnet_params: dict = {
            "weights": None,  # "IMAGENET1K_V2"
            "norm_type": "batchnorm",
        },
        emb_dim: int = 8192
    ):
        super().__init__()
        self.emb_dim = emb_dim

        from utils import PatchEncoder_CXR
        self.patch_encoder = PatchEncoder_CXR(
            image_size=320,
            patch_size=64,
            in_channels=3,
            emb_dim=emb_dim,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_encoder(x)
        return x

class ECGEncoder_EF(nn.Module):
    def __init__(
        self, 
        resnet_params: dict = {
            "weights": None,  # "IMAGENET1K_V1"
            "norm_type": "batchnorm",
        },
        emb_dim: int = 8192
    ):
        super().__init__()
        self.emb_dim = emb_dim

        from utils import PatchEncoder_ECG
        self.patch_encoder = PatchEncoder_ECG(
            input_size=(5000, 12),
            patch_size=(250, 12),
            in_channels=1,
            emb_dim=emb_dim,
        )
    
        self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_encoder(x)
        return x

class LabsEncoder_EF(nn.Module):
    def __init__(
        self,
        emb_dim: int = 8192,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        
        self.fc = nn.Linear(100, emb_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.kaiming_normal_(m.weight, mode="fan_out")
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc(x)
        return x


class CXREncoder_Hybrid(nn.Module):
    def __init__(
        self,
        resnet_params: dict = {
            "weights": None,
            "norm_type": "batchnorm",
        },
        emb_dim: int = 8192
    ):
        super().__init__()
        self.emb_dim = emb_dim

        self.resnet = models.resnet50(weights=resnet_params["weights"])
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, emb_dim, bias=True)
        from utils import PatchEncoder_CXR
        self.residual_branch = PatchEncoder_CXR(
            image_size=320,
            patch_size=32,
            in_channels=3,
            emb_dim=emb_dim,
            num_tokens=None,
        )
        self.residual_branch_norm = nn.LayerNorm(emb_dim)
        self.residual_drop = nn.Dropout(0.0)
        nn.init.kaiming_normal_(self.resnet.fc.weight, mode="fan_out")
        nn.init.zeros_(self.resnet.fc.bias)
        self.apply(self._init_weights)
        self.init_residual_identity_()

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_residual_identity_(self, branch_scale: float = 1e-3) -> None:
        with torch.no_grad():
            proj = self.residual_branch.proj
            proj.weight.zero_()
            k = min(proj.out_features, proj.in_features)
            proj.weight[:k, :k] = branch_scale * torch.eye(
                k, device=proj.weight.device, dtype=proj.weight.dtype
            )
            if proj.bias is not None:
                proj.bias.zero_()

    def forward(self, x):
        strong = self.resnet(x)
        residual_tokens = self.residual_branch(x)
        strong = strong[:, None, :]
        residual_tokens = self.residual_drop(self.residual_branch_norm(residual_tokens))
        return torch.cat([strong, residual_tokens], dim=1)

class ECGEncoder_Hybrid(nn.Module):
    def __init__(
        self,
        resnet_params: dict = {
            "weights": None,
            "norm_type": "batchnorm",
        },
        emb_dim: int = 8192
    ):
        super().__init__()
        self.emb_dim = emb_dim

        self.resnet = models.resnet18(weights=resnet_params["weights"])
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, emb_dim, bias=True)
        from utils import PatchEncoder_ECG
        self.residual_branch = PatchEncoder_ECG(
            input_size=(5000, 12),
            patch_size=(125, 12),
            in_channels=1,
            emb_dim=emb_dim,
            num_tokens=None,
        )
        self.residual_branch_norm = nn.LayerNorm(emb_dim)
        self.residual_drop = nn.Dropout(0.0)
        nn.init.kaiming_normal_(self.resnet.fc.weight, mode="fan_out")
        nn.init.zeros_(self.resnet.fc.bias)
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode="fan_out")
        self.apply(self._init_weights)
        self.init_residual_identity_()

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_residual_identity_(self, branch_scale: float = 1e-3) -> None:
        with torch.no_grad():
            proj = self.residual_branch.proj
            proj.weight.zero_()
            k = min(proj.out_features, proj.in_features)
            proj.weight[:k, :k] = branch_scale * torch.eye(
                k, device=proj.weight.device, dtype=proj.weight.dtype
            )
            if proj.bias is not None:
                proj.bias.zero_()

    def forward(self, x):
        strong = self.resnet(x)
        residual_tokens = self.residual_branch(x)
        strong = strong[:, None, :]
        residual_tokens = self.residual_drop(self.residual_branch_norm(residual_tokens))
        return torch.cat([strong, residual_tokens], dim=1)

class LabsEncoder_Hybrid(nn.Module):
    def __init__(
        self,
        emb_dim: int = 8192,
    ):
        super().__init__()
        self.input_dim = 100
        self.emb_dim = emb_dim

        self.strong_encoder = nn.Sequential(
            nn.Linear(self.input_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.residual_proj = nn.Linear(self.input_dim, emb_dim)
        self.residual_norm = nn.LayerNorm(emb_dim)
        self.residual_drop = nn.Dropout(0.0)
        self.apply(self._init_weights)
        self.init_residual_identity_()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_residual_identity_(self, branch_scale: float = 1e-3) -> None:
        if not isinstance(self.residual_proj, nn.Linear):
            raise TypeError("LabsEncoder_Residual.residual_proj is expected to be a Linear layer.")
        if not all(isinstance(self.strong_encoder[idx], nn.Linear) for idx in (0, 2, 4)):
            raise TypeError("LabsEncoder.strong_encoder does not have the expected Linear/Act/Linear/Act/Linear structure.")

        with torch.no_grad():
            proj = self.residual_proj
            proj.weight.zero_()
            k = min(proj.out_features, proj.in_features)
            proj.weight[:k, :k] = torch.eye(k, device=proj.weight.device, dtype=proj.weight.dtype)
            if proj.bias is not None:
                proj.bias.zero_()

            # Keep the residual branch near zero while leaving the strong branch
            # free to learn normally.
            if proj.weight.shape[0] > 0 and proj.weight.shape[1] > 0:
                proj.weight[:k, :k] = branch_scale * torch.eye(
                    k, device=proj.weight.device, dtype=proj.weight.dtype
                )

    def forward(self, x):
        strong = self.strong_encoder(x)
        residual = self.residual_drop(self.residual_norm(self.residual_proj(x)))
        return strong + residual



""" 
Symile-M3
"""
class AudioEncoder(nn.Module):
    def __init__(
        self, 
        input_dim: int = 1280,
        emb_dim: int = 8192,
    ):
        super().__init__()
        
        self.fc = nn.Linear(input_dim, emb_dim, bias=True)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, audio_embed, missingness_ind: int = 0):
        x = self.fc(audio_embed)
        x = self.layer_norm(x)
        return x

class ImageEncoder(nn.Module):
    def __init__(
        self, 
        input_dim: int = 1024, 
        emb_dim: int = 8192,
    ):
        super().__init__()

        self.fc = nn.Linear(input_dim, emb_dim, bias=True)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, image_embed, missingness_ind: int = 0):
        x = self.fc(image_embed)
        x = self.layer_norm(x)
        return x

class TextEncoder(nn.Module):
    def __init__(
        self, 
        model_params: dict = {
            "text_model_id": "xlm-roberta-large",
        },
        emb_dim: int = 8192,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_params["text_model_id"])

        self.embeddings = self.encoder.embeddings
        self.encoder_layer = self.encoder.encoder.layer[0]

        # first freeze all parameters, then unfreeze relevant parameters
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.embeddings.parameters():
            p.requires_grad = True
        for p in self.encoder_layer.parameters():
            p.requires_grad = True

        self.fc = nn.Linear(1024, emb_dim, bias=True)
        self.layer_norm = nn.LayerNorm(emb_dim)


    def forward(self, x):
        # https://github.com/huggingface/transformers/blob/a0857740c0e6127485c11476650314df3accc2b6/src/transformers/modeling_utils.py#L941
        # attention mask has shape (batch_sz, seq_len)
        # we make the mask broadcastable to (batch_sz, num_heads, seq_len, seq_len)
        extended_attention_mask = x["attention_mask"][:, None, None, :]
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.encoder.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.encoder.dtype).min

        embedding_output = self.embeddings(x["input_ids"])
        encoder_outputs = self.encoder_layer(embedding_output, attention_mask=extended_attention_mask)
        x = encoder_outputs[0]
        x = self.fc(x)
        x = x.mean(dim=1)
        x = self.layer_norm(x)
        return x



""" 
UK Biobank
"""
class UKBTabularEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list[int] = [512, 1024, 512],
        hidden_dropouts: list[float] = [0.1, 0.1, 0.1],
        emb_dim: int = 8192,
        combine_eids_as: str = "intersect",
        shared_adapter: nn.Module = None,
        modality_name: str = None,
        geometry_preserving: bool = False,
        leaky_relu_negative_slope: float = 0.0,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.emb_dim = emb_dim
        self.modality_name = modality_name
        self.geometry_preserving = geometry_preserving

        self.combine_eids_as = combine_eids_as
        if self.combine_eids_as == "union":
            # also pass binary missing mask as input to the MLP
            input_dim = input_dim * 2
        if geometry_preserving:
            self.residual_proj = nn.Linear(input_dim, emb_dim, bias=True)
            self.residual_drop = nn.Dropout(0.0)  # float(hidden_dropouts[0] / 2)
            self.residual_norm = nn.LayerNorm(emb_dim)

        layers = []
        prev = input_dim
        for hidden_dim, hidden_dropout in zip(hidden_dims, hidden_dropouts):
            layers.append(nn.Linear(prev, hidden_dim, bias=True))
            layers.append(nn.LeakyReLU(leaky_relu_negative_slope))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(hidden_dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, emb_dim, bias=True))
        self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        if geometry_preserving:
            # self.init_near_identity_(noise_scale=1e-2)
            self.init_residual_identity_()

        if shared_adapter is not None:
            self.mlp = nn.Sequential(
                self.mlp,
                shared_adapter,
            )
            self.shared_adapter = shared_adapter

    def _init_weights(
        self,
        m
    ) -> None: 
        if isinstance(m, (torch.nn.LayerNorm)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def init_residual_identity_(self, branch_scale: float = 1e-3) -> None:
        """
        Initialize the encoder so the residual shortcut carries an identity-like
        map and the nonlinear MLP branch starts near zero.
        """
        if not isinstance(self.residual_proj, nn.Linear):
            raise TypeError("UKBTabularEncoder_MLP.residual_proj is expected to be a Linear layer.")

        linears = [m for m in self.mlp if isinstance(m, nn.Linear)]
        if len(linears) == 0:
            raise TypeError("UKBTabularEncoder_MLP.mlp does not contain any Linear layers.")

        with torch.no_grad():
            proj = self.residual_proj
            proj.weight.zero_()
            k = min(proj.out_features, proj.in_features)
            proj.weight[:k, :k] = torch.eye(k, device=proj.weight.device, dtype=proj.weight.dtype)
            if proj.bias is not None:
                proj.bias.zero_()

            for layer in linears:
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()

            first = linears[0]
            last = linears[-1]
            k_first = min(first.out_features, first.in_features)
            k_last = min(last.out_features, last.in_features)
            first.weight[:k_first, :k_first] = branch_scale * torch.eye(
                k_first, device=first.weight.device, dtype=first.weight.dtype
            )
            last.weight[:k_last, :k_last] = branch_scale * torch.eye(
                k_last, device=last.weight.device, dtype=last.weight.dtype
            )

    def init_near_identity_(self, noise_scale: float = 1e-3) -> None:
        """
        Initialize the plain labs MLP as a near-identity stack while keeping
        the GELU activations in place.
        """
        _init_linear_near_identity_(self.mlp[0], noise_scale=noise_scale)
        _init_linear_near_identity_(self.mlp[4], noise_scale=noise_scale)
        _init_linear_near_identity_(self.mlp[8], noise_scale=noise_scale)
        _init_linear_near_identity_(self.mlp[12], noise_scale=noise_scale)

    def forward(self, x):
        if self.combine_eids_as == "union":
            nanmask = torch.isnan(x).float()
            x = torch.nan_to_num(x, nan=0.0)
            x = torch.cat([x, nanmask], dim=1)
        if torch.isnan(x).any():
            # raise ValueError("NaN values present in input")
            x = torch.nan_to_num(x, nan=0.0)
        
        if self.geometry_preserving:
            residual = self.residual_drop(self.residual_norm(self.residual_proj(x)))
            return residual + self.mlp(x)
        return self.mlp(x)

class UKBTabularEncoder_EF(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list[int] = [512, 1024, 512],
        hidden_dropouts: list[float] = [0.1, 0.1, 0.1],
        emb_dim: int = 8192,
        combine_eids_as: str = "intersect",
        shared_adapter: nn.Module = None,
        modality_name: str = None,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.modality_name = modality_name

        self.combine_eids_as = combine_eids_as
        if self.combine_eids_as == "union":
            # also pass binary missing mask as input to the MLP
            input_dim = input_dim * 2

        proj_dropout = float(max(hidden_dropouts)) if len(hidden_dropouts) > 0 else 0.0
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.Dropout(proj_dropout),
        )

        self.apply(self._init_weights)

        if shared_adapter is not None:
            self.mlp = nn.Sequential(
                self.mlp,
                shared_adapter,
            )
            self.shared_adapter = shared_adapter

    def _init_weights(
        self,
        m
    ) -> None: 
        if isinstance(m, (torch.nn.LayerNorm)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.combine_eids_as == "union":
            nanmask = torch.isnan(x).float()
            x = torch.nan_to_num(x, nan=0.0)
            x = torch.cat([x, nanmask], dim=1)
        if torch.isnan(x).any():
            # raise ValueError("NaN values present in input")
            x = torch.nan_to_num(x, nan=0.0)

        return self.mlp(x)


"""
MIMIC (radiology reports + time-series + demographics)
"""
class MimicRadBERTEncoder(nn.Module):
    def __init__(
        self,
        model_params: dict = {
            "text_model_id": "StanfordAIMI/RadBERT",
            "proj_hidden_dim": 512,
            "dropout": 0.0,
            "lora": False,
        },
        emb_dim: int = 8192,
        geometry_preserving: bool = False,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        text_model_id = str(model_params.get("text_model_id", "StanfordAIMI/RadBERT"))
        self.text_encoder = AutoModel.from_pretrained(text_model_id)
        self.input_dim = int(self.text_encoder.config.hidden_size)
        self.geometry_preserving = bool(geometry_preserving)
        proj_hidden_dim = int(model_params.get("proj_hidden_dim", 512))
        dropout = float(model_params.get("dropout", 0.0))

        for p in self.text_encoder.parameters():
            p.requires_grad = False
        if model_params["lora"]:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=8,
                lora_alpha=8,
                lora_dropout=0.1,
                target_modules=["query", "value"],
                bias="none",
            )
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)

        if self.geometry_preserving:
            self.residual_proj = nn.Linear(self.input_dim, self.emb_dim)
            self.residual_dropout = nn.Dropout(dropout)
        else:
            self.residual_proj = None
            self.residual_dropout = None

        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, proj_hidden_dim),
            nn.LeakyReLU(negative_slope=0.0),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.LeakyReLU(negative_slope=0.0),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden_dim, self.emb_dim),
        )
        self.proj.apply(self._init_weights)
        if self.geometry_preserving:
            self.residual_proj.apply(self._init_weights)
            self.init_residual_identity_()

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def init_residual_identity_(self, branch_scale: float = 1e-3) -> None:
        if not isinstance(self.residual_proj, nn.Linear):
            raise TypeError("MimicRadBERTEncoder.residual_proj is expected to be a Linear layer.")

        linears = [m for m in self.proj if isinstance(m, nn.Linear)]
        with torch.no_grad():
            self.residual_proj.weight.zero_()
            k = min(self.residual_proj.out_features, self.residual_proj.in_features)
            self.residual_proj.weight[:k, :k] = torch.eye(
                k,
                device=self.residual_proj.weight.device,
                dtype=self.residual_proj.weight.dtype,
            )
            if self.residual_proj.bias is not None:
                self.residual_proj.bias.zero_()

            for layer in linears:
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()
            linears[0].weight[: min(linears[0].out_features, linears[0].in_features), : min(linears[0].out_features, linears[0].in_features)] = branch_scale * torch.eye(
                min(linears[0].out_features, linears[0].in_features),
                device=linears[0].weight.device,
                dtype=linears[0].weight.dtype,
            )
            linears[-1].weight[: min(linears[-1].out_features, linears[-1].in_features), : min(linears[-1].out_features, linears[-1].in_features)] = branch_scale * torch.eye(
                min(linears[-1].out_features, linears[-1].in_features),
                device=linears[-1].weight.device,
                dtype=linears[-1].weight.dtype,
            )

    def forward(self, x):
        if torch.is_tensor(x):
            pooled = x
        elif isinstance(x, dict) and "pooled_features" in x:
            pooled = x["pooled_features"]
        elif isinstance(x, dict):
            input_ids = x["input_ids"]
            attention_mask = x["attention_mask"].to(dtype=torch.long)
            out = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state
            weights = attention_mask.to(dtype=out.dtype).unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1.0)
            pooled = (out * weights).sum(dim=1) / denom
        else:
            raise TypeError(
                "MimicRadBERTEncoder expects tokenized dict input, pooled feature dict, or pooled tensor."
            )
        if self.geometry_preserving:
            return self.residual_dropout(self.residual_proj(pooled)) + self.proj(pooled)
        return self.proj(pooled)

class MimicMLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [512, 512],
        hidden_dropouts: list[float] = [0.1, 0.1],
        emb_dim: int = 8192,
        geometry_preserving: bool = False,
        leaky_relu_negative_slope: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.emb_dim = int(emb_dim)
        self.geometry_preserving = bool(geometry_preserving)

        layers = []
        prev = self.input_dim
        for hidden_dim, hidden_dropout in zip(hidden_dims, hidden_dropouts):
            layers.append(nn.Linear(prev, hidden_dim, bias=True))
            layers.append(nn.LeakyReLU(leaky_relu_negative_slope))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(hidden_dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, self.emb_dim, bias=True))
        self.mlp = nn.Sequential(*layers)

        if self.geometry_preserving:
            self.residual_proj = nn.Linear(self.input_dim, self.emb_dim, bias=True)
            self.residual_drop = nn.Dropout(0.0)
            self.residual_norm = nn.LayerNorm(self.emb_dim)
        else:
            self.residual_proj = None
            self.residual_drop = None
            self.residual_norm = None

        self.apply(self._init_weights)
        if self.geometry_preserving:
            self.init_residual_identity_()

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def init_residual_identity_(self, branch_scale: float = 1e-3) -> None:
        if not isinstance(self.residual_proj, nn.Linear):
            raise TypeError("MimicMLPEncoder.residual_proj is expected to be a Linear layer.")
        linears = [m for m in self.mlp if isinstance(m, nn.Linear)]
        with torch.no_grad():
            self.residual_proj.weight.zero_()
            k = min(self.residual_proj.out_features, self.residual_proj.in_features)
            self.residual_proj.weight[:k, :k] = torch.eye(
                k,
                device=self.residual_proj.weight.device,
                dtype=self.residual_proj.weight.dtype,
            )
            if self.residual_proj.bias is not None:
                self.residual_proj.bias.zero_()

            for layer in linears:
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()
            first = linears[0]
            last = linears[-1]
            k_first = min(first.out_features, first.in_features)
            k_last = min(last.out_features, last.in_features)
            first.weight[:k_first, :k_first] = branch_scale * torch.eye(
                k_first, device=first.weight.device, dtype=first.weight.dtype
            )
            last.weight[:k_last, :k_last] = branch_scale * torch.eye(
                k_last, device=last.weight.device, dtype=last.weight.dtype
            )

    def forward(self, x):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        if self.geometry_preserving:
            residual = self.residual_drop(self.residual_norm(self.residual_proj(x)))
            return residual + self.mlp(x)
        return self.mlp(x)


"""
Synthetic XNOR
"""
class SyntheticXNOREncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        emb_dim: int = 8192,
        geometry_preserving: str = False,
        input_layout: str = "1d",
        bounded_svd_params: dict = {
            "sv_min": 1.0,
            "sv_max": 2.0,
            "learnable_singular_values": True,
        },
        activation_fn_params: dict = {
            "leaky_relu_negative_slope": 0.01,
        },
        sequence_encoder_params: dict = {
            "nhead": 4,
            "num_layers": 2,
            "dropout": 0.0,
            "dim_feedforward_mult": 4,
            "residuals": True,
            "residual_alpha": 1.0,
            "layer_type": "transformer",
        },
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.emb_dim = emb_dim
        self.geometry_preserving = geometry_preserving
        self.input_layout = str(input_layout)

        self.input_proj = nn.Linear(input_dim, emb_dim)

        if self.geometry_preserving == "residual":
            self.residual_proj = nn.Linear(input_dim, emb_dim)
        if self.geometry_preserving == "bounded_svd_linear":
            from utils import BoundedSVDLinear
            linear_cls = BoundedSVDLinear

        if self.input_layout == "1d":
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, emb_dim),  # , **bounded_svd_params
                nn.LeakyReLU(activation_fn_params["leaky_relu_negative_slope"]),  # ReLU
                #nn.ReLU(),
                #nn.Identity(),
                nn.Linear(emb_dim, emb_dim),  # , **bounded_svd_params
                nn.LeakyReLU(activation_fn_params["leaky_relu_negative_slope"]),  # ReLU
                #nn.ReLU(),
                #nn.Identity(),
                nn.Linear(emb_dim, emb_dim)  # , **bounded_svd_params
            )
            self.apply(self._init_weights)
        elif self.input_layout == "2d":
            nhead = sequence_encoder_params["nhead"]
            num_layers = sequence_encoder_params["num_layers"]
            dropout = 0.0
            ff_mult = sequence_encoder_params["dim_feedforward_mult"]
            seq_len = sequence_encoder_params["seq_len"]
            residuals_enabled = bool(sequence_encoder_params.get("residuals", True))
            residual_alpha = float(sequence_encoder_params.get("residual_alpha", 1.0))
            layer_type = str(sequence_encoder_params.get("layer_type", "transformer"))
            self.seq_len = seq_len

            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, emb_dim))
            nn.init.normal_(self.pos_embed, std=0.02)

            self.seq_input_proj = nn.Linear(self.input_dim, self.emb_dim)
            if layer_type == "transformer":
                if residuals_enabled:
                    from utils import ScaledResidualTransformerEncoderLayer
                    encoder_layer = ScaledResidualTransformerEncoderLayer(
                        d_model=self.emb_dim,
                        nhead=nhead,
                        dim_feedforward=max(self.emb_dim, ff_mult * self.emb_dim),
                        dropout=dropout,
                        batch_first=True,
                        norm_first=True,
                        activation="relu",
                        residual_alpha=residual_alpha,
                    )
                else:
                    from utils import NoResidualTransformerEncoderLayer
                    encoder_layer = NoResidualTransformerEncoderLayer(
                        d_model=self.emb_dim,
                        nhead=nhead,
                        dim_feedforward=max(self.emb_dim, ff_mult * self.emb_dim),
                        dropout=dropout,
                        batch_first=True,
                        norm_first=True,
                        activation="relu",
                    )
            elif layer_type == "tokenwise_mlp":
                from utils import TokenwiseMLPEncoderLayer
                encoder_layer = TokenwiseMLPEncoderLayer(
                    d_model=self.emb_dim,
                    nhead=nhead,
                    dim_feedforward=max(self.emb_dim, ff_mult * self.emb_dim),
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="relu",
                )
            elif layer_type == "uniform_attention":
                from utils import UniformAttentionEncoderLayer
                encoder_layer = UniformAttentionEncoderLayer(
                    d_model=self.emb_dim,
                    nhead=nhead,
                    dim_feedforward=max(self.emb_dim, ff_mult * self.emb_dim),
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="relu",
                )
            else:
                raise ValueError(
                    f"Unknown sequence_encoder_params.layer_type: {layer_type}. "
                    "Expected one of ['transformer', 'tokenwise_mlp', 'uniform_attention']."
                )

            self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.seq_out_proj = nn.Linear(self.emb_dim, self.emb_dim)

            self.apply(self._init_weights)

            #for layer in self.seq_encoder.layers:
            #    layer.self_attn.out_proj.weight.data.mul_(1e-3)
            #    layer.self_attn.out_proj.bias.data.zero_()
            #    layer.linear2.weight.data.mul_(1e-3)
            #    layer.linear2.bias.data.zero_()

        #self.apply_burkholz_relu_init(self.mlp, mode="orthogonal")
        if self.geometry_preserving == "residual":
            self.init_residual_identity_()
    
    def _init_weights(
        self,
        m
    ) -> None: 
        if isinstance(m, (torch.nn.LayerNorm)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    @staticmethod
    def _init_linear_near_identity_(
        layer: nn.Linear,
        noise_scale: float = 1e-3,
    ) -> None:
        """
        Initialize a Linear layer as the closest identity-like map allowed by
        its shape, plus a tiny perturbation.

        For non-square layers this becomes a padded/truncated identity. This is
        useful when input_dim != emb_dim but we still want to preserve input
        geometry as much as possible at initialization.
        """
        if not isinstance(layer, nn.Linear):
            raise TypeError("Expected nn.Linear in _init_linear_near_identity_.")

        with torch.no_grad():
            layer.weight.zero_()
            k = min(layer.out_features, layer.in_features)
            layer.weight[:k, :k] = torch.eye(
                k, device=layer.weight.device, dtype=layer.weight.dtype
            )
            if noise_scale > 0.0:
                layer.weight.add_(noise_scale * torch.randn_like(layer.weight))
            if layer.bias is not None:
                layer.bias.zero_()

    def init_identity_(self) -> None:
        """
        Reconfigure the MLP to be an identity-like map.

        The ReLU activations are replaced by Identity modules. The first Linear
        layer is initialized as a padded/truncated identity when
        input_dim != emb_dim, while the remaining square Linear layers are set
        to exact identity.
        """
        if not isinstance(self.mlp[0], nn.Linear) or not isinstance(self.mlp[2], nn.Linear) or not isinstance(self.mlp[4], nn.Linear):
            raise TypeError("SyntheticXNOREncoder.mlp does not have the expected Linear/ReLU/Linear/ReLU/Linear structure.")

        self.mlp[1] = nn.Identity()
        self.mlp[3] = nn.Identity()

        with torch.no_grad():
            first = self.mlp[0]
            first.weight.zero_()
            k = min(first.out_features, first.in_features)
            first.weight[:k, :k] = torch.eye(k, device=first.weight.device, dtype=first.weight.dtype)
            if first.bias is not None:
                first.bias.zero_()

            for idx in (2, 4):
                layer = self.mlp[idx]
                layer.weight.zero_()
                layer.weight.add_(torch.eye(layer.out_features, device=layer.weight.device, dtype=layer.weight.dtype))
                if layer.bias is not None:
                    layer.bias.zero_()

    def init_near_identity_(self, noise_scale: float = 1e-3) -> None:
        """
        Initialize the plain MLP path as a near-identity map.

        This keeps the ReLU activations in place, but makes every Linear layer
        start as the closest identity-like transform allowed by its shape,
        perturbed only by tiny noise.
        """
        if not isinstance(self.mlp[0], nn.Linear) or not isinstance(self.mlp[2], nn.Linear) or not isinstance(self.mlp[4], nn.Linear):
            raise TypeError("SyntheticXNOREncoder.mlp does not have the expected Linear/ReLU/Linear/ReLU/Linear structure.")

        for idx in (0, 2, 4):
            self._init_linear_near_identity_(self.mlp[idx], noise_scale=noise_scale)

    def init_residual_identity_(self, branch_scale: float = 1e-3) -> None:
        """
        Initialize the residual encoder so the shortcut carries an identity-like
        map and the nonlinear MLP branch starts near zero.

        This is useful for residual formulations like:
            y = residual_proj(x) + mlp(x)
        where we want to preserve input geometry at initialization while still
        keeping a trainable nonlinear branch.
        """
        if not isinstance(self.residual_proj, nn.Linear):
            raise TypeError("SyntheticXNOREncoder.residual_proj is expected to be a Linear layer.")

        with torch.no_grad():
            proj = self.residual_proj
            proj.weight.zero_()
            k = min(proj.out_features, proj.in_features)
            proj.weight[:k, :k] = torch.eye(k, device=proj.weight.device, dtype=proj.weight.dtype)
            if proj.bias is not None:
                proj.bias.zero_()

            if self.input_layout == "1d":
                if not isinstance(self.mlp[0], nn.Linear) or not isinstance(self.mlp[2], nn.Linear) or not isinstance(self.mlp[4], nn.Linear):
                    raise TypeError("SyntheticXNOREncoder.mlp does not have the expected Linear/Act/Linear/Act/Linear structure.")

                for idx in (0, 2, 4):
                    layer = self.mlp[idx]
                    layer.weight.zero_()
                    if layer.bias is not None:
                        layer.bias.zero_()

                # Keep a tiny non-zero branch so gradients can start shaping it,
                # while the residual path dominates at initialization.
                first = self.mlp[0]
                last = self.mlp[4]
                k_first = min(first.out_features, first.in_features)
                k_last = min(last.out_features, last.in_features)
                first.weight[:k_first, :k_first] = branch_scale * torch.eye(
                    k_first, device=first.weight.device, dtype=first.weight.dtype
                )
                last.weight[:k_last, :k_last] = branch_scale * torch.eye(
                    k_last, device=last.weight.device, dtype=last.weight.dtype
                )
            elif self.input_layout == "2d":
                # Keep transformer branch near zero at initialization while
                # residual path carries identity-like mapping.
                self.seq_input_proj.weight.zero_()
                k_seq = min(self.seq_input_proj.out_features, self.seq_input_proj.in_features)
                self.seq_input_proj.weight[:k_seq, :k_seq] = branch_scale * torch.eye(
                    k_seq,
                    device=self.seq_input_proj.weight.device,
                    dtype=self.seq_input_proj.weight.dtype,
                )
                if self.seq_input_proj.bias is not None:
                    self.seq_input_proj.bias.zero_()
            else:
                raise ValueError(f"Unknown input_layout: {self.input_layout}")

    def apply_burkholz_relu_init(self, model: nn.Module, mode: str = "orthogonal"):
        from utils import burkholz_relu_init_linear
        for m in model.modules():
            if isinstance(m, nn.Linear):
                burkholz_relu_init_linear(m, mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_layout == "1d":
            z = self.mlp(x)
            if self.geometry_preserving == "residual":
                residual = self.residual_proj(x)
                return residual + z
            return z
        elif self.input_layout == "2d":
            z_tokens = self.seq_input_proj(x)
            z_tokens = z_tokens + self.pos_embed

            if self.geometry_preserving == "residual":
                residual = self.residual_proj(x)
                if residual.ndim == 2:
                    residual = residual.unsqueeze(1)
                z_tokens = z_tokens + residual

            z_tokens = self.seq_encoder(z_tokens)
            z = self.seq_out_proj(z_tokens.mean(dim=1))
            return z
        else:
            raise ValueError(f"Unknown input_layout: {self.input_layout}")


class SyntheticXNOREncoder_Res(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        emb_dim: int = 8192,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.emb_dim = emb_dim

        self.residual_proj = nn.Linear(input_dim, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        # self.apply(self._init_weights)

        self.init_residual_identity_()
    
    def _init_weights(
        self,
        m
    ) -> None: 
        if isinstance(m, (torch.nn.LayerNorm)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def init_identity_(self) -> None:
        """
        Reconfigure the MLP to be an identity-like map.

        The ReLU activations are replaced by Identity modules. The first Linear
        layer is initialized as a padded/truncated identity when
        input_dim != emb_dim, while the remaining square Linear layers are set
        to exact identity.
        """
        if not isinstance(self.mlp[0], nn.Linear) or not isinstance(self.mlp[2], nn.Linear) or not isinstance(self.mlp[4], nn.Linear):
            raise TypeError("SyntheticXNOREncoder.mlp does not have the expected Linear/ReLU/Linear/ReLU/Linear structure.")

        self.mlp[1] = nn.Identity()
        self.mlp[3] = nn.Identity()

        with torch.no_grad():
            first = self.mlp[0]
            first.weight.zero_()
            k = min(first.out_features, first.in_features)
            first.weight[:k, :k] = torch.eye(k, device=first.weight.device, dtype=first.weight.dtype)
            if first.bias is not None:
                first.bias.zero_()

            for idx in (2, 4):
                layer = self.mlp[idx]
                layer.weight.zero_()
                layer.weight.add_(torch.eye(layer.out_features, device=layer.weight.device, dtype=layer.weight.dtype))
                if layer.bias is not None:
                    layer.bias.zero_()

    def init_residual_identity_(self, branch_scale: float = 1e-3) -> None:
        """
        Initialize the residual encoder so the shortcut carries an identity-like
        map and the nonlinear MLP branch starts near zero.

        This is useful for residual formulations like:
            y = residual_proj(x) + mlp(x)
        where we want to preserve input geometry at initialization while still
        keeping a trainable nonlinear branch.
        """
        if not isinstance(self.residual_proj, nn.Linear):
            raise TypeError("SyntheticXNOREncoder.residual_proj is expected to be a Linear layer.")
        if not isinstance(self.mlp[0], nn.Linear) or not isinstance(self.mlp[2], nn.Linear) or not isinstance(self.mlp[4], nn.Linear):
            raise TypeError("SyntheticXNOREncoder.mlp does not have the expected Linear/Act/Linear/Act/Linear structure.")

        with torch.no_grad():
            proj = self.residual_proj
            proj.weight.zero_()
            k = min(proj.out_features, proj.in_features)
            proj.weight[:k, :k] = torch.eye(k, device=proj.weight.device, dtype=proj.weight.dtype)
            if proj.bias is not None:
                proj.bias.zero_()

            for idx in (0, 2, 4):
                layer = self.mlp[idx]
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()

            # Keep a tiny non-zero branch so gradients can start shaping it,
            # while the residual path dominates at initialization.
            first = self.mlp[0]
            last = self.mlp[4]
            k_first = min(first.out_features, first.in_features)
            k_last = min(last.out_features, last.in_features)
            first.weight[:k_first, :k_first] = branch_scale * torch.eye(
                k_first, device=first.weight.device, dtype=first.weight.dtype
            )
            last.weight[:k_last, :k_last] = branch_scale * torch.eye(
                k_last, device=last.weight.device, dtype=last.weight.dtype
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual_proj(x) + self.mlp(x)

"""
MC-MED
"""
class MCMEDWaveformEncoder(nn.Module):
    def __init__(
        self,
        input_length: int = 5000,
        emb_dim: int = 8192,
        d_model: int = 256,
        num_blocks: int = 3,
        kernel_size: int = 7,
        downsample_stride: int = 2,
        num_input_proj_layers: int = 2,
        input_proj_leaky_relu_negative_slope: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_length = int(input_length)
        self.emb_dim = int(emb_dim)
        self.d_model = int(d_model)
        self.downsample_stride = int(downsample_stride)
        self.num_input_proj_layers = int(num_input_proj_layers)
        self.input_proj_leaky_relu_negative_slope = float(input_proj_leaky_relu_negative_slope)

        stem_kernel = 7
        stem_padding = stem_kernel // 2
        stem_layers = []
        in_channels = 2
        for _ in range(self.num_input_proj_layers):
            stem_layers.append(
                nn.Conv1d(
                    in_channels,
                    self.d_model,
                    kernel_size=stem_kernel,
                    stride=self.downsample_stride,
                    padding=stem_padding,
                )
            )
            stem_layers.append(nn.BatchNorm1d(self.d_model))
            stem_layers.append(nn.LeakyReLU(negative_slope=self.input_proj_leaky_relu_negative_slope))
            in_channels = self.d_model
        self.input_proj = nn.Sequential(*stem_layers)
        self.resnet = nn.Sequential(
            *[
                _MCMEDResBlock1D(
                    channels=self.d_model,
                    kernel_size=int(kernel_size),
                    dropout=float(dropout),
                )
                for _ in range(int(num_blocks))
            ]
        )
        self.window_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(self.d_model, emb_dim)

        self.apply(self._init_weights)
        self.init_resnet_identity_()

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_resnet_identity_(self) -> None:
        """
        Initialize residual branches inside each 1D ResBlock near zero so the
        skip path dominates at initialization.
        """
        with torch.no_grad():
            for block in self.resnet:
                if not isinstance(block, _MCMEDResBlock1D):
                    continue
                block.conv2.weight.zero_()
                if block.conv2.bias is not None:
                    block.conv2.bias.zero_()
                block.bn2.weight.zero_()
                block.bn2.bias.zero_()

    def forward(self, x):
        windows = x["windows"]  # [B, N, L, 1]
        batch_size, num_windows, signal_len, num_channels = windows.shape

        if num_windows == 0:
            pooled = self.proj.weight.new_zeros((batch_size, self.d_model))
            return self.proj(pooled)

        # valid window if at least one observed sample exists in the window.
        window_valid = ~torch.isnan(windows).reshape(batch_size, num_windows, -1).all(dim=-1)  # [B, N]
        nan_mask = torch.isnan(windows).to(dtype=windows.dtype)
        packed = torch.cat([torch.nan_to_num(windows, nan=0.0), nan_mask], dim=-1)  # [B, N, L, 2]

        x = packed.reshape(batch_size * num_windows, signal_len, 2).transpose(1, 2)  # [B*N, 2, L]
        x = self.input_proj(x)
        x = self.resnet(x)
        x = self.window_pool(x).squeeze(-1).view(batch_size, num_windows, self.d_model)  # [B, N, D]

        weights = window_valid.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (x * weights).sum(dim=1) / denom
        return self.proj(pooled)

class MCMEDRadiologyEncoder(nn.Module):
    def __init__(
        self,
        model_params: dict = {
            "text_model_id": "StanfordAIMI/RadBERT",
            "proj_hidden_dim": 512,
            "dropout": 0.0,
        },
        emb_dim: int = 8192,
        geometry_preserving: bool = False,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        text_model_id = str(model_params.get("text_model_id", "StanfordAIMI/RadBERT"))
        self.text_encoder = AutoModel.from_pretrained(text_model_id)
        self.input_dim = int(self.text_encoder.config.hidden_size)
        self.geometry_preserving = bool(geometry_preserving)
        proj_hidden_dim = int(model_params.get("proj_hidden_dim", 512))
        dropout = float(model_params.get("dropout", 0.0))
        for p in self.text_encoder.parameters():
            p.requires_grad = True

        if self.geometry_preserving:
            self.residual_proj = nn.Linear(self.input_dim, emb_dim)
            self.residual_dropout = nn.Dropout(dropout)
        else:
            self.residual_proj = None
            self.residual_dropout = None
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, proj_hidden_dim),
            nn.LeakyReLU(negative_slope=0.4),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.LeakyReLU(negative_slope=0.4),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden_dim, emb_dim),
        )
        self.proj.apply(self._init_weights)
        if self.geometry_preserving:
            self.residual_proj.apply(self._init_weights)
            self.init_residual_identity_()

        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def init_residual_identity_(self, branch_scale: float = 1e-3) -> None:
        if not isinstance(self.residual_proj, nn.Linear):
            raise TypeError("MCMEDRadiologyEncoder.residual_proj is expected to be a Linear layer.")

        linears = [m for m in self.proj if isinstance(m, nn.Linear)]
        if len(linears) == 0:
            raise TypeError("MCMEDRadiologyEncoder.proj does not contain any Linear layers.")

        with torch.no_grad():
            self.residual_proj.weight.zero_()
            k = min(self.residual_proj.out_features, self.residual_proj.in_features)
            self.residual_proj.weight[:k, :k] = torch.eye(
                k,
                device=self.residual_proj.weight.device,
                dtype=self.residual_proj.weight.dtype,
            )
            if self.residual_proj.bias is not None:
                self.residual_proj.bias.zero_()

            for layer in linears:
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()

            first = linears[0]
            last = linears[-1]
            k_first = min(first.out_features, first.in_features)
            k_last = min(last.out_features, last.in_features)
            first.weight[:k_first, :k_first] = branch_scale * torch.eye(
                k_first,
                device=first.weight.device,
                dtype=first.weight.dtype,
            )
            last.weight[:k_last, :k_last] = branch_scale * torch.eye(
                k_last,
                device=last.weight.device,
                dtype=last.weight.dtype,
            )

    def forward(self, x):
        if not isinstance(x, dict):
            raise TypeError("MCMEDRadiologyEncoder expects dict input with tokenized radiology tensors.")

        input_ids = x["input_ids"]  # [B, R, L]
        attention_mask = x["attention_mask"]  # [B, R, L]
        report_mask = x["report_mask"]  # [B, R]

        batch_size, num_reports, seq_len = input_ids.shape
        if num_reports == 0:
            return self.proj[-1].weight.new_zeros((batch_size, self.emb_dim))

        flat_input_ids = input_ids.reshape(batch_size * num_reports, seq_len)
        flat_attention = attention_mask.reshape(batch_size * num_reports, seq_len).to(dtype=torch.long)
        bert_out = self.text_encoder(
            input_ids=flat_input_ids,
            attention_mask=flat_attention,
            return_dict=True,
        ).last_hidden_state  # [B*R, L, H]

        token_weights = flat_attention.to(dtype=bert_out.dtype).unsqueeze(-1)
        token_denom = token_weights.sum(dim=1).clamp_min(1.0)
        report_embeddings = (bert_out * token_weights).sum(dim=1) / token_denom  # [B*R, H]
        report_embeddings = report_embeddings.view(batch_size, num_reports, self.input_dim)

        if self.geometry_preserving:
            x = self.residual_dropout(self.residual_proj(report_embeddings)) + self.proj(report_embeddings)
        else:
            x = self.proj(report_embeddings)

        weights = report_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

class MCMEDClinicalEncoder(nn.Module):
    def __init__(
        self,
        model_params: dict = None,
        emb_dim: int = 8192,
        geometry_preserving: bool = False,
    ):
        super().__init__()
        if model_params is None:
            model_params = {
                "text_model_id": "emilyalsentzer/Bio_ClinicalBERT",
                "proj_hidden_dim": 512,
                "dropout": 0.0,
            }

        self.emb_dim = int(emb_dim)
        text_model_id = str(model_params.get("text_model_id", "emilyalsentzer/Bio_ClinicalBERT"))
        self.text_encoder = AutoModel.from_pretrained(text_model_id)
        self.input_dim = int(self.text_encoder.config.hidden_size)
        self.geometry_preserving = bool(geometry_preserving)
        proj_hidden_dim = int(model_params.get("proj_hidden_dim", 512))
        dropout = float(model_params.get("dropout", 0.0))
        for p in self.text_encoder.parameters():
            p.requires_grad = True

        if self.geometry_preserving:
            self.residual_proj = nn.Linear(self.input_dim, self.emb_dim)
            self.residual_dropout = nn.Dropout(dropout)
        else:
            self.residual_proj = None
            self.residual_dropout = None
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, proj_hidden_dim),
            nn.LeakyReLU(negative_slope=0.4),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.LeakyReLU(negative_slope=0.4),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden_dim, self.emb_dim),
        )
        self.proj.apply(self._init_weights)
        if self.geometry_preserving:
            self.residual_proj.apply(self._init_weights)
            self.init_residual_identity_()

        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def init_residual_identity_(self, branch_scale: float = 1e-3) -> None:
        if not isinstance(self.residual_proj, nn.Linear):
            raise TypeError("MCMEDClinicalEncoder.residual_proj is expected to be a Linear layer.")

        linears = [m for m in self.proj if isinstance(m, nn.Linear)]
        if len(linears) == 0:
            raise TypeError("MCMEDClinicalEncoder.proj does not contain any Linear layers.")

        with torch.no_grad():
            self.residual_proj.weight.zero_()
            k = min(self.residual_proj.out_features, self.residual_proj.in_features)
            self.residual_proj.weight[:k, :k] = torch.eye(
                k,
                device=self.residual_proj.weight.device,
                dtype=self.residual_proj.weight.dtype,
            )
            if self.residual_proj.bias is not None:
                self.residual_proj.bias.zero_()

            for layer in linears:
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()

            first = linears[0]
            last = linears[-1]
            k_first = min(first.out_features, first.in_features)
            k_last = min(last.out_features, last.in_features)
            first.weight[:k_first, :k_first] = branch_scale * torch.eye(
                k_first,
                device=first.weight.device,
                dtype=first.weight.dtype,
            )
            last.weight[:k_last, :k_last] = branch_scale * torch.eye(
                k_last,
                device=last.weight.device,
                dtype=last.weight.dtype,
            )

    def forward(self, x):
        if not isinstance(x, dict):
            raise TypeError("MCMEDClinicalEncoder expects dict input with tokenized clinical tensors.")

        input_ids = x["input_ids"]  # [B, E, L]
        attention_mask = x["attention_mask"]  # [B, E, L]
        event_mask = x["event_mask"]  # [B, E]

        batch_size, num_events, seq_len = input_ids.shape
        if num_events == 0:
            return self.proj[-1].weight.new_zeros((batch_size, self.emb_dim))

        flat_input_ids = input_ids.reshape(batch_size * num_events, seq_len)
        flat_attention = attention_mask.reshape(batch_size * num_events, seq_len).to(dtype=torch.long)
        bert_out = self.text_encoder(
            input_ids=flat_input_ids,
            attention_mask=flat_attention,
            return_dict=True,
        ).last_hidden_state  # [B*E, L, H]

        token_weights = flat_attention.to(dtype=bert_out.dtype).unsqueeze(-1)
        token_denom = token_weights.sum(dim=1).clamp_min(1.0)
        event_embeddings = (bert_out * token_weights).sum(dim=1) / token_denom  # [B*E, H]
        event_embeddings = event_embeddings.view(batch_size, num_events, self.input_dim)

        if self.geometry_preserving:
            x = self.residual_dropout(self.residual_proj(event_embeddings)) + self.proj(event_embeddings)
        else:
            x = self.proj(event_embeddings)

        weights = event_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (x * weights).sum(dim=1) / denom
        return pooled


class MCMEDDemographicsEncoder(nn.Module):
    def __init__(
        self,
        model_params: dict = {
            "proj_hidden_dim": 512,
            "dropout": 0.0,
        },
        emb_dim: int = 8192,
        geometry_preserving: bool = False,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.geometry_preserving = bool(geometry_preserving)
        self.proj_hidden_dim = int(model_params.get("proj_hidden_dim", 512))
        self.dropout = float(model_params.get("dropout", 0.0))
        self.input_dim = int(model_params.get("input_dim", 12))
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, self.proj_hidden_dim),
            nn.LeakyReLU(negative_slope=0.4),
            nn.Dropout(self.dropout),
            nn.Linear(self.proj_hidden_dim, self.proj_hidden_dim),
            nn.LeakyReLU(negative_slope=0.4),
            nn.Dropout(self.dropout),
            nn.Linear(self.proj_hidden_dim, self.emb_dim),
        )
        self.proj.apply(self._init_weights)
        if self.geometry_preserving:
            self.residual_proj = nn.Linear(self.input_dim, self.emb_dim)
            self.residual_dropout = nn.Dropout(self.dropout)
            self.residual_proj.apply(self._init_weights)

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        cont = x["continuous"].to(dtype=torch.float32)  # [B, Dc]
        cont_mask = x["continuous_mask"].to(dtype=torch.bool)  # [B, Dc]
        cat = x["categorical"].to(dtype=torch.float32)  # [B, Dk]
        cat_mask = x["categorical_mask"].to(dtype=torch.bool)  # [B, Dk]

        cont_filled = torch.where(cont_mask, cont, torch.zeros_like(cont))
        cat_filled = torch.where(cat_mask, cat, torch.zeros_like(cat))
        features = torch.cat(
            [
                cont_filled,
                cont_mask.to(dtype=cont.dtype),
                cat_filled,
                cat_mask.to(dtype=cat.dtype),
            ],
            dim=-1,
        )  # [B, 2 * (Dc + Dk)]

        if self.geometry_preserving:
            return self.residual_dropout(self.residual_proj(features)) + self.proj(features)
        return self.proj(features)

class MCMEDNumericsEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 12,
        emb_dim: int = 8192,
        trend_feature_dim: int = 7,
        d_model: int = 256,
        num_blocks: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.trend_feature_dim = int(trend_feature_dim)
        self.d_model = int(d_model)

        in_features = self.input_dim * self.trend_feature_dim * 2
        self.input_proj = nn.Linear(in_features, self.d_model)
        self.resnet = nn.Sequential(
            *[
                _MCMEDResBlock1D(
                    channels=self.d_model,
                    kernel_size=int(kernel_size),
                    dropout=float(dropout),
                )
                for _ in range(int(num_blocks))
            ]
        )
        self.output_proj = nn.Linear(self.d_model, int(emb_dim))
        self.apply(self._init_weights)
        self.init_resnet_identity_()

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_resnet_identity_(self) -> None:
        """
        Initialize residual branches inside each 1D ResBlock near zero so the
        skip path dominates at initialization.
        """
        with torch.no_grad():
            for block in self.resnet:
                if not isinstance(block, _MCMEDResBlock1D):
                    continue
                block.conv2.weight.zero_()
                if block.conv2.bias is not None:
                    block.conv2.bias.zero_()
                block.bn2.weight.zero_()
                block.bn2.bias.zero_()

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        trend_values = x["trend_values"]  # [B, T, V, F]

        batch_size, seq_len, _, _ = trend_values.shape
        if seq_len == 0:
            pooled = self.output_proj.weight.new_zeros((batch_size, self.d_model))
            return self.output_proj(pooled)

        # valid[t] is true iff at least one feature in the time bin is observed.
        time_valid = ~torch.isnan(trend_values).reshape(batch_size, seq_len, -1).all(dim=-1)
        nan_mask = torch.isnan(trend_values).to(dtype=trend_values.dtype)
        packed = torch.cat([torch.nan_to_num(trend_values, nan=0.0), nan_mask], dim=-1)
        x = packed.reshape(batch_size, seq_len, self.input_dim * self.trend_feature_dim * 2)

        x = self.input_proj(x)  # [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.resnet(x)  # [B, D, T]
        x = x.transpose(1, 2)  # [B, T, D]

        weights = time_valid.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (x * weights).sum(dim=1) / denom
        return self.output_proj(pooled)
