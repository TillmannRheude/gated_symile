import torch
import torch.nn as nn

from torchvision import models
from transformers import AutoModel


"""
Symile-MIMIC 
"""
class CXREncoder_ResNet(nn.Module):
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

        self.resnet = models.resnet50(weights=resnet_params["weights"])
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, emb_dim, bias=True)

        nn.init.kaiming_normal_(self.resnet.fc.weight, mode="fan_out")
        nn.init.zeros_(self.resnet.fc.bias)

    def forward(self, x):
        x = self.resnet(x)
        return x

class ECGEncoder_ResNet(nn.Module):
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

        self.resnet = models.resnet18(weights=resnet_params["weights"])
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, emb_dim, bias=True)

        nn.init.kaiming_normal_(self.resnet.fc.weight, mode="fan_out")
        nn.init.zeros_(self.resnet.fc.bias)
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode="fan_out")

    def forward(self, x):
        x = self.resnet(x)
        return x

class LabsEncoder_ResNet(nn.Module):
    def __init__(
        self,
        emb_dim: int = 8192,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, emb_dim)
        self.gelu = nn.GELU()

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
        # freeze all layers
        #for param in self.parameters():
        #    param.requires_grad = False
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.fc3(x)
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


class CXREncoder(nn.Module):
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
            patch_size=64,
            in_channels=3,
            emb_dim=emb_dim,
        )
        self.residual_attn = nn.Linear(emb_dim, 1, bias=False)
        self.residual_norm = nn.LayerNorm(emb_dim)
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
        attn = torch.softmax(self.residual_attn(residual_tokens).squeeze(-1), dim=1)
        residual = torch.sum(residual_tokens * attn.unsqueeze(-1), dim=1)
        return self.residual_norm(strong + residual)

class ECGEncoder(nn.Module):
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
            patch_size=(250, 12),
            in_channels=1,
            emb_dim=emb_dim,
        )
        self.residual_attn = nn.Linear(emb_dim, 1, bias=False)
        self.residual_norm = nn.LayerNorm(emb_dim)
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
        attn = torch.softmax(self.residual_attn(residual_tokens).squeeze(-1), dim=1)
        residual = torch.sum(residual_tokens * attn.unsqueeze(-1), dim=1)
        return self.residual_norm(strong + residual)

class LabsEncoder(nn.Module):
    def __init__(
        self,
        emb_dim: int = 8192,
    ):
        super().__init__()
        self.input_dim = 100
        self.emb_dim = emb_dim

        self.residual_proj = nn.Linear(self.input_dim, emb_dim)
        self.residual_norm = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
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
        if not all(isinstance(self.mlp[idx], nn.Linear) for idx in (0, 2, 4)):
            raise TypeError("LabsEncoder_Residual.mlp does not have the expected Linear/Act/Linear/Act/Linear structure.")

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

    def forward(self, x):
        return self.residual_norm(self.residual_proj(x) + self.mlp(x))



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
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.emb_dim = emb_dim
        self.modality_name = modality_name

        self.combine_eids_as = combine_eids_as
        if self.combine_eids_as == "union":
            # also pass binary missing mask as input to the MLP
            input_dim = input_dim * 2
        self.residual_proj = nn.Linear(input_dim, emb_dim, bias=True)
        self.out_norm = nn.LayerNorm(emb_dim)

        layers = []
        prev = input_dim
        for hidden_dim, hidden_dropout in zip(hidden_dims, hidden_dropouts):
            layers.append(nn.Linear(prev, hidden_dim, bias=True))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(hidden_dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, emb_dim, bias=True))
        self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
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

    def forward(self, x):
        if self.combine_eids_as == "union":
            nanmask = torch.isnan(x).float()
            x = torch.nan_to_num(x, nan=0.0)
            x = torch.cat([x, nanmask], dim=1)
        if torch.isnan(x).any():
            # raise ValueError("NaN values present in input")
            x = torch.nan_to_num(x, nan=0.0)

        return self.out_norm(self.residual_proj(x) + self.mlp(x))

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
Synthetic XNOR
"""
class SyntheticXNOREncoder(nn.Module):
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
            nn.Linear(emb_dim, emb_dim),
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
