import torch
import torch.nn as nn

from torchvision import models
from transformers import AutoModel


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

class ECGEncoder(nn.Module):
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

class LabsEncoder(nn.Module):
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

        self.emb_dim = emb_dim
        self.modality_name = modality_name

        self.combine_eids_as = combine_eids_as
        if self.combine_eids_as == "union":
            # also pass binary missing mask as input to the MLP
            input_dim = input_dim * 2

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
        self.emb_dim = emb_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.ReLU(inplace=True),
            #nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            #nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
        )
        self.apply(self._init_weights)
    
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
