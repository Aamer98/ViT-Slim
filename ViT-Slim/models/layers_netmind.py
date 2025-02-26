# Huawei Canada proprietary
# Noah's Ark Lab, Montreal, NetMind team
# Supervisor: hongliang.li2@huawei.com
# Authors:
#   hang.li1@h-partners.com
#   amin.darabi@h-partners.com
import torch
import torch.nn as nn

# from timm.layers import DropPath


# class Block(torch.nn.Module):
#     """
#     A ViT module contains:

#     layer_norm, attention(QKV, Linear), add [by input],
#     layer_norm, MLP(Linear, activation, Linear),
#     add [by the output of previous add]
#     """

#     def __init__(
#             self,
#             dim: int,
#             head: int,
#             head_size: int,
#             mlp_size: int,
#             *,
#             qkv_bias: bool = False,
#             qk_scale: float | None = None,
#             proj_drop: float = 0,
#             attn_drop: float = 0,
#             drop_path: float = 0,
#             act_layer=torch.nn.GELU,
#             norm_layer=torch.nn.LayerNorm,
#             pruning: list[str] | str = None
#     ):

#         super().__init__()

#         self.params = []
#         self.masks = []

#         self.has_attention = bool(head and head_size)
#         self.has_mlp = bool(mlp_size)

#         if self.has_attention:
#             self.norm1 = norm_layer(dim)
#             self.attn = Attention(
#                 dim, num_heads=head, head_size=head_size,
#                 qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 attn_drop=attn_drop, proj_drop=proj_drop
#             )
#             self.attn = _make_searchable_attn(self.attn, pruning)
#             self.params.extend(self.norm1.parameters())
#             self.params.extend(self.attn.params)
#             self.masks.extend(self.attn.masks)

#         # NOTE: drop path for stochastic depth,
#         # we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path)

#         if self.has_mlp:
#             self.norm2 = norm_layer(dim)
#             self.mlp = MLP(
#                 in_features=dim,
#                 hidden_features=mlp_size,
#                 act_layer=act_layer,
#                 drop_out=proj_drop
#             )
#             self.mlp = _make_searchable_mlp(self.mlp, pruning)
#             self.params.extend(self.norm2.parameters())
#             self.params.extend(self.mlp.params)
#             self.masks.extend(self.mlp.masks)

#     def __set_params(self) -> None:
#         """
#         Set params and masks which will be used in optimizer parameters.
#         Should be called only when a change in parameters is happend.
#         NO NEED TO BE CALLED ANY TIME RIGHT NOW!
#         """

#         self.params = []
#         self.masks = []

#         if self.has_attention:
#             self.attn.set_params()
#             self.params.extend(self.norm1.parameters())
#             self.params.extend(self.attn.params)
#             self.masks.extend(self.attn.masks)

#         if self.has_mlp:
#             self.mlp.set_params()
#             self.params.extend(self.norm2.parameters())
#             self.params.extend(self.mlp.params)
#             self.masks.extend(self.mlp.masks)

#     def forward(self, x):

#         if self.has_attention:
#             x = x + self.drop_path(self.attn(self.norm1(x)))
#         if self.has_mlp:
#             x = x + self.drop_path(self.mlp(self.norm2(x)))

#         return x

#     def get_layers_importance(self) -> tuple[float, float]:
#         """
#         Return sigmoid(zetas). [attn, mlp]
#         """

#         importance = [0.0, 0.0]
#         if self.has_attention:
#             importance[0] = torch.sigmoid(self.attn.zeta_layer).item()
#         if self.has_mlp:
#             importance[1] = torch.sigmoid(self.mlp.zeta_layer).item()
#         return importance

#     # NOTE: not pruned compressasion right now is not working well!
#     def compress(
#             self,
#             head: int = -1,
#             head_size: int = -1,
#             mlp_size: int = -1
#     ) -> None:

#         num_heads = self.attn.num_heads if head < 0 else head
#         head_size = self.attn.head_size if head_size < 0 else head_size
#         mlp_size = self.mlp.fc1.out_features if mlp_size < 0 else mlp_size

#         self.has_attention = bool(num_heads and head_size)
#         self.has_mlp = bool(mlp_size)

#         if not self.has_attention:
#             self.attn = None
#         if not self.has_mlp:
#             self.mlp = None

#         if (
#             self.has_attention and
#             (
#                 num_heads < self.attn.num_heads or
#                 head_size < self.attn.head_size
#             )
#         ):
#             self.attn = self.attn.compress(
#                 num_heads=num_heads, head_size=head_size)

#         if self.has_mlp and mlp_size < self.mlp.fc1.out_features:
#             self.mlp = self.mlp.compress(mlp_size)

#         self.params = []
#         self.masks = []

#         if self.has_attention:
#             self.attn.set_params()
#             self.params.extend(self.norm1.parameters())
#             self.params.extend(self.attn.params)
#             self.masks.extend(self.attn.masks)

#         if self.has_mlp:
#             self.mlp.set_params()
#             self.params.extend(self.norm2.parameters())
#             self.params.extend(self.mlp.params)
#             self.masks.extend(self.mlp.masks)




class Attention(torch.nn.Module):

    def __init__(
            self,
            dim: int,
            head_size: int,
            num_heads: int = 8,
            qkv_bias: int = False,
            qk_scale: float | None = None,
            attn_drop: float = 0,
            proj_drop: float = 0
    ):

        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.qkv_bias = qkv_bias
        self.scale = qk_scale if qk_scale is not None else head_size ** -0.5

        self.qkv = torch.nn.Linear(
            dim, head_size * num_heads * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(head_size * num_heads, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

        # 
        self._prune_head = False
        self._prune_layer = False
        self._prune_node = False

        self.params = []
        self.masks = []

        self.params.extend(self.qkv.parameters())
        self.params.extend(self.proj.parameters())

    def forward(self, x):

        batch_size, sequence_length, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, sequence_length, 3, self.num_heads, self.head_size
        ).permute(2, 0, 3, 1, 4)

        # B, H, N, d
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(
            batch_size, sequence_length, self.num_heads * self.head_size)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SparseAttention(Attention):
    """
    """

    def __init__(
            self,
            attn_module: Attention,
            prune_head: bool = False,
            prune_layer: bool = False,
            prune_node: bool = False
    ):

        super().__init__(
            attn_module.qkv.in_features,
            attn_module.num_heads,
            attn_module.head_size,
            attn_module.qkv_bias,
            attn_module.scale,
            attn_module.attn_drop.p,
            attn_module.proj_drop.p
        )
        
        self.is_searched = False

        self._prune_head = prune_head
        self._prune_layer = prune_layer
        self._prune_node = prune_node

        if self._prune_node:
            self.zeta_node = torch.nn.Parameter(
                torch.ones(1, 1, self.num_heads, 1, self.head_size))
            self.masks.append(self.zeta_node)
        else:
            self.zeta_node = None

        if self._prune_head:
            self.zeta_head = torch.nn.Parameter(
                torch.ones(1, 1, self.num_heads, 1))
            self.masks.append(self.zeta_head)
        else:
            self.zeta_head = None

        if self._prune_layer:
            self.zeta_layer = torch.nn.Parameter(torch.ones(1))
            self.masks.append(self.zeta_layer)
        else:
            self.zeta_layer = None

    def forward(self, x):

        batch_size, sequence_length, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, sequence_length, 3, self.num_heads, self.head_size
        ).permute(2, 0, 3, 1, 4)  # 3, B, H, N, d(C/H)

        if self._prune_node:
            qkv = qkv * torch.nn.functional.sigmoid(self.zet_node)

        # B, H, N, d
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)
        if self._prune_head:
            x = x * torch.nn.functional.sigmoid(self.zeta_head)

        x = x.reshape(
            batch_size, sequence_length, self.num_heads * self.head_size)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self._prune_layer:
            x = x * torch.nn.functional.sigmoid(self.zeta_layer)

        return x

    def compress(
            self,
            num_heads: int = -1,
            head_size: int = -1
    ):

        num_heads = self.num_heads if num_heads < 0 else num_heads
        head_size = self.head_size if head_size < 0 else head_size

        module = Attention(
            self.qkv.in_features,
            num_heads,
            head_size,
            self.qkv_bias,
            self.scale,
            self.attn_drop.p,
            self.proj_drop.p
        )

        # Handle qkv parameters
        desired_qkv = self.qkv.state_dict()['weight'].reshape(
            self.qkv.in_features, 3, self.num_heads, self.head_size)
        if self._prune_head:
            desired_qkv = desired_qkv[
                :, :,
                torch.argsort(
                    self.zeta_head[0, 0, :, 0], descending=True
                )[:num_heads], :
            ]
        if self._prune_node:
            desired_qkv = torch.take_along_dim(
                desired_qkv,
                torch.argsort(
                    self.zeta_node, descending=True
                )[:, :, :, 0, :head_size],
                dim=-1
            )
        desired_qkv = desired_qkv.reshape(self.qkv.in_features, -1)
        module.qkv.state_dict()['weight'].copy_(desired_qkv)
        if self.qkv_bias:
            desired_qkv_bias = self.qkv.state_dict()['bias'].reshape(
                3, self.num_heads, self.head_size)
            if self._prune_head:
                desired_qkv_bias = desired_qkv_bias[
                    :, :,
                    torch.argsort(
                        self.zeta_head[0, 0, :, 0], descending=True
                    )[:num_heads]
                ]
            if self._prune_node:
                desired_qkv_bias = torch.take_along_dim(
                    desired_qkv_bias,
                    torch.argsort(
                        self.zeta_node, descending=True
                    )[0, :, :, 0, :head_size],
                    dim=-1
                )
            desired_qkv_bias = desired_qkv_bias.reshape(-1)
            module.qkv.state_dict()['bias'].copy_(desired_qkv_bias)

        # Handle projection parameters
        desired_proj = self.proj.state_dict()['weight'].reshape(
            self.num_heads, self.head_size, self.proj.out_features)
        if self._prune_head:
            desired_proj = desired_proj[
                torch.argsort(
                    self.zeta_head[0, 0, :, 0], descending=True
                )[:num_heads],
                :, :
            ]
        if self._prune_node:
            desired_proj = torch.take_along_dim(
                desired_proj,
                torch.argsort(
                    self.zeta_node, descending=True
                )[0, 0, :, :, :].transpose(-1, -2),
                dim=1
            )
        desired_proj = desired_proj.reshape(-1, self.proj.out_features)
        module.proj.state_dict()['weight'].copy(desired_proj)

        return module

    @staticmethod
    def from_attn(attn_module, head_search=False, uniform_search=False):
        attn_module = SparseAttention(attn_module, head_search, uniform_search)
        return attn_module


class Mlp(torch.nn.Module):

    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            act_layer: torch.nn.Module = torch.nn.GELU,
            drop_out: float = 0
    ):

        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = torch.nn.Linear(hidden_features, in_features)
        self.drop = torch.nn.Dropout(drop_out)

        self._prune_node = False
        self._prune_layer = False

        self.act_layer = act_layer
        self.params = []
        self.masks = []

        self.params.extend(self.fc1.parameters())
        self.params.extend(self.fc2.parameters())

    def forward(self, x):

        x = self.act(self.fc1(x))
        x = self.fc2(self.drop(x))

        return self.drop(x)


class SparseMlp(Mlp):

    def __init__(
            self,
            mlp_module: Mlp,
            prune_node: bool = False,
            prune_layer: bool = False
    ):

        super().__init__(
            mlp_module.fc1.in_features, mlp_module.fc1.out_features,
            act_layer=nn.GELU, drop_out=mlp_module.drop.p
        )

        self.is_searched = False

        self._prune_node = prune_node
        self._prune_layer = prune_layer

        if self._prune_node:
            self.zeta_node = torch.nn.Parameter(
                torch.ones(1, 1, mlp_module.fc1.out_features))
            self.masks.append(self.zeta_node)
        else:
            self.zeta_node = None

        if self._prune_layer:
            self.zeta_layer = torch.nn.Parameter(torch.ones(1))
            self.masks.append(self.zeta_layer)
        else:
            self.zeta_layer = None

    def forward(self, x, patch_zeta=None):

        if patch_zeta is not None:
            x*=patch_zeta
        x = self.act(self.fc1(x))
        if self._prune_node:
            x = x * torch.nn.functional.sigmoid(self.zeta_node)
        x = self.fc2(self.drop(x))
        if self._prune_layer:
            x = x * torch.nn.functional.sigmoid(self.zeta_layer)

        return self.drop(x)

    def compress(
            self,
            mlp_size: int = -1
    ):

        mlp_size = self.mlp.fc1.out_features if mlp_size < 0 else mlp_size

        module = Mlp(
            self.fc1.in_features,
            self.fc1.out_features,
            act_layer=self.act_layer,
            drop_out=self.drop.p
        )

        desired_fc1 = self.fc1.state_dict()['weight']
        desired_fc1_bias = self.fc1.state_dict()['bias']
        desired_fc2 = self.fc2.state_dict()['weight']

        if self._prune_node:
            desired_fc1 = desired_fc1[
                :, torch.argsort(self.zeta_node[0, 0, :])[:mlp_size]]
            desired_fc1_bias = desired_fc1_bias[
                torch.argsort(self.zeta_node[0, 0])[:mlp_size]]
            desired_fc2 = desired_fc2[
                torch.argsort(self.zeta_node[0, 0, :])[:mlp_size], :]

        module.fc1.state_dict()['weight'].copy_(desired_fc1)
        module.fc1.state_dict()['bias'].copy_(desired_fc1_bias)
        module.fc2.state_dict()['weight'].copy_(desired_fc2)

        return module

    @staticmethod
    def from_mlp(mlp_module):
        mlp_module = SparseMlp(mlp_module)
        return mlp_module


class ModuleInjection:
    method = 'full'
    searchable_modules = []

    @staticmethod
    def make_searchable_attn(attn_module, head_search=False, uniform_search=False):
        if ModuleInjection.method == 'full':
            return attn_module
        attn_module = SparseAttention.from_attn(attn_module, head_search, uniform_search)
        ModuleInjection.searchable_modules.append(attn_module)
        return attn_module

    @staticmethod
    def make_searchable_mlp(mlp_module):
        if ModuleInjection.method == 'full':
            return mlp_module
        mlp_module = SparseMlp.from_mlp(mlp_module)
        ModuleInjection.searchable_modules.append(mlp_module)
        return mlp_module


def _make_searchable_attn(attn_module: Attention, pruning: list[str] | str):

    if pruning is None:
        return attn_module
    if isinstance(pruning, str):
        pruning = [pruning]
    pruning = [p.lower() for p in pruning]

    prune_head = 'head' in pruning
    prune_layer = 'layer' in pruning
    prune_node = 'node' in pruning

    if not (prune_head or prune_layer or prune_node):
        return attn_module

    return SparseAttention(
        attn_module=attn_module,
        prune_head=prune_head,
        prune_layer=prune_layer,
        prune_node=prune_node
    )


def _make_searchable_mlp(mlp_module: Attention, pruning: list[str] | str):

    if pruning is None:
        return mlp_module
    if isinstance(pruning, str):
        pruning = [pruning]
    pruning = [p.lower() for p in pruning]

    prune_layer = 'layer' in pruning
    prune_node = 'node' in pruning

    if not (prune_layer or prune_node):
        return mlp_module

    return SparseMlp(
        mlp_module=mlp_module,
        prune_layer=prune_layer,
        prune_node=prune_node
    )
