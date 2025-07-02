from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, repeat
# from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.vision_transformer import PatchEmbed
from U25031301_ViT import CrossBlock, Block, Out_Block
import torch.nn.functional as F
# from He_utils.pos_embed import get_2d_sincos_pos_embed
# from He_utils.pos_embed import get_1d_sincos_pos_embed
from He_utils.pos_embed import get_120_sincos_pos_embed

class PatchEmbed2W(nn.Module):
    """ 1D Patch Embedding for 1D signal """
    def __init__(self, signal_length=1000, patch_size=100, in_chans=12, embed_dim=256):
    # def __init__(self, signal_length=1000, patch_size=50, in_chans=12, embed_dim=256):
        super().__init__()
        self.signal_length = signal_length
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Ensure signal length is divisible by patch size
        assert signal_length % patch_size == 0, "Signal length must be divisible by patch size"
        self.num_patches = signal_length // patch_size
        self.linear = nn.Linear(512, embed_dim)
        self.t_patch = nn.Linear(100, embed_dim)

        self.time_mlp = nn.Sequential(
            nn.Flatten(start_dim=2),  # 将每个 (12, 100) 的 patch 展平为 (1200,)
            nn.Linear(12 * 100, 256),  # 映射到 256 个特征
            # nn.ReLU(inplace=True),
            # nn.Linear(512, embed_dim)  # 保持输出尺寸一致
        )

        self.chan_mlp = nn.Sequential(
            # nn.Flatten(start_dim=2),  # 将每个 (12, 100) 的 patch 展平为 (1200,)
            nn.Linear(1000, 256),  # 映射到 256 个特征
            # nn.ReLU(inplace=True),
            # nn.Linear(512, embed_dim)  # 保持输出尺寸一致
        )

    def forward(self, x):
        batch, _, _ = x.size()


        x = x.unfold(dimension=2, size=100, step=100)
        x = self.t_patch(x)
        x = x.view(batch, 120, 256)

        return x

class FreqFilter_multihead(nn.Module):
    '''freq filter layer w multihead design, either maxpooling or projection'''
    def __init__(self, head=8, length=256, mode='self'):
        super().__init__()
        assert mode in ['pool', 'self']
        self.mode = mode
        self.head, self.dim = head, length
        self.complex_weight = nn.Parameter(torch.randn(head, length, 2, dtype=torch.float32) * 0.02)

        if self.mode == 'self':
            self.filter_weight = nn.Linear(length, head)

    def forward(self, x):
        '''input size: B x N x C'''
        B, N, C = x.shape
        device = x.device

        x = x.to(torch.float32)
        x = torch.fft.rfft(x, dim=(1), norm='ortho')

        if self.mode == 'self':
            head_weight = self.filter_weight(x.real)

        x = repeat(x, 'b p d -> b p h d', h=self.head)
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight

        if self.mode == 'pool':
            x = rearrange(x, 'b p h d -> (b p d) h')
            _, indices = F.max_pool1d(x.abs(), kernel_size=x.shape[-1], return_indices=True)
            batch_range = torch.arange(x.shape[0], device = x.device)[:, None]
            x = x[batch_range, indices][:, 0]
            x = rearrange(x, '(b p d) -> b p d', b=B, d=self.dim)

        elif self.mode == 'self':
            x = x * head_weight.unsqueeze(-1)
            x = torch.sum(x, dim=-2)

        x = torch.fft.irfft(x, n=N, dim=(1), norm='ortho')
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None, dropout = 0.):
        super().__init__()
        if out_dim is None:
            out_dim = dim

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class FF_Block(nn.Module):
    def __init__(self, embed_dim, head, length, hidden_dim, dropout=0.0):
        super(FF_Block, self).__init__()
        # 定义 FreqFilter 和 FeedForward
        self.FreqFilter = FreqFilter_multihead(head=head, length=length, mode='self')
        self.FeedForward = FeedForward(dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x):
        # FreqFilter 操作
        x = x + self.FreqFilter(x)
        # FeedForward 操作
        x = x + self.FeedForward(x)
        return x

class DescribeResize(nn.Module):
    def __init__(self, input_dim: int = 768, output_channels: int = 8, output_length: int = 1000):
        super(DescribeResize, self).__init__()
        self.output_channels = output_channels
        self.output_length = output_length
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_channels * output_length)
        )

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        x = x.view(-1, self.output_channels, self.output_length)
        return x

class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(in_channels, max(1, in_channels // 16), kernel_size=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv1d(max(1, in_channels // 16), in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        device = x.device  # 获取输入的设备
        self.conv1 = self.conv1.to(device)  # 确保卷积层在同一设备上
        self.conv2 = self.conv2.to(device)  # 确保卷积层在同一设备上
        x = self.avg_pool(x)  # 输出形状为 (batch_size, in_channels, 1)
        x = self.act(self.conv1(x))  # 输出形状为 (batch_size, max(1, in_channels // 16), 1)
        x = self.sigmoid(self.conv2(x))  # 输出形状为 (batch_size, in_channels, 1)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, query_channels: int, key_value_channels: int, num_heads: int = 1, d_k: int = None):
        super().__init__()
        if d_k is None:
            d_k = query_channels
        self.scale = d_k ** -0.5
        self.query_proj = nn.Linear(query_channels, num_heads * d_k)
        self.key_proj = nn.Linear(key_value_channels, num_heads * d_k)
        self.value_proj = nn.Linear(key_value_channels, num_heads * d_k)
        self.output_proj = nn.Linear(num_heads * d_k, query_channels)
        self.num_heads = num_heads
        self.d_k = d_k

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        batch_size, _, _ = query.shape
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_k)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_k)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_k)
        attn_weights = torch.einsum('bqhd,bkhd->bhqk', query, key) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)
        query = torch.einsum('bhqk,bkhd->bqhd', attn_weights, value)
        query = query.contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.output_proj(query)

class ConditionalFusionModule(nn.Module):
    def __init__(self, latent_channels: int, encode_channels: int):
        super().__init__()
        self.channel_attention = ChannelAttentionBlock(latent_channels)
        # self.downsample = nn.Conv1d(noisy_channels + condition_channels, noisy_channels, kernel_size=1)
        self.cross_attention = CrossAttentionBlock(query_channels=latent_channels,
                                                   key_value_channels=encode_channels)

    def forward(self, feature: torch.Tensor, describe_encode: torch.Tensor):
        device = feature.device  # 获取输入的设备
        channel_num_n = feature.size(1)
        # channel_num_c = feature_condition.size(1)
        encode_channels = describe_encode.size(1)

        weight_concat = self.channel_attention(feature)
        x = feature * weight_concat

        # x = feature

        # x = feature
        # x = x + x0
        # x = self.downsample(x)
        x = self.cross_attention(x.permute(0, 2, 1), describe_encode.permute(0, 2, 1), describe_encode.permute(0, 2, 1))
        return x.permute(0, 2, 1)

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, signal_length=1000, patch_size=100, in_chans=12,
    # def __init__(self, signal_length=1000, patch_size=50, in_chans=12,
                 embed_dim=256, depth=8, num_heads=8,
                 decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed2W(signal_length=signal_length, patch_size=patch_size, in_chans=in_chans,
                                        embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches * 12
        # num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),requires_grad=False)  # fixed sin-cos embedding

        self.blocks_t = nn.ModuleList([
            # Block(dim=256, num_heads=8, mlp_ratio=4.0, qkv_bias=True, norm_layer=norm_layer)
            Out_Block(dim=256, num_heads=8, mlp_ratio=4.0, qkv_bias=True, norm_layer=norm_layer)
            for i in range(4)])
        self.norm_t = norm_layer(embed_dim)

        self.blocks_l = nn.ModuleList([
            # Block(dim=256, num_heads=8, mlp_ratio=4.0, qkv_bias=True, norm_layer=norm_layer)
            Out_Block(dim=256, num_heads=8, mlp_ratio=4.0, qkv_bias=True, norm_layer=norm_layer)
            for i in range(4)])
        self.norm_l = norm_layer(embed_dim)

        self.blocks = nn.ModuleList([
            # Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            Out_Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        self.FreqFilter = FreqFilter_multihead(head=8, length=256, mode='self')
        self.FeedFoward = FeedForward(dim=256, hidden_dim=1024, out_dim=None, dropout = 0.)
        self.ff_blocks = nn.ModuleList([
            FF_Block(embed_dim, head=8, length=256, hidden_dim=1024, dropout=0.)
            for _ in range(1)
        ])
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed_l = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_t = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])
        self.decoder_blocks_l = nn.ModuleList([
            # Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            Out_Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_blocks_t = nn.ModuleList([
            # Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            Out_Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm_l = norm_layer(decoder_embed_dim)
        self.decoder_norm_t = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        self.decoder_pred_l = nn.Linear(decoder_embed_dim, patch_size, bias=True)
        self.decoder_pred_t = nn.Linear(decoder_embed_dim, patch_size, bias=True)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        self.describe_resize = DescribeResize(input_dim=768, output_channels=8, output_length=256)
        self.cond_fusion = ConditionalFusionModule(79 , 8)

    def initialize_weights(self):
        pos_embed = get_120_sincos_pos_embed(
            embed_dim=self.pos_embed.shape[-1],  # 嵌入维度
            grid_size=120,  # patch总数
            # grid_size=240,  # patch总数
            cls_token=True  # 包含CLS token
        )

        # 确保设备一致
        device = self.pos_embed.device  # 获取 pos_embed 当前设备
        pos_embed_tensor = torch.from_numpy(pos_embed).float().unsqueeze(0).to(device)

        # 检查形状是否匹配
        if pos_embed_tensor.shape != self.pos_embed.shape:
            raise ValueError(f"Shape mismatch: expected {self.pos_embed.shape}, got {pos_embed_tensor.shape}")

        # 使用 .data.copy_ 复制数据
        self.pos_embed.data.copy_(pos_embed_tensor)

        decoder_pos_embed = get_120_sincos_pos_embed(
            embed_dim=self.pos_embed.shape[-1],  # 嵌入维度
            grid_size=120,  # patch总数
            # grid_size=240,  # patch总数
            cls_token=True  # 包含CLS token
        )

        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, signals):

        p = 100  # patch size
        c = 12  # number of channels
        num_patches = 10  # number of patches

        # Reshape to [N, c, num_patches, p] without permute
        patches = signals.reshape(signals.shape[0], c, num_patches, p)

        # Reshape to [N, num_patches * c, p], this will give you channel1_1, channel1_2, ..., channel12_10
        patches = patches.reshape(signals.shape[0], c * num_patches, p)

        return patches
    def unpatchify(self, patches):
        # p = self.patch_embed.patch_size  # 每个 patch 的长度
        p = 100
        # p = 50
        # c = self.patch_embed.in_chans  # 信号的物理通道数
        c = 12
        # num_patches_per_channel = self.patch_embed.num_patches  # 每个通道的 patch 数
        num_patches_per_channel = 10
        # num_patches_per_channel = 20

        assert patches.shape[1] == num_patches_per_channel * c, \
            "The number of patches does not match the expected structure"

        # Reshape patches to [N, C, num_patches_per_channel, patch_size]
        signals = patches.reshape(patches.shape[0], c, num_patches_per_channel, p)
        # Rearrange to [N, C, L]
        signals = signals.permute(0, 1, 2, 3).reshape(patches.shape[0], c, -1)
        return signals

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = round(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def LeadView_masking(self, x, mask_ratio):
        x_0 = x[:, 0:10, :]
        x_1 = x[:, 10:20, :]
        x_2 = x[:, 20:30, :]
        x_3 = x[:, 30:40, :]
        x_4 = x[:, 40:50, :]
        x_5 = x[:, 50:60, :]
        x_6 = x[:, 60:70, :]
        x_7 = x[:, 70:80, :]
        x_8 = x[:, 80:90, :]
        x_9 = x[:, 90:100, :]
        x_10 = x[:, 100:110, :]
        x_11 = x[:, 110:120, :]

        bs, ps, ed = x_0.shape
        len_keep = round(10 * (1 - mask_ratio))

        x_0, mask_0, ids_restore_0 = self.random_masking(x_0, mask_ratio)
        x_1, mask_1, ids_restore_1 = self.random_masking(x_1, mask_ratio)
        x_2, mask_2, ids_restore_2 = self.random_masking(x_2, mask_ratio)
        x_3, mask_3, ids_restore_3 = self.random_masking(x_3, mask_ratio)
        x_4, mask_4, ids_restore_4 = self.random_masking(x_4, mask_ratio)
        x_5, mask_5, ids_restore_5 = self.random_masking(x_5, mask_ratio)
        x_6, mask_6, ids_restore_6 = self.random_masking(x_6, mask_ratio)
        x_7, mask_7, ids_restore_7 = self.random_masking(x_7, mask_ratio)
        x_8, mask_8, ids_restore_8 = self.random_masking(x_8, mask_ratio)
        x_9, mask_9, ids_restore_9 = self.random_masking(x_9, mask_ratio)
        x_10, mask_10, ids_restore_10 = self.random_masking(x_10, mask_ratio)
        x_11, mask_11, ids_restore_11 = self.random_masking(x_11, mask_ratio)

        ids_restore_1 = ids_restore_1 + 10
        ids_restore_2 = ids_restore_2 + 20
        ids_restore_3 = ids_restore_3 + 30
        ids_restore_4 = ids_restore_4 + 40
        ids_restore_5 = ids_restore_5 + 50
        ids_restore_6 = ids_restore_6 + 60
        ids_restore_7 = ids_restore_7 + 70
        ids_restore_8 = ids_restore_8 + 80
        ids_restore_9 = ids_restore_9 + 90
        ids_restore_10 = ids_restore_10 + 100
        ids_restore_11 = ids_restore_11 + 110

        mask = torch.cat(
            (mask_0, mask_1, mask_2, mask_3, mask_4, mask_5, mask_6, mask_7, mask_8, mask_9, mask_10, mask_11), dim=1)
        # ids_restore = torch.cat((ids_restore_0,ids_restore_1,ids_restore_2,ids_restore_3,ids_restore_4,ids_restore_5,
        #                          ids_restore_6,ids_restore_7,ids_restore_8,ids_restore_9,ids_restore_10,ids_restore_11), dim=1)
        ids_keep = torch.cat((ids_restore_0[:, :len_keep], ids_restore_1[:, :len_keep], ids_restore_2[:, :len_keep], ids_restore_3[:, :len_keep],
                              ids_restore_4[:, :len_keep], ids_restore_5[:, :len_keep], ids_restore_6[:, :len_keep], ids_restore_7[:, :len_keep],
                              ids_restore_8[:, :len_keep], ids_restore_9[:, :len_keep], ids_restore_10[:, :len_keep], ids_restore_11[:, :len_keep]),
                             dim=1)
        ids_mask = torch.cat((ids_restore_0[:, len_keep:], ids_restore_1[:, len_keep:], ids_restore_2[:, len_keep:], ids_restore_3[:, len_keep:],
                              ids_restore_4[:, len_keep:], ids_restore_5[:, len_keep:], ids_restore_6[:, len_keep:], ids_restore_7[:, len_keep:],
                              ids_restore_8[:, len_keep:], ids_restore_9[:, len_keep:], ids_restore_10[:, len_keep:], ids_restore_11[:, len_keep:]),
                             dim=1)
        ids_restore = torch.cat((ids_keep, ids_mask), dim=1)

        x_masked = torch.cat((x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11), dim=1)
        return x_masked, mask, ids_restore

    def TimeView_masking(self, x, mask_ratio):
        bs, ps, ed = x.shape
        len_keep = round(ps * (1 - mask_ratio))

        x = x.view(bs, 12, 10, 256)
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(bs, 120, 256)

        x_0 = x[:, 0:12, :]
        x_1 = x[:, 12:24, :]
        x_2 = x[:, 24:36, :]
        x_3 = x[:, 36:48, :]
        x_4 = x[:, 48:60, :]
        x_5 = x[:, 60:72, :]
        x_6 = x[:, 72:84, :]
        x_7 = x[:, 84:96, :]
        x_8 = x[:, 96:108, :]
        x_9 = x[:, 108:120, :]

        x_0, mask_0, ids_restore_0 = self.random_masking(x_0, mask_ratio)
        x_1, mask_1, ids_restore_1 = self.random_masking(x_1, mask_ratio)
        x_2, mask_2, ids_restore_2 = self.random_masking(x_2, mask_ratio)
        x_3, mask_3, ids_restore_3 = self.random_masking(x_3, mask_ratio)
        x_4, mask_4, ids_restore_4 = self.random_masking(x_4, mask_ratio)
        x_5, mask_5, ids_restore_5 = self.random_masking(x_5, mask_ratio)
        x_6, mask_6, ids_restore_6 = self.random_masking(x_6, mask_ratio)
        x_7, mask_7, ids_restore_7 = self.random_masking(x_7, mask_ratio)
        x_8, mask_8, ids_restore_8 = self.random_masking(x_8, mask_ratio)
        x_9, mask_9, ids_restore_9 = self.random_masking(x_9, mask_ratio)

        ids_restore_0 = ids_restore_0*10
        ids_restore_1 = ids_restore_1*10+1
        ids_restore_2 = ids_restore_2*10+2
        ids_restore_3 = ids_restore_3*10+3
        ids_restore_4 = ids_restore_4*10+4
        ids_restore_5 = ids_restore_5*10+5
        ids_restore_6 = ids_restore_6*10+6
        ids_restore_7 = ids_restore_7*10+7
        ids_restore_8 = ids_restore_8*10+8
        ids_restore_9 = ids_restore_9*10+9

        mask = torch.cat(
            (mask_0, mask_1, mask_2, mask_3, mask_4, mask_5, mask_6, mask_7, mask_8, mask_9), dim=1)
        mask = mask.view(bs, 10, 12)
        mask = mask.permute(0, 2, 1)
        mask = mask.contiguous().view(bs, 120)

        x_masked = torch.cat((x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9), dim=1)
        x_masked = x_masked.view(bs, 10, round(12 * (1 - mask_ratio)), 256)
        x_masked = x_masked.permute(0, 2, 1, 3)
        x_masked = x_masked.contiguous().view(bs, round(120 * (1 - mask_ratio)), 256)

        # len_keep = int(12 * (1 - mask_ratio))
        # ids_keep = torch.cat((ids_restore_0[:, :len_keep], ids_restore_1[:, :len_keep], ids_restore_2[:, :len_keep],
        #                       ids_restore_3[:, :len_keep], ids_restore_4[:, :len_keep], ids_restore_5[:, :len_keep],
        #                       ids_restore_6[:, :len_keep],ids_restore_7[:, :len_keep],
        #                       ids_restore_8[:, :len_keep], ids_restore_9[:, :len_keep]),dim=1)
        # ids_mask = torch.cat((ids_restore_0[:, len_keep:], ids_restore_1[:, len_keep:], ids_restore_2[:, len_keep:],
        #                       ids_restore_3[:, len_keep:],ids_restore_4[:, len_keep:], ids_restore_5[:, len_keep:],
        #                       ids_restore_6[:, len_keep:],ids_restore_7[:, len_keep:],
        #                       ids_restore_8[:, len_keep:], ids_restore_9[:, len_keep:]),dim=1)
        # ids_restore = torch.cat((ids_keep, ids_mask), dim=1)

        ids_restore = torch.cat((ids_restore_0,ids_restore_1,ids_restore_2,ids_restore_3,ids_restore_4,ids_restore_5,
                                 ids_restore_6,ids_restore_7,ids_restore_8,ids_restore_9), dim=1)

        ids_restore = ids_restore.view(bs, 10, 12)
        ids_restore = ids_restore.permute(0, 2, 1)
        ids_restore = ids_restore.contiguous().view(bs, 120)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, epoch):

        # Embed patches
        # x = x.unsqueeze(1)
        x = self.patch_embed(x)
        # x = self.FreqFilter(x)

        bs, ps, ed = x.shape
        len_keep = round(ps * (1 - mask_ratio))

        # Add positional embeddings
        x = x + self.pos_embed[:, 1:, :]

        mask_ratio_l = 0.6
        mask_ratio_t = 0.75

        x_l, mask_l, ids_restore_l = self.LeadView_masking(x, mask_ratio=mask_ratio_l)
        x_t, mask_t, ids_restore_t = self.TimeView_masking(x, mask_ratio=mask_ratio_t)
        # xl_ = x_l
        # xt_ = x_t
        # xl_ = x_l+self.FreqFilter(x_l)
        # xt_ = x_t+self.FreqFilter(x_t)
        # x_l = x_l + self.FreqFilter(x_l) + self.FeedFoward(x_l+self.FreqFilter(x_l))
        # x_t = x_t + self.FreqFilter(x_t) + self.FeedFoward(x_t+self.FreqFilter(x_t))
        # for ff_block in self.ff_blocks:
        #     x_l = ff_block(x_l)
        # for ff_block in self.ff_blocks:
        #     x_t = ff_block(x_t)
        # x_l = x_l + xl_
        # x_t = x_t + xt_
        x_l = x_l.view(bs, 12, round(10 * (1 - mask_ratio_l)), 256)
        x_l = x_l.contiguous().view(bs * 12, round(10 * (1 - mask_ratio_l)), 256)
        x_t = x_t.view(bs, round(12 * (1 - mask_ratio_t)), 10, 256)
        x_t = x_t.permute(0, 2, 1, 3)
        x_t = x_t.contiguous().view(bs * 10, round(12 * (1 - mask_ratio_t)), 256)

        for blk in self.blocks_l:
            x_l, _ = blk(x_l)
        x_l = self.norm_l(x_l)
        x_l = x_l.view(bs, 12, round(10 * (1 - mask_ratio_l)), 256)
        x_l = x_l.view(bs, round(120 * (1 - mask_ratio_l)), 256)

        for blk in self.blocks_t:
            x_t, _ = blk(x_t)
        x_t = self.norm_t(x_t)
        x_t = x_t.view(bs, 10, round(12 * (1 - mask_ratio_t)), 256)
        x_t = x_t.permute(0, 2, 1, 3)
        x_t = x_t.contiguous().view(bs, round(120 * (1 - mask_ratio_t)), 256)

        x = torch.cat([x_l, x_t], dim=1)
        # Append CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for ff_block in self.ff_blocks:
            x = ff_block(x)

        # Transformer blocks
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.norm(x)

        return x, mask_l, ids_restore_l, mask_t, ids_restore_t

    def forward_decoder(self, x, ids_restore_l, ids_restore_t):
        mask_ratio_l = 0.6
        mask_ratio_t = 0.75

        cls_m = x[:, :1, :]
        x = x[:, 1:, :]
        x_l = x[:, :round(120 * (1 - mask_ratio_l)), :]
        x_t = x[:, round(120 * (1 - mask_ratio_l)):, :]
        x_l = torch.cat((cls_m, x_l), dim=1)
        x_t = torch.cat((cls_m, x_t), dim=1)

        # embed tokens
        x_l = self.decoder_embed_l(x_l)
        x_t = self.decoder_embed_t(x_t)

        # append mask tokens to sequence
        mask_tokens_l = self.mask_token.repeat(x_l.shape[0], ids_restore_l.shape[1] + 1 - x_l.shape[1], 1)
        xl_ = torch.cat([x_l[:, 1:, :], mask_tokens_l], dim=1)  # no cls token
        xl_ = torch.gather(xl_, dim=1, index=ids_restore_l.unsqueeze(-1).repeat(1, 1, x_l.shape[2]))  # unshuffle
        x_l = torch.cat([x_l[:, :1, :], xl_], dim=1)  # append cls token

        mask_tokens_t = self.mask_token.repeat(x_t.shape[0], ids_restore_t.shape[1] + 1 - x_t.shape[1], 1)
        xt_ = torch.cat([x_t[:, 1:, :], mask_tokens_t], dim=1)  # no cls token
        xt_ = torch.gather(xt_, dim=1, index=ids_restore_t.unsqueeze(-1).repeat(1, 1, x_t.shape[2]))  # unshuffle
        x_t = torch.cat([x_t[:, :1, :], xt_], dim=1)  # append cls token

        x_l = x_l + self.decoder_pos_embed
        x_t = x_t + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks_l:
            x_l, _ = blk(x_l)
        x_l = self.decoder_norm_l(x_l)

        for blk in self.decoder_blocks_t:
            x_t, _ = blk(x_t)
        x_t = self.decoder_norm_t(x_t)

        x_l = self.decoder_pred_l(x_l)
        x_t = self.decoder_pred_t(x_t)

        x_l = x_l[:, 1:, :]
        x_t = x_t[:, 1:, :]

        return x_l, x_t

    def forward_loss(self, imgs, pred_l, mask_l, pred_t, mask_t):
        target = self.patchify(imgs)
        loss_t = (pred_t - target) ** 2
        loss_t = loss_t.mean(dim=-1)
        loss_t = (loss_t * mask_t).sum() / mask_t.sum()
        loss_l = (pred_l - target) ** 2
        loss_l = loss_l.mean(dim=-1)
        loss_l = (loss_l * mask_l).sum() / mask_l.sum()
        loss = 1 * loss_t + 1 * loss_l
        return loss

    def forward(self, imgs, texts, epoch, mask_ratio):
        bs, _, _ = imgs.shape
        latent, mask_l, ids_restore_l, mask_t, ids_restore_t = self.forward_encoder(imgs, mask_ratio, epoch)
        describe_encode = self.describe_resize(texts)
        # describe_encode = texts.view(bs, 1, 768)
        fusion = self.cond_fusion(latent, describe_encode)
        # latent = fusion
        latent = latent + fusion
        # latent = torch.cat([latent, fusion], dim=2)
        pred_l, pred_t = self.forward_decoder(latent, ids_restore_l, ids_restore_t)
        pred_sig = self.unpatchify(pred_t)
        loss = self.forward_loss(imgs, pred_l, mask_l, pred_t, mask_t)
        return loss, pred_sig, pred_t, mask_t

def mae_vit(norm_pix_loss=False):
    # 示例：创建一个适配你的 1D 信号的模型
    model = MaskedAutoencoderViT(
        signal_length=1000,  # 输入信号长度
        patch_size=100,  
        # patch_size=50, # 每个 patch 的长度
        in_chans=12,          # 输入信号通道数
        embed_dim=256,       # 编码器嵌入维度
        depth=8,            # 编码器 Transformer 层数
        num_heads=8,        # 编码器多头数
        decoder_embed_dim=256,  # 解码器嵌入维度
        decoder_depth=8,     # 解码器 Transformer 层数
        decoder_num_heads=8,  # 解码器多头数
        mlp_ratio=4.0,       # MLP 扩展比例
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # LayerNorm
        norm_pix_loss=norm_pix_loss  # 是否归一化像素损失
    )
    return model