# 2025/06/28: 
# 这是一个使用wav2vec进行音频特征编码，输出motion latent的自回归模型，
# 注意到当前这是一个deterministic的模型，无法产生多样性的结果

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import PretrainedConfig,HubertModel
from timm.models.vision_transformer import Mlp
from transformers import PreTrainedModel, Wav2Vec2Processor, Wav2Vec2Model, PretrainedConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Attention
from typing import Optional, Tuple, Union

from diffusers.models.embeddings import Timesteps, TimestepEmbedding, get_1d_rotary_pos_embed
from diffusers import FlowMatchEulerDiscreteScheduler
class WanTimeEmbedding(nn.Module):
    """
    Modified from:
    Wan: Open and Advanced Large-Scale Video Generative Models
    https://huggingface.co/docs/diffusers/main/api/models/wan_transformer_3d
    """
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
    ):
        super().__init__()
        # generate sinusoidal time embeddings
        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        # project to model dimension
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim,
            time_embed_dim=dim,
        )

    def forward(self, timestep: torch.Tensor):  # timestep: (batch,)
        # 1. sinusoidal embedding: (batch, time_freq_dim)
        timestep = self.timesteps_proj(timestep)
        # ensure dtype matches embedder
        emb_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != emb_dtype and emb_dtype != torch.int8:
            timestep = timestep.to(emb_dtype)
        # 2. linear + activation: (batch, dim)
        temb = self.time_embedder(timestep)
        return temb

class RoPEEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout)
        # Compute the frequencies for rotary embeddings: theta_j = 10000^(-2j/d_model)
        theta = 10000 ** (-2 * torch.arange(0, d_model//2, dtype=torch.float) / d_model)
        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        angles = positions * theta
        # Precompute cosines and sines for efficiency
        self.register_buffer('cos_angles', angles.cos())
        self.register_buffer('sin_angles', angles.sin())

    def forward(self, x):
        """
        Apply rotary embeddings to the input tensor.
        Input shape: [bs, seq_len, d_model]
        Output shape: [bs, seq_len, d_model]
        """
        seq_len = x.size(1)
        # Slice precomputed angles to match the sequence length
        cos_angles = self.cos_angles[:seq_len].to(x.device)
        sin_angles = self.sin_angles[:seq_len].to(x.device)
        # Split input into even and odd indices for pairwise rotation
        x_even = x[:, :, 0::2]  # [bs, seq_len, d_model//2]
        x_odd = x[:, :, 1::2]   # [bs, seq_len, d_model//2]
        # Apply rotary transformation
        x_rot = x.clone()
        x_rot[:, :, 0::2] = x_even * cos_angles - x_odd * sin_angles
        x_rot[:, :, 1::2] = x_even * sin_angles + x_odd * cos_angles
        return self.dropout(x_rot)

class SelfAttention_Rope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_seq_len=128):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.rope = RoPEEncoding(d_model, dropout=dropout, max_seq_len=max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        """
        :param x: B x T x d_model input tensor
        :param attn_mask: B * num_heads x L x S mask with L=target sequence length, S=source sequence length
                          for a float mask: values will be added to attention weight
                          for a binary mask: True indicates that the element is not allowed to attend
        :param key_padding_mask: B x S mask
                          for a float mask: values will be added directly to the corresponding key values
                          for a binary mask: True indicates that the corresponding key value will be ignored
        :return: B x T x d_model output tensor
        """
        x = self.self_attn(
            self.rope(x),
            self.rope(x),
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = self.dropout(x)
        return x

class CrossAttention_Rope(nn.Module):
    def __init__(self, d_model: int, d_cond: int, num_heads: int, dropout: float = 0.1, max_seq_len=128):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=d_cond,
            vdim=d_cond,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.rope = RoPEEncoding(d_model, dropout=dropout, max_seq_len=max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        """
        :param x: B x T_target x d_model input tensor
        :param cond: B x T_cond x d_cond condition tensor
        :param attn_mask: B * num_heads x L x S mask with L=target sequence length, S=source sequence length
                          for a float mask: values will be added to attention weight
                          for a binary mask: True indicates that the element is not allowed to attend
        :param key_padding_mask: B x S mask
                          for a float mask: values will be added directly to the corresponding key values
                          for a binary mask: True indicates that the corresponding key value will be ignored
        :return: B x T x d_model output tensor
        """
        x = self.cross_attn(
            self.rope(x),
            self.rope(cond),
            cond,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = self.dropout(x)
        return x

class SelfAttention_Pos(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_seq_len=128):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.pe = PositionalEncoding(
            d_model, dropout=dropout, max_seq_len=max_seq_len
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        """
        :param x: B x T x d_model input tensor
        :param attn_mask: B * num_heads x L x S mask with L=target sequence length, S=source sequence length
                          for a float mask: values will be added to attention weight
                          for a binary mask: True indicates that the element is not allowed to attend
        :param key_padding_mask: B x S mask
                          for a float mask: values will be added directly to the corresponding key values
                          for a binary mask: True indicates that the corresponding key value will be ignored
        :return: B x T x d_model output tensor
        """
        x = self.self_attn(
            self.pe(x),
            self.pe(x),
            self.pe(x),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = self.dropout(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_seq_len=128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class AutoModelConfig(PretrainedConfig):
    def __init__(self, config_obj=None, **kwargs):
        if config_obj is not None:
            cfg_dict = OmegaConf.to_container(config_obj, resolve=True)
            kwargs.update(cfg_dict)
            self.model_type = kwargs.pop("model_type", "my_model")
        super().__init__(**kwargs)

class Audio2FaceGPTBlock(nn.Module):
    """
    GPT decoder block for Audio2Face generation with causal attention.
    包含 SelfAttention_Rope -> CrossAttention_Rope -> SelfAttention_Pos -> FFN
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1, max_seq_len=128):
        super().__init__()
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.norm4 = nn.LayerNorm(hidden_size)
        
        # Attention layers
        self.self_attn_rope = SelfAttention_Rope(hidden_size, num_heads, dropout, max_seq_len=max_seq_len)
        #self.cross_attn_rope = CrossAttention_Rope(hidden_size, hidden_size, num_heads, dropout, max_seq_len=max_seq_len)
        self.self_attn_pos = SelfAttention_Pos(hidden_size, num_heads, dropout, max_seq_len=max_seq_len)
        
        # FFN
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)

    def forward(self, x, audio_features, causal_mask=None, cross_causal_mask=None):
        """
        :param x: [bs, seq_len, hidden_size] face latent features
        :param audio_features: [bs, seq_len, hidden_size] audio features
        :param anchor_hidden: [bs, 1, hidden_size] anchor latent features
        :param causal_mask: [seq_len, seq_len] causal mask for self attention
        :param cross_causal_mask: [seq_len, seq_len] causal mask for cross attention
        """
        # Self attention with RoPE
        residual = x
        x = self.norm1(x)
        x = self.self_attn_rope(x, attn_mask=causal_mask)
        x = residual + x
        
        # Cross attention with audio features
        residual = x
        x = self.norm2(x)
        x = audio_features + x
        x = residual + x
        
        # Self attention with positional encoding
        residual = x
        x = self.norm3(x)
        x = self.self_attn_pos(x, attn_mask=causal_mask)
        x = residual + x
        
        # FFN
        residual = x
        x = self.norm4(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


def make_attention_causal(attn: Wav2Vec2Attention):
    q_proj, k_proj, v_proj, out_proj = attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj
    n_head, head_dim, p = attn.num_heads, attn.head_dim, attn.dropout

    def f(self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **_):
        B, T, _ = x.shape
        q = q_proj(x).view(B, T, n_head, head_dim).transpose(1, 2)
        k = k_proj(x).view(B, T, n_head, head_dim).transpose(1, 2)
        v = v_proj(x).view(B, T, n_head, head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=p if self.training else 0.0,
            is_causal=False
        )
        y = out_proj(y.transpose(1, 2).reshape(B, T, n_head * head_dim))
        return (y, None, None) if output_attentions else (y, None, None)

    attn.forward = f.__get__(attn, attn.__class__)

class WrapedWav2Vec(nn.Module):
    def __init__(self, layers: int = 8):
        super().__init__()
        base = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.feature_extractor = base.feature_extractor
        self.feature_projection = base.feature_projection
        self.encoder = base.encoder
        self.encoder.layers = self.encoder.layers[:layers]
        for l in self.encoder.layers:
            make_attention_causal(l.attention)

    def forward(
        self,
        x: torch.Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        **_
    ):
        low = self.feature_extractor(x).transpose(1, 2)
        h, _ = self.feature_projection(low.detach())
        enc = self.encoder(
            h,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return {"low_level": low, "high_level": enc[0]}


class DiffusionBlock(nn.Module):
    """
    Diffusion Block 使用现有的注意力组件
    包含: SelfAttention_Rope -> CrossAttention_Rope (with GPT) -> CrossAttention_Rope (with past frames) -> SelfAttention_Pos -> FFN
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1, max_seq_len=128):
        super().__init__()
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size,elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size,elementwise_affine=False)
        
        # FFN
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp1 = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)
        
        self.mlp2 = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)
        

        self.adaLN_modulation1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        
        self.adaLN_modulation2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        
        
    def forward(self, hidden_states, gpt_hidden, temb=None):
        """
        :param x: [bs, 1, hidden_size] 当前帧的noisy特征
        :param gpt_hidden: [bs, 1, hidden_size] GPT输出的条件
        :param past_hidden: [bs, T_past, hidden_size] 历史帧的条件
        :param time_emb: [bs, 1, hidden_size] 时间嵌入（可选）
        """
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation1(temb).chunk(3, dim=-1)
        
        shift_mlp2, scale_mlp2, gate_mlp2 = self.adaLN_modulation2(gpt_hidden).chunk(3, dim=-1)
        
        

        # 4. Feed-forward
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_mlp) + shift_mlp).type_as(hidden_states)
        ff_output = self.mlp1(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp).type_as(hidden_states)

        norm_hidden_states = (self.norm2(hidden_states.float()) * (1 + scale_mlp2) + shift_mlp2).type_as(hidden_states)
        ff_output = self.mlp2(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp2).type_as(hidden_states)

        return hidden_states
        


class DiffusionHead(nn.Module):
    """
    Diffusion Head使用现有组件，用于去噪face latent
    输入:
        - noisy_face_latent: [bs, 1, face_dim] 当前帧的加噪face latent
        - gpt_output: [bs, 1, face_dim] 当前帧的GPT输出
        - past_gt_frames: [bs, T-1, face_dim] 前面所有帧的GT
        - timestep: [bs, 1] 噪声时间步（可选）
    输出:
        - denoised_face_latent: [bs, 1, face_dim] 去噪后的face latent
    """
    def __init__(
        self,
        face_dim=512,
        hidden_size=768,
        num_layers=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        max_seq_len=128,
    ):
        super().__init__()
        self.face_dim = face_dim
        self.hidden_size = hidden_size
        
        # 输入投影层
        self.noisy_proj = nn.Linear(face_dim, hidden_size)
        self.gpt_proj = nn.Linear(face_dim, hidden_size)
        self.past_proj = nn.Linear(face_dim, hidden_size)
        self.anchor_proj = nn.Linear(face_dim, hidden_size)
        
        
        # Diffusion blocks
        self.blocks = nn.ModuleList([
            DiffusionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, face_dim)
        
        
    def forward(self, noisy_face_latent, gpt_output, temb=None):
        """
        :param noisy_face_latent: [bs, 1, face_dim] 当前帧的加噪face latent
        :param gpt_output: [bs, 1, face_dim] 当前帧的GPT输出
        :param past_gt_frames: [bs, T-1, face_dim] 前面所有帧的GT
        :param timestep: [bs, 1] 噪声时间步（可选）
        """
        bs = noisy_face_latent.shape[0]
        device = noisy_face_latent.device
                
        # 投影各个输入
        noisy_hidden = self.noisy_proj(noisy_face_latent)  # [bs, 1, hidden_size]
        gpt_hidden = self.gpt_proj(gpt_output)  # [bs, 1, hidden_size]
        
        # 通过Diffusion blocks处理
        x = noisy_hidden
        for block in self.blocks:
            x = block(x, gpt_hidden, temb)
        
        # 输出投影
        x = self.output_norm(x)
        denoised = self.output_proj(x)  # [bs, 1, face_dim]
        
        return denoised

def pad_audio(audio, audio_unit=320, pad_threshold=80):
    # audio: (B, S) 16k waveform, we center-pad a bit for safer conv receptive field
    batch_size, audio_len = audio.shape
    n_units = audio_len // audio_unit
    side_len = math.ceil((audio_unit * n_units + pad_threshold - audio_len) / 2)
    if side_len >= 0:
        reflect_len = side_len // 2
        replicate_len = side_len % 2
        if reflect_len > 0:
            audio = F.pad(audio, (reflect_len, reflect_len), mode='reflect')
            audio = F.pad(audio, (reflect_len, reflect_len), mode='reflect')
        if replicate_len > 0:
            audio = F.pad(audio, (1, 1), mode='replicate')
    return audio



class Audio2FaceGPT(nn.Module):
    """
    GPT自回归模型，从audio特征生成face latent
    输入: 
        - audio2face_fea [bs, 24, 768]
        - anchor_latent [bs, 1, 512] 锚点latent
    输出: face_latent [bs, 24, 512]
    """
    def __init__(
        self,
        cfg=None,
        audio_dim=768,
        face_dim=52,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        max_seq_len=1024,
        diffusion_num_layers=6,
    ):
        super().__init__()
        self.cfg = cfg
        self.audio_encoder = HubertModel.from_pretrained('./ckpts/hf_transformer_mhubert_base_vp_en_es_fr_it3')
        # freeze feature extractor and some early layers
        self.audio_encoder.feature_extractor._freeze_parameters()
        frozen_layers = [0, 1]
        for name, param in self.audio_encoder.named_parameters():
            if name.startswith("feature_projection") or name.startswith("feature_extractor"):
                param.requires_grad = False
            if name.startswith("encoder.layers"):
                layer = int(name.split(".")[2])
                if layer in frozen_layers:
                    param.requires_grad = False

        #self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_dim = audio_dim
        self.face_dim = face_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # 输入投影层
        self.audio_proj = nn.Linear(audio_dim, hidden_size)

        
        self.face_embed = nn.Linear(face_dim, hidden_size)

        
        self.time_embed = WanTimeEmbedding(
            dim=hidden_size,
            time_freq_dim=hidden_size,
        )
        # GPT decoder blocks
        self.blocks = nn.ModuleList([
            Audio2FaceGPTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])
        
        # 输出投影层
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, face_dim)
        
        self.diffusion_head = DiffusionHead(
            face_dim=face_dim,
            hidden_size=hidden_size,
            num_layers=diffusion_num_layers,  # 可以调整
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        
        self.cfg_audio = 2
        self.drop_audio = 0.1
        

    
    def generate_causal_mask(self, seq_len, device):
        """生成causal mask，上三角为-inf"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def generate_cross_causal_mask(self, seq_len, device):
        """生成cross attention的causal mask，确保第i帧只能看到前i帧的音频"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    

    def extract_audio_feature(self, audio, frame_num=None):
        frame_num = frame_num or 64
        hidden_states = self.audio_encoder(pad_audio(audio)).last_hidden_state  # (B, L, 768)
        hidden_states = hidden_states.transpose(1, 2)  # (B, 768, L)
        hidden_states = F.interpolate(hidden_states, size=frame_num, align_corners=False, mode='linear')  # (B, 768, T)
        hidden_states = hidden_states.transpose(1, 2)  # (B, T, 768)
        audio_feat = self.audio_proj(hidden_states)  # (B, T, D)
        return audio_feat

    def forward(self,audio,motion,noise_motion,time_step):
        """
        :param audio_features: [bs, seq_len, 768] 音频特征
        :param anchor_latent: [bs, 1, 512] 锚点latent
        :param face_latent_gt: [bs, seq_len, 512] ground truth face latent (用于teacher forcing)
        :return: [bs, seq_len, 512] 预测的face latent
        """
        bs, n, _ = motion.shape


        audio_features = self.extract_audio_feature(audio)
        bs, seq_len, _ = audio_features.shape
        device = audio_features.device
        # 投影音频特征和锚点    
        audio_hidden = audio_features      
        audio_hidden = audio_hidden[:,1:]
        drop_audio_mask = torch.rand(bs,seq_len-1,1,device =motion.device)<self.drop_audio
        drop_audio_mask = drop_audio_mask.float()
        audio_hidden = audio_hidden*(1-drop_audio_mask)
        
                # 生成causal masks
        causal_mask = self.generate_causal_mask(seq_len-1, device)
        cross_causal_mask = self.generate_cross_causal_mask(seq_len-1, device)
        
        
        face_hidden = self.face_embed(motion)  # [bs, seq_len, hidden_size]
            

        
        
        x = face_hidden
        for block in self.blocks:
            x = block(x, 
                      audio_hidden, 
                      causal_mask, 
                      cross_causal_mask)
        
        # 输出投影
        x = self.output_norm(x)
        gpt_output = self.output_proj(x)  # [bs, seq_len, face_dim]
        

        time_embedding = self.time_embed(time_step).unsqueeze(1)
    
        # 通过diffusion head去噪
        output = self.diffusion_head(
            noise_motion, 
            gpt_output, 
            temb=time_embedding if time_step is not None else None,
        )
        

        
        return output

    @torch.no_grad()
    def infer_from_audio(self, audio,pre_motion_token,gen_len=1,pre_audio=None,num_inference_steps=1):
        noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_karras_sigmas=False
        )
        device = pre_motion_token.device

        noise_scheduler.set_timesteps(num_inference_steps, device=device)
        
        timesteps = noise_scheduler.timesteps
        gen_tokens = []
        if_categorial = False
        
        if pre_audio is not None:
            audio = torch.cat([pre_audio,audio],dim=1)

        audio_feat = self.extract_audio_feature(audio,frame_num=63)
        audio_hidden = audio_feat
        audio_hidden = torch.cat([audio_hidden*0,audio_hidden],dim=0)   
        assert audio_feat.shape[1] == len(pre_motion_token)+gen_len-1
        
        for i in range(gen_len):
        
            input_motion_token = torch.cat([pre_motion_token,torch.concat(gen_tokens,dim=-2)],dim=-2) if len(gen_tokens)>0 else pre_motion_token
            T = pre_motion_token.shape[0]+i
            motion_feat = input_motion_token
            motion_feat = self.face_embed(motion_feat) 
            # 生成causal masks
            causal_mask = self.generate_causal_mask(T, device)
            cross_causal_mask = self.generate_cross_causal_mask(T, device)
            x = motion_feat.unsqueeze(0).repeat(2,1,1)
            for block in self.blocks:
                x = block(x, audio_hidden[:, :T, :], causal_mask, cross_causal_mask)
            # 输出投影
            x = self.output_norm(x[:, -1:])  
            gpt_output_t = self.output_proj(x)
            bs=1

            # 2. 使用Diffusion进行去噪
            if noise_scheduler is not None:
                # 初始化噪声latent
                noise_scheduler.set_timesteps(num_inference_steps, device=device)
                latent_t = torch.randn_like(gpt_output_t[:bs])
    
                # Denoising loop
                for i, timestep in enumerate(timesteps):
                    # 扩展timestep维度
                    t_batch = torch.full((bs,), timestep, device=device, dtype=torch.long)
                    
                    # Scale model input
                    latent_model_input = latent_t
                    
                    time_embedding = self.time_embed(t_batch).unsqueeze(1)

                    # 通过diffusion head去噪
                    output_batch = self.diffusion_head(
                        latent_model_input,
                        gpt_output_t, 
                        temb=time_embedding,
                    )
                    
                    
                    # Split predictions using chunk
                    noise_pred_uncond,noise_pred_cond_audio= output_batch.chunk(2, dim=0)
                    
                    # Apply CFG in batch
                    noise_pred = noise_pred_uncond + \
                        self.cfg_audio * (noise_pred_cond_audio - noise_pred_uncond) 
                
                    
                    sigma_idx = noise_scheduler.step_index
                    if sigma_idx is None: 
                        noise_scheduler._init_step_index(timestep)
                        sigma_idx = noise_scheduler.step_index
                    sigma = noise_scheduler.sigmas[sigma_idx].to(device=device)
                    velocity = (latent_t - noise_pred) / (sigma + 1e-9)
                    
                    latent_t = noise_scheduler.step(
                        velocity, timestep, latent_t, return_dict=False
                    )[0]
                
                
                # 使用去噪后的结果
                denoised_output_t = latent_t

            
            gen_tokens.append(denoised_output_t.squeeze(0))

        gen_tokens = torch.cat(gen_tokens,dim=-2)
        return gen_tokens
        

    @torch.no_grad()
    def infer_from_audio1(self, audio, pre_motion_token, gen_len=1, pre_audio=None, num_inference_steps=5):
        device = pre_motion_token.device
        
        # ====== 1. 预创建 scheduler 并缓存到 GPU ======
        if not hasattr(self, '_noise_scheduler'):
            self._noise_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_karras_sigmas=False
            )
        noise_scheduler = self._noise_scheduler
        noise_scheduler.set_timesteps(num_inference_steps, device=device)
        
        # ====== 2. 预缓存 timesteps 和 sigmas 到 GPU ======
        timesteps = noise_scheduler.timesteps.to(device)
        sigmas = noise_scheduler.sigmas.to(device)
        
        # ====== 3. 预创建最大尺寸的 causal mask（复用） ======
        max_T = pre_motion_token.shape[0] + gen_len
        if not hasattr(self, '_cached_causal_mask') or self._cached_causal_mask.shape[0] < max_T:
            self._cached_causal_mask = self.generate_causal_mask(max_T, device)
            self._cached_cross_mask = self.generate_cross_causal_mask(max_T, device)
        
        # ====== 4. 预分配输出 tensor（避免 append + cat） ======
        gen_tokens = torch.empty(gen_len, self.face_dim, device=device, dtype=pre_motion_token.dtype)
        
        # ====== 5. 预创建 timestep batch tensor ======
        bs = 1
        t_batch_template = torch.empty(bs, device=device, dtype=torch.long)
        
        if pre_audio is not None:
            audio = torch.cat([pre_audio, audio], dim=1)

        audio_feat = self.extract_audio_feature(audio, frame_num=63)
        audio_hidden = audio_feat
        audio_hidden = torch.cat([audio_hidden * 0, audio_hidden], dim=0)
        
        # 用于累积生成的 tokens
        all_motion = pre_motion_token.clone()
        
        for frame_idx in range(gen_len):
            T = pre_motion_token.shape[0] + frame_idx
            motion_feat = self.face_embed(all_motion)
            
            # ====== 6. 使用预缓存的 mask 切片（无新分配） ======
            causal_mask = self._cached_causal_mask[:T, :T]
            cross_causal_mask = self._cached_cross_mask[:T, :T]
            
            x = motion_feat.unsqueeze(0).expand(2, -1, -1)  # expand 不复制数据
            
            for block in self.blocks:
                x = block(x, audio_hidden[:, :T, :], causal_mask, cross_causal_mask)
            
            x = self.output_norm(x[:, -1:])
            gpt_output_t = self.output_proj(x)

            # ====== 7. 预生成随机噪声，复用 shape ======
            latent_t = torch.randn(bs, 1, self.face_dim, device=device, dtype=gpt_output_t.dtype)

            noise_scheduler._step_index = None

            for step_idx, timestep in enumerate(timesteps):
                # ====== 8. 就地填充而非创建新 tensor ======
                t_batch_template.fill_(timestep)
                
                latent_model_input = latent_t
                time_embedding = self.time_embed(t_batch_template).unsqueeze(1)

                output_batch = self.diffusion_head(
                    latent_model_input,
                    gpt_output_t,
                    temb=time_embedding,
                )
                
                noise_pred_uncond, noise_pred_cond_audio = output_batch.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + self.cfg_audio * (noise_pred_cond_audio - noise_pred_uncond)

                # ====== 9. 直接使用预缓存的 sigma ======
                sigma_idx = noise_scheduler.step_index
                if sigma_idx is None:
                    noise_scheduler._init_step_index(timestep)
                    sigma_idx = noise_scheduler.step_index
                sigma = sigmas[sigma_idx]  # 直接索引，已在 GPU
                
                velocity = (latent_t - noise_pred) / (sigma + 1e-9)
                latent_t = noise_scheduler.step(velocity, timestep, latent_t, return_dict=False)[0]

            # ====== 10. 直接写入预分配的 tensor ======
            gen_tokens[frame_idx] = latent_t.squeeze()
            
            # 更新 all_motion 用于下一帧
            all_motion = torch.cat([all_motion, latent_t.squeeze(0)], dim=0)

        return gen_tokens




    @torch.no_grad()
    def infer_from_audio2(self, audio, pre_motion_token, gen_len=1, pre_audio=None, num_inference_steps=5):
        device = pre_motion_token.device
        
        # ====== 1. 预创建 scheduler 并缓存到 GPU ======
        if not hasattr(self, '_noise_scheduler'):
            self._noise_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_karras_sigmas=False
            )
        noise_scheduler = self._noise_scheduler
        noise_scheduler.set_timesteps(num_inference_steps, device=device)
        
        # ====== 2. 预缓存 timesteps 和 sigmas 到 GPU ======
        timesteps = noise_scheduler.timesteps.to(device)
        sigmas = noise_scheduler.sigmas.to(device)
        
        # ====== 3. 预创建最大尺寸的 causal mask（复用） ======
        max_T = pre_motion_token.shape[0] + gen_len
        if not hasattr(self, '_cached_causal_mask') or self._cached_causal_mask.shape[0] < max_T:
            self._cached_causal_mask = self.generate_causal_mask(max_T, device)
            self._cached_cross_mask = self.generate_cross_causal_mask(max_T, device)
        
        # ====== 4. 预分配输出 tensor（避免 append + cat） ======
        gen_tokens = torch.empty(gen_len, self.face_dim, device=device, dtype=pre_motion_token.dtype)
        
        # ====== 5. 预创建 timestep batch tensor ======
        bs = 1
        t_batch_template = torch.empty(bs, device=device, dtype=torch.long)
        
        if pre_audio is not None:
            audio = torch.cat([pre_audio, audio], dim=1)

        audio_feat = self.extract_audio_feature(audio, frame_num=63)
        audio_hidden = audio_feat
        audio_hidden = torch.cat([audio_hidden * 0, audio_hidden], dim=0)
        
        # 用于累积生成的 tokens
        all_motion = pre_motion_token.clone()
        
        for frame_idx in range(gen_len):
            T = pre_motion_token.shape[0] + frame_idx
            motion_feat = self.face_embed(all_motion)
            
            # ====== 6. 使用预缓存的 mask 切片（无新分配） ======
            causal_mask = self._cached_causal_mask[:T, :T]
            cross_causal_mask = self._cached_cross_mask[:T, :T]
            
            x = motion_feat.unsqueeze(0).expand(2, -1, -1)  # expand 不复制数据
            for block in self.blocks:
                x = block(x, audio_hidden[:, :T, :], causal_mask, cross_causal_mask)
            
            x = self.output_norm(x[:, -1:])
            gpt_output_t = self.output_proj(x)

            # ====== 7. 预生成随机噪声，复用 shape ======
            latent_t = torch.randn(bs, 1, self.face_dim, device=device, dtype=gpt_output_t.dtype)

            noise_scheduler._step_index = None

            for step_idx, timestep in enumerate(timesteps):
                # ====== 8. 就地填充而非创建新 tensor ======
                t_batch_template.fill_(timestep)
                
                latent_model_input = latent_t
                time_embedding = self.time_embed(t_batch_template).unsqueeze(1)

                output_batch = self.diffusion_head(
                    latent_model_input,
                    gpt_output_t,
                    temb=time_embedding,
                )
                
                noise_pred_uncond, noise_pred_cond_audio = output_batch.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + self.cfg_audio * (noise_pred_cond_audio - noise_pred_uncond)

                # ====== 9. 直接使用预缓存的 sigma ======
                sigma_idx = noise_scheduler.step_index
                if sigma_idx is None:
                    noise_scheduler._init_step_index(timestep)
                    sigma_idx = noise_scheduler.step_index
                sigma = sigmas[sigma_idx]  # 直接索引，已在 GPU
                
                velocity = (latent_t - noise_pred) / (sigma + 1e-9)
                latent_t = noise_scheduler.step(velocity, timestep, latent_t, return_dict=False)[0]

            # ====== 10. 直接写入预分配的 tensor ======
            gen_tokens[frame_idx] = latent_t.squeeze()
            
            # 更新 all_motion 用于下一帧
            all_motion = torch.cat([all_motion, latent_t.squeeze(0)], dim=0)

        return gen_tokens