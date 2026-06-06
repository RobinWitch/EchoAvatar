import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Basic Causal Convolution ---
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        # For a causal convolution, padding is only on the left
        self.padding = (kernel_size - 1) * dilation + 1 - stride
        # Standard Conv1d with padding=0 because we handle it manually
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # x: (N, C, L)
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0)) # Pad (left, right) for the last dimension (L)
        return self.conv(x)

# --- Activation Function (SiLU/Swish) ---
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def get_activation_fn(activation_str):
    if activation_str == "relu":
        return nn.ReLU()
    elif activation_str == "silu" or activation_str == "swish":
        return Swish()
    elif activation_str == "gelu":
        return nn.GELU()
    else:
        return nn.Identity()
    
# --- Causal Residual Convolution Block ---
class ResConv1DBlock_Causal(nn.Module):
    def __init__(self, n_in, n_state, kernel_size=3, dilation=1, activation='silu', norm=None, dropout=0.0):
        super().__init__()
        self.norm_type = norm

        if norm == "LN": # LayerNorm
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_state) # Norm applied on n_state before conv2
        elif norm == "GN": # GroupNorm
            self.norm1 = nn.GroupNorm(num_groups=min(32, n_in), num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=min(32, n_state), num_channels=n_state, eps=1e-6, affine=True)
        elif norm == "BN": # BatchNorm
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_state, eps=1e-6, affine=True)
        else: # No norm
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.activation1 = get_activation_fn(activation)
        self.activation2 = get_activation_fn(activation) # Activation after second norm

        self.conv1 = CausalConv1d(n_in, n_state, kernel_size, dilation=dilation)
        # The second convolution in a ResBlock is often a 1x1 pointwise conv
        self.conv2 = CausalConv1d(n_state, n_in, kernel_size=1) # Kernel size 1, no dilation needed
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x: (N, C, L)
        x_orig = x

        # First block
        if self.norm_type == "LN":
            h = self.norm1(x.transpose(-2, -1)).transpose(-2, -1)
        else:
            h = self.norm1(x)
        h = self.activation1(h)
        h = self.conv1(h)

        # Second block
        if self.norm_type == "LN":
            h = self.norm2(h.transpose(-2, -1)).transpose(-2, -1)
        else:
            h = self.norm2(h)
        h = self.activation2(h)
        h = self.conv2(h)
        
        h = self.dropout(h)
        x = x_orig + h
        return x
    
# --- Causal ResNet1D Stack ---
class Resnet1D_Causal(nn.Module):
    def __init__(self, n_in, n_depth, kernel_size=3, dilation_growth_rate=1,
                 reverse_dilation=False, activation='silu', norm=None, dropout=0.0, n_state_factor=1):
        super().__init__()
        n_state = int(n_in * n_state_factor) # Bottleneck/expansion factor for state dim
        blocks = [ResConv1DBlock_Causal(n_in, n_state, kernel_size,
                                     dilation=dilation_growth_rate ** depth,
                                     activation=activation, norm=norm, dropout=dropout)
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
    
# --- Causal Encoder ---
class Encoder_Causal(nn.Module):
    def __init__(self, input_emb_width=3, output_emb_width=512, down_t=2, stride_t=2,
                 width=512, depth=3, kernel_size_initial=7, kernel_size_res=3,
                 dilation_growth_rate=3, activation='silu', norm=None, dropout=0.0):
        super().__init__()
        
        blocks = []
        
        # 1. Initial Convolution (adjust kernel_size and padding for causality)
        # This conv increases channels from input_emb_width to width
        blocks.append(CausalConv1d(input_emb_width, width, kernel_size=kernel_size_initial, stride=1))
        blocks.append(get_activation_fn(activation)) # Activation after initial conv
        
        current_channels = width
        
        # 2. Downsampling Stages
        for i in range(down_t):
            # Downsampling Conv: kernel size for strided conv is often stride_t or stride_t*2
            # Let's make it configurable, e.g., default to stride_t*2 like Jukebox
            ds_kernel_size = stride_t * 2 
            
            # Causal strided convolution for downsampling
            downsample_conv = CausalConv1d(current_channels, width, kernel_size=ds_kernel_size, stride=stride_t)
            
            # ResNet block
            resnet_block = Resnet1D_Causal(width, depth, kernel_size=kernel_size_res,
                                           dilation_growth_rate=dilation_growth_rate,
                                           activation=activation, norm=norm, dropout=dropout)
            
            blocks.append(nn.Sequential(downsample_conv, resnet_block))
            current_channels = width # In this Jukebox-like design, width often stays constant

        # 3. Final Convolution to output_emb_width
        blocks.append(CausalConv1d(width, output_emb_width, kernel_size=3, stride=1))
        # No activation typically after the final projection to latent space in an AE
        
        self.model = nn.Sequential(*blocks)
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width

    def forward(self, x):
        # x: (batch, seq_len, feat_dim) -> (B, L, C_in)
        x = x.permute(0, 2, 1) # (B, C_in, L) for Conv1D
        x = self.model(x)      # (B, C_out, L_downsampled)
        x = x.permute(0, 2, 1) # (B, L_downsampled, C_out) for consistency with input/output
        return x
    
# --- Causal Decoder ---
class Decoder_Causal(nn.Module):
    def __init__(self, input_emb_width=3, # This is the final output channel dim
                 latent_emb_width=512,   # This is the input from encoder (output_emb_width of encoder)
                 up_t=2, stride_t=2,
                 width=512, depth=3, kernel_size_initial=3, kernel_size_res=3,
                 dilation_growth_rate=3, activation='silu', norm=None, dropout=0.0):
        super().__init__()
        
        blocks = []
        
        # 1. Initial Convolution (from latent_emb_width to width)
        blocks.append(CausalConv1d(latent_emb_width, width, kernel_size=kernel_size_initial, stride=1))
        blocks.append(get_activation_fn(activation))
        
        current_channels = width
        
        # 2. Upsampling Stages
        for i in range(up_t):
            # ResNet block (dilation reversed for decoder often)
            resnet_block = Resnet1D_Causal(width, depth, kernel_size=kernel_size_res,
                                           dilation_growth_rate=dilation_growth_rate,
                                           reverse_dilation=True, # Usually True for decoders
                                           activation=activation, norm=norm, dropout=dropout)
            
            # Upsampling layer (e.g., TransposedConv or Upsample + Conv)
            # Using Upsample + CausalConv1d is generally more stable and easier for causality
            # The kernel for conv after upsample is often 3 or related to stride
            upsample_conv_kernel = stride_t * 2 -1 if stride_t > 1 else 3 # e.g. k=3 for s=2 to cover new samples
            if stride_t == 1: upsample_conv_kernel = 3 # default

            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride_t, mode='nearest'), # Nearest is causal-friendly
                CausalConv1d(current_channels, width, kernel_size=upsample_conv_kernel, stride=1)
            )
            
            blocks.append(nn.Sequential(resnet_block, upsample_layer, get_activation_fn(activation)))
            current_channels = width

        # 3. Final Convolution to reconstruct to input_emb_width (original feature dim)
        # Your example has Conv -> ReLU -> Conv here
        blocks.append(CausalConv1d(width, width, kernel_size=3, stride=1))
        blocks.append(get_activation_fn(activation))
        blocks.append(CausalConv1d(width, input_emb_width, kernel_size=3, stride=1))
        # Typically no activation on the very final output of a reconstructive AE
        
        self.model = nn.Sequential(*blocks)
        self.latent_emb_width = latent_emb_width
        self.output_channel_dim = input_emb_width


    def forward(self, x):
        # x: (batch, seq_len_latent, latent_dim) -> (B, L_latent, C_latent)
        x = x.permute(0, 2, 1) # (B, C_latent, L_latent) for Conv1D
        x = self.model(x)      # (B, C_out, L_reconstructed)
        x = x.permute(0, 2, 1) # (B, L_reconstructed, C_out)
        return x
    
# --- Complete Causal Autoencoder (Jukebox-Style) ---
class CausalConvAutoencoder_Jukebox(nn.Module):
    def __init__(self, feat_dim=315, latent_dim=512,
                 num_downsampling_stages=2, stride_t=2, # Determines seq_len_factor
                 width=512, depth_res_blocks=3,
                 kernel_size_initial_enc=3, kernel_size_initial_dec=3,
                 kernel_size_res=3, dilation_growth_rate=2,
                 activation='relu', norm='LN', dropout=0.2):
        super().__init__()
        
        self.encoder = Encoder_Causal(
            input_emb_width=feat_dim,
            output_emb_width=latent_dim,
            down_t=num_downsampling_stages,
            stride_t=stride_t,
            width=width,
            depth=depth_res_blocks,
            kernel_size_initial=kernel_size_initial_enc,
            kernel_size_res=kernel_size_res,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
            dropout=dropout
        )
        
        self.decoder = Decoder_Causal(
            input_emb_width=feat_dim, # Final output channels
            latent_emb_width=latent_dim, # Input from encoder
            up_t=num_downsampling_stages, # Should match encoder's down_t
            stride_t=stride_t,          # Should match encoder's stride_t
            width=width,
            depth=depth_res_blocks,
            kernel_size_initial=kernel_size_initial_dec,
            kernel_size_res=kernel_size_res,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
            dropout=dropout
        )

    def forward(self, x):
        # x: (B, L, C_feat)
        encoded = self.encoder(x) # (B, L_latent, C_latent)
        decoded = self.decoder(encoded) # (B, L, C_feat)
        return decoded


from models.vq.residual_vq import ResidualVQ
# --- Complete Causal Autoencoder (Jukebox-Style) ---
class CausalConvVQVAE(nn.Module):
    def __init__(self,args, feat_dim=315, latent_dim=512,
                 num_downsampling_stages=2, stride_t=2, # Determines seq_len_factor
                 width=512, depth_res_blocks=3,
                 kernel_size_initial_enc=3, kernel_size_initial_dec=3,
                 kernel_size_res=3, dilation_growth_rate=2,
                 activation='relu', norm='LN', dropout=0.2, nb_code=512, code_dim=512):
        super().__init__()
        
        self.encoder = Encoder_Causal(
            input_emb_width=feat_dim,
            output_emb_width=latent_dim,
            down_t=num_downsampling_stages,
            stride_t=stride_t,
            width=width,
            depth=depth_res_blocks,
            kernel_size_initial=kernel_size_initial_enc,
            kernel_size_res=kernel_size_res,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
            dropout=dropout
        )
        
        self.decoder = Decoder_Causal(
            input_emb_width=feat_dim, # Final output channels
            latent_emb_width=latent_dim, # Input from encoder
            up_t=num_downsampling_stages, # Should match encoder's down_t
            stride_t=stride_t,          # Should match encoder's stride_t
            width=width,
            depth=depth_res_blocks,
            kernel_size_initial=kernel_size_initial_dec,
            kernel_size_res=kernel_size_res,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
            dropout=dropout
        )

        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)

    def forward(self, x):
        # x: (B, L, C_feat)
        encoded = self.encoder(x)
        x_ = encoded.permute(0, 2, 1).contiguous()
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_, sample_codebook_temp=0.5)
        x_quantized = x_quantized.permute(0, 2, 1).contiguous()
        decoded = self.decoder(x_quantized)

        return  {
            'rec_pose': decoded,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
        }
        
    def encode(self, x):
        # x: (B, L, C_feat)
        encoded = self.encoder(x)
        x_ = encoded.permute(0, 2, 1).contiguous()
        code_idx, all_codes = self.quantizer.quantize(x_, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes
    
    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        x = x_d.sum(dim=0)
        decoded = self.decoder(x)
        return decoded

class CausalConvAE(nn.Module):
    def __init__(self,args, feat_dim=315, latent_dim=512,
                 num_downsampling_stages=2, stride_t=2, # Determines seq_len_factor
                 width=512, depth_res_blocks=3,
                 kernel_size_initial_enc=3, kernel_size_initial_dec=3,
                 kernel_size_res=3, dilation_growth_rate=2,
                 activation='relu', norm='LN', dropout=0.2, nb_code=512, code_dim=512):
        super().__init__()
        
        self.encoder = Encoder_Causal(
            input_emb_width=feat_dim,
            output_emb_width=latent_dim,
            down_t=num_downsampling_stages,
            stride_t=stride_t,
            width=width,
            depth=depth_res_blocks,
            kernel_size_initial=kernel_size_initial_enc,
            kernel_size_res=kernel_size_res,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
            dropout=dropout
        )
        
        self.decoder = Decoder_Causal(
            input_emb_width=feat_dim, # Final output channels
            latent_emb_width=latent_dim, # Input from encoder
            up_t=num_downsampling_stages, # Should match encoder's down_t
            stride_t=stride_t,          # Should match encoder's stride_t
            width=width,
            depth=depth_res_blocks,
            kernel_size_initial=kernel_size_initial_dec,
            kernel_size_res=kernel_size_res,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
            dropout=dropout
        )

        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)

    def forward(self, x):
        # x: (B, L, C_feat)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        commit_loss = torch.tensor(0.0, device=x.device)
        perplexity = torch.tensor(0.0, device=x.device)

        
        return  {
            'rec_pose': decoded,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
        }
        
    def encode(self, x):
        # x: (B, L, C_feat)
        encoded = self.encoder(x)
        x_ = encoded.permute(0, 2, 1).contiguous()
        code_idx, all_codes = self.quantizer.quantize(x_, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes
    
    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        x = x_d.sum(dim=0)
        decoded = self.decoder(x)
        return decoded




from models.timm_transformer.transformer import Block,Mlp

class CausalConvAttentionVQVAE(nn.Module):
    def __init__(self,args, feat_dim=315, latent_dim=512,
                 num_downsampling_stages=2, stride_t=2, # Determines seq_len_factor
                 width=512, depth_res_blocks=3,
                 kernel_size_initial_enc=3, kernel_size_initial_dec=3,
                 kernel_size_res=3, dilation_growth_rate=2,
                 activation='relu', norm='LN', dropout=0.2, nb_code=512, code_dim=512,
                 encoder_transformer_depth=0,
                 decoder_transformer_depth=3,
                 lookback=3,
                 ):
        super().__init__()
        
        self.encoder = Encoder_Causal(
            input_emb_width=feat_dim,
            output_emb_width=latent_dim,
            down_t=num_downsampling_stages,
            stride_t=stride_t,
            width=width,
            depth=depth_res_blocks,
            kernel_size_initial=kernel_size_initial_enc,
            kernel_size_res=kernel_size_res,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
            dropout=dropout
        )
        
        self.decoder = Decoder_Causal(
            input_emb_width=feat_dim, # Final output channels
            latent_emb_width=latent_dim, # Input from encoder
            up_t=num_downsampling_stages, # Should match encoder's down_t
            stride_t=stride_t,          # Should match encoder's stride_t
            width=width,
            depth=depth_res_blocks,
            kernel_size_initial=kernel_size_initial_dec,
            kernel_size_res=kernel_size_res,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
            dropout=dropout
        )
        self.decoder_transformer_depth = decoder_transformer_depth
        self.decoder_transformer = nn.ModuleList([
            Block(dim=code_dim, num_heads=8, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, drop_path=0,is_causal=True)
                for _ in range(decoder_transformer_depth)])
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)
        self.lookback = lookback
    def forward(self, x):
        # x: (B, L, C_feat)
        encoded = self.encoder(x)
        x_ = encoded.permute(0, 2, 1).contiguous()
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_, sample_codebook_temp=0.5)
        x_quantized = x_quantized.permute(0, 2, 1).contiguous()
        
        if self.decoder_transformer_depth>0:
            seq_len = x_quantized.shape[1]
            lookback = self.lookback
            mask = torch.zeros((seq_len, seq_len), dtype=int,device=x_quantized.device)
            # 填充mask
            for i in range(seq_len):
                for j in range(max(0, i - lookback), i + 1):
                    mask[i, j] = 1
            attn_mask = mask.bool()
            for decoder in self.decoder_transformer:
                x_quantized = decoder(x_quantized,attn_mask)
        
        decoded = self.decoder(x_quantized)

        return  {
            'rec_pose': decoded,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
        }
        
    def encode(self, x):
        # x: (B, L, C_feat)
        encoded = self.encoder(x)
        x_ = encoded.permute(0, 2, 1).contiguous()
        code_idx, all_codes = self.quantizer.quantize(x_, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes
    
    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        x = x_d.sum(dim=0)
        decoded = self.decoder(x)
        return decoded


import numpy as np

class CausalAttention_RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 lookback=15,
                 ):

        super().__init__()
        
        # 暂时增大depth
        # depth = 12
        
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        # self.quant = args.quantizer
        
        self.input_liner = nn.Linear(input_width, output_emb_width)
        self.output_liner = nn.Linear(output_emb_width, input_width)
        self.encoder = nn.ModuleList([
            Block(dim=output_emb_width, num_heads=8, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, drop_path=0,is_causal=True)
                for _ in range(depth)])
        self.decoder = nn.ModuleList([
            Block(dim=output_emb_width, num_heads=8, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, drop_path=0,is_causal=True)
                for _ in range(depth)])
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)
        
        self.lookback = lookback
        seq_len = 64
        mask = np.zeros((seq_len, seq_len), dtype=int)

        # 填充mask
        for i in range(seq_len):
            for j in range(max(0, i - lookback), i + 1):
                mask[i, j] = 1
        self.attn_mask = torch.tensor(mask).bool().cuda()
            

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        x = self.input_liner(x)
        #x_in = self.preprocess(x)
        # Encode
        bs,seq,feat = x.shape
        
        seq_len = seq
        lookback = self.lookback
        mask = np.zeros((seq_len, seq_len), dtype=int)

        # 填充mask
        for i in range(seq_len):
            for j in range(max(0, i - lookback), i + 1):
                mask[i, j] = 1
        attn_mask = torch.tensor(mask).bool().to(x.device)
        
        for encoder in self.encoder:
            x = encoder(x,attn_mask)
        # print(x_encoder.shape)
        x_encoder = x.permute(0,2,1).contiguous()
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes

    def forward(self, x):
        x = self.input_liner(x)
        #x_in = self.preprocess(x)
        # Encode
        
        for encoder in self.encoder:
            attn_mask = None
            attn_mask = self.attn_mask
            x = encoder(x,attn_mask)

        ## quantization
        # x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5,
        #                                                                 force_dropout_index=0) #TODO hardcode
        
        x_encoder = x.permute(0,2,1).contiguous()
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
        x_quantized = x_quantized.permute(0,2,1).contiguous()
        
        # print(code_idx[0, :, 1])
        ## decoder
        
        for decoder in self.decoder:
            attn_mask = None
            # 下面这行得得注释，在明天测试无mask的下情况
            # attn_mask = torch.eye(x.shape[1]).to(x.device)
            attn_mask = self.attn_mask
            x_quantized = decoder(x_quantized,attn_mask)
        
        x_out = self.output_liner(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return  {
            'rec_pose': x_out,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
        }

    def forward_once(self, x):
        x = self.input_liner(x)
        #x_in = self.preprocess(x)
        # Encode
        bs,seq,feat = x.shape
        
        seq_len = seq
        lookback = self.lookback
        mask = np.zeros((seq_len, seq_len), dtype=int)

        # 填充mask
        for i in range(seq_len):
            for j in range(max(0, i - lookback), i + 1):
                mask[i, j] = 1
        attn_mask = torch.tensor(mask).bool().to(x.device)
        
        for encoder in self.encoder:
            x = encoder(x,attn_mask)

        ## quantization
        # x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5,
        #                                                                 force_dropout_index=0) #TODO hardcode
        
        x_encoder = x.permute(0,2,1).contiguous()
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
        x_quantized = x_quantized.permute(0,2,1).contiguous()
        
        # print(code_idx[0, :, 1])
        ## decoder
        
        for decoder in self.decoder:
            x_quantized = decoder(x_quantized,attn_mask)
        
        x_out = self.output_liner(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return  {
            'rec_pose': x_out,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
        }
        
    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0)
        
        bs,seq,feat = x.shape
        seq_len = seq
        lookback = self.lookback
        mask = np.zeros((seq_len, seq_len), dtype=int)

        # 填充mask
        for i in range(seq_len):
            for j in range(max(0, i - lookback), i + 1):
                mask[i, j] = 1
        attn_mask = torch.tensor(mask).bool().to(x.device)
        x_quantized = x
        for decoder in self.decoder:
            x_quantized = decoder(x_quantized,attn_mask)
            
        x_out = self.output_liner(x_quantized)
        return x_out
    
    def map2latent(self,x):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        x_encoder = x_encoder.permute(0,2,1)
        return x_encoder

    def latent2origin(self,x):
        x = x.permute(0,2,1)
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x, sample_codebook_temp=0.5)
        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return x_out, commit_loss, perplexity



class CausalAttentionPooling4_RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 lookback=15,
                 ):

        super().__init__()
        
        # 暂时增大depth
        # depth = 12
        
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        # self.quant = args.quantizer
        
        self.input_liner = nn.Linear(input_width, output_emb_width)
        self.output_liner = nn.Linear(output_emb_width, input_width)
        self.encoder = nn.ModuleList([
            Block(dim=output_emb_width, num_heads=8, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, drop_path=0,is_causal=True)
                for _ in range(depth)])
        self.decoder = nn.ModuleList([
            Block(dim=output_emb_width, num_heads=8, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, drop_path=0,is_causal=True)
                for _ in range(depth)])
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)
        
        self.lookback = lookback
        seq_len = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attn_mask1 = self.make_attn_mask(seq_len, lookback)
        self.attn_mask2 = self.make_attn_mask(seq_len//2, (lookback+1)//2-1)
        self.attn_mask3 = self.make_attn_mask(seq_len//4, (lookback+1)//4-1)
        if depth==3:
            self.attn_mask = [self.attn_mask1,self.attn_mask2,self.attn_mask3]
        elif depth==4:
            self.attn_mask = [self.attn_mask1,self.attn_mask1,self.attn_mask2,self.attn_mask3]
        elif depth==6:
            self.attn_mask = [self.attn_mask1,self.attn_mask1,self.attn_mask2,self.attn_mask2,self.attn_mask3,self.attn_mask3]
        


    def make_attn_mask(self, seq_len, lookback):
        mask = np.zeros((seq_len, seq_len), dtype=int)
        # 填充mask
        for i in range(seq_len):
            for j in range(max(0, i - lookback), i + 1):
                mask[i, j] = 1
        attn_mask = torch.tensor(mask,device=self.device).bool()
        return attn_mask


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        x = self.input_liner(x)
        #x_in = self.preprocess(x)
        # Encode
        bs,seq,feat = x.shape
        
        seq_len = seq
        lookback = self.lookback
        mask = np.zeros((seq_len, seq_len), dtype=int)

        # 填充mask
        for i in range(seq_len):
            for j in range(max(0, i - lookback), i + 1):
                mask[i, j] = 1
        attn_mask = torch.tensor(mask).bool().to(x.device)
        
        for encoder in self.encoder:
            x = encoder(x,attn_mask)
        # print(x_encoder.shape)
        x_encoder = x.permute(0,2,1).contiguous()
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes

    def forward(self, x):
        x = self.input_liner(x)
        #x_in = self.preprocess(x)
        # Encode
        

        for i, encoder in enumerate(self.encoder):
            attn_mask = self.attn_mask[i]
            x = encoder(x,attn_mask)
            if i<=1:
                x = (x[:, ::2, :]+x[:, 1::2, :])/2

        x_encoder = x.permute(0,2,1).contiguous()
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
        x_quantized = x_quantized.permute(0,2,1).contiguous()
        

        for i, decoder in enumerate(self.decoder):
            attn_mask = self.attn_mask[len(self.attn_mask)-i-1]
            x_quantized = decoder(x_quantized,attn_mask)
            if i<=1:
                x_quantized = x_quantized.repeat_interleave(2, dim=1)

        x_out = self.output_liner(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return  {
            'rec_pose': x_out,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
        }

    def forward_once(self, x):
        x = self.input_liner(x)
        #x_in = self.preprocess(x)
        # Encode
        bs,seq,feat = x.shape
        
        seq_len = seq
        lookback = self.lookback
        attn_mask1 = self.make_attn_mask(seq_len, lookback)
        attn_mask2 = self.make_attn_mask(seq_len//2, (lookback+1)//2-1)
        attn_mask3 = self.make_attn_mask(seq_len//4, (lookback+1)//4-1)
        if len(self.attn_mask)==3:
            attn_masks = [attn_mask1,attn_mask2,attn_mask3]
        elif len(self.attn_mask)==4:
            attn_masks = [attn_mask1,attn_mask1,attn_mask2,attn_mask3]
        elif len(self.attn_mask)==6:
            attn_masks = [attn_mask1,attn_mask1,attn_mask2,attn_mask2,attn_mask3,attn_mask3]

        for i, encoder in enumerate(self.encoder):
            attn_mask = attn_masks[i]
            x = encoder(x,attn_mask)
            if i<=1:
                x = (x[:, ::2, :]+x[:, 1::2, :])/2

        x_encoder = x.permute(0,2,1).contiguous()
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
        x_quantized = x_quantized.permute(0,2,1).contiguous()
        for i, decoder in enumerate(self.decoder):
            attn_mask = attn_masks[len(attn_masks)-i-1]
            x_quantized = decoder(x_quantized,attn_mask)
            if i<=1:
                x_quantized = x_quantized.repeat_interleave(2, dim=1)
        
        x_out = self.output_liner(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return  {
            'rec_pose': x_out,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
        }
        
    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0)
        
        bs,seq,feat = x.shape
        seq_len = seq
        lookback = self.lookback
        mask = np.zeros((seq_len, seq_len), dtype=int)

        # 填充mask
        for i in range(seq_len):
            for j in range(max(0, i - lookback), i + 1):
                mask[i, j] = 1
        attn_mask = torch.tensor(mask).bool().to(x.device)
        x_quantized = x
        for decoder in self.decoder:
            x_quantized = decoder(x_quantized,attn_mask)
            
        x_out = self.output_liner(x_quantized)
        return x_out
    
    def map2latent(self,x):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        x_encoder = x_encoder.permute(0,2,1)
        return x_encoder

    def latent2origin(self,x):
        x = x.permute(0,2,1)
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x, sample_codebook_temp=0.5)
        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return x_out, commit_loss, perplexity





class CausalAttentionMLP4_RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 lookback=15,
                 ):

        super().__init__()
        
        # 暂时增大depth
        # depth = 12
        
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        # self.quant = args.quantizer
        
        self.input_liner = nn.Linear(input_width, output_emb_width)
        self.output_liner = nn.Linear(output_emb_width, input_width)
        self.encoder = nn.ModuleList([
            Block(dim=output_emb_width, num_heads=8, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, drop_path=0,is_causal=True)
                for _ in range(depth)])
        self.decoder = nn.ModuleList([
            Block(dim=output_emb_width, num_heads=8, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, drop_path=0,is_causal=True)
                for _ in range(depth)])
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)
        
        self.lookback = lookback
        seq_len = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attn_mask1 = self.make_attn_mask(seq_len, lookback)
        self.attn_mask2 = self.make_attn_mask(seq_len//2, (lookback+1)//2-1)
        self.attn_mask3 = self.make_attn_mask(seq_len//4, (lookback+1)//4-1)
        if depth==3:
            self.attn_mask = [self.attn_mask1,self.attn_mask2,self.attn_mask3]
        elif depth==4:
            self.attn_mask = [self.attn_mask1,self.attn_mask1,self.attn_mask2,self.attn_mask3]
        elif depth==6:
            self.attn_mask = [self.attn_mask1,self.attn_mask1,self.attn_mask2,self.attn_mask2,self.attn_mask3,self.attn_mask3]
        
        self.down_sample = torch.nn.ModuleList([Mlp(in_features=output_emb_width*2, hidden_features=output_emb_width*4, out_features=output_emb_width, act_layer=nn.GELU, drop=0.) for _ in range(2)])
        self.up_sample = torch.nn.ModuleList([Mlp(in_features=output_emb_width, hidden_features=output_emb_width*4, out_features=output_emb_width*2, act_layer=nn.GELU, drop=0.) for _ in range(2)])


    def make_attn_mask(self, seq_len, lookback):
        mask = np.zeros((seq_len, seq_len), dtype=int)
        # 填充mask
        for i in range(seq_len):
            for j in range(max(0, i - lookback), i + 1):
                mask[i, j] = 1
        attn_mask = torch.tensor(mask,device=self.device).bool()
        return attn_mask


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        x = self.input_liner(x)
        #x_in = self.preprocess(x)
        # Encode
        bs,seq,feat = x.shape
        
        seq_len = seq
        lookback = self.lookback
        mask = np.zeros((seq_len, seq_len), dtype=int)

        # 填充mask
        for i in range(seq_len):
            for j in range(max(0, i - lookback), i + 1):
                mask[i, j] = 1
        attn_mask = torch.tensor(mask).bool().to(x.device)
        
        for encoder in self.encoder:
            x = encoder(x,attn_mask)
        # print(x_encoder.shape)
        x_encoder = x.permute(0,2,1).contiguous()
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes

    def forward(self, x):
        x = self.input_liner(x)
        #x_in = self.preprocess(x)
        # Encode
        

        for i, encoder in enumerate(self.encoder):
            attn_mask = self.attn_mask[i]
            x = encoder(x,attn_mask)
            if i<=1:
                x = self.down_sample[i](torch.concat([x[:, ::2, :], x[:, 1::2, :]], dim=-1))

        x_encoder = x.permute(0,2,1).contiguous()
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
        x_quantized = x_quantized.permute(0,2,1).contiguous()
        

        for i, decoder in enumerate(self.decoder):
            attn_mask = self.attn_mask[len(self.attn_mask)-i-1]
            x_quantized = decoder(x_quantized,attn_mask)
            if i<=1:
                x_quantized_combine = self.up_sample[i](x_quantized)
                x_quantized_combine1 = x_quantized_combine[...,:self.code_dim]
                x_quantized_combine2 = x_quantized_combine[...,self.code_dim:]
                x_quantized = torch.zeros((x_quantized.shape[0], x_quantized.shape[1]*2, x_quantized.shape[2]), device=x_quantized.device)
                x_quantized[:, ::2, :] = x_quantized_combine1
                x_quantized[:, 1::2, :] = x_quantized_combine2
                

        x_out = self.output_liner(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return  {
            'rec_pose': x_out,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
        }

    def forward_once(self, x):
        x = self.input_liner(x)
        #x_in = self.preprocess(x)
        # Encode
        bs,seq,feat = x.shape
        
        seq_len = seq
        lookback = self.lookback
        attn_mask1 = self.make_attn_mask(seq_len, lookback)
        attn_mask2 = self.make_attn_mask(seq_len//2, (lookback+1)//2-1)
        attn_mask3 = self.make_attn_mask(seq_len//4, (lookback+1)//4-1)
        if len(self.attn_mask)==3:
            attn_masks = [attn_mask1,attn_mask2,attn_mask3]
        elif len(self.attn_mask)==4:
            attn_masks = [attn_mask1,attn_mask1,attn_mask2,attn_mask3]
        elif len(self.attn_mask)==6:
            attn_masks = [attn_mask1,attn_mask1,attn_mask2,attn_mask2,attn_mask3,attn_mask3]

        for i, encoder in enumerate(self.encoder):
            attn_mask = attn_masks[i]
            x = encoder(x,attn_mask)
            if i<=1:
                x = self.down_sample[i](torch.concat([x[:, ::2, :], x[:, 1::2, :]], dim=-1))

        x_encoder = x.permute(0,2,1).contiguous()
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
        x_quantized = x_quantized.permute(0,2,1).contiguous()
        for i, decoder in enumerate(self.decoder):
            attn_mask = attn_masks[len(attn_masks)-i-1]
            x_quantized = decoder(x_quantized,attn_mask)
            if i<=1:
                x_quantized_combine = self.up_sample[i](x_quantized)
                x_quantized_combine1 = x_quantized_combine[...,:self.code_dim]
                x_quantized_combine2 = x_quantized_combine[...,self.code_dim:]
                x_quantized = torch.zeros((x_quantized.shape[0], x_quantized.shape[1]*2, x_quantized.shape[2]), device=x_quantized.device)
                x_quantized[:, ::2, :] = x_quantized_combine1
                x_quantized[:, 1::2, :] = x_quantized_combine2
        
        x_out = self.output_liner(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return  {
            'rec_pose': x_out,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
        }
        
    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0)
        
        bs,seq,feat = x.shape
        seq_len = seq
        lookback = self.lookback
        mask = np.zeros((seq_len, seq_len), dtype=int)

        # 填充mask
        for i in range(seq_len):
            for j in range(max(0, i - lookback), i + 1):
                mask[i, j] = 1
        attn_mask = torch.tensor(mask).bool().to(x.device)
        x_quantized = x
        for decoder in self.decoder:
            x_quantized = decoder(x_quantized,attn_mask)
            
        x_out = self.output_liner(x_quantized)
        return x_out
    
    def map2latent(self,x):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        x_encoder = x_encoder.permute(0,2,1)
        return x_encoder

    def latent2origin(self,x):
        x = x.permute(0,2,1)
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x, sample_codebook_temp=0.5)
        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return x_out, commit_loss, perplexity






class CausalAttentionPoolingMLP4_RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 lookback=15,
                 is_causal=True,
                 ):

        super().__init__()
        
        # 暂时增大depth
        # depth = 12
        
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        # self.quant = args.quantizer
        
        self.input_liner = nn.Linear(input_width, output_emb_width)
        self.output_liner = nn.Linear(output_emb_width, input_width)
        self.encoder = nn.ModuleList([
            Block(dim=output_emb_width, num_heads=8, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, drop_path=0,is_causal=is_causal)
                for _ in range(depth)])
        self.decoder = nn.ModuleList([
            Block(dim=output_emb_width, num_heads=8, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, drop_path=0,is_causal=is_causal)
                for _ in range(depth)])
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)
        
        self.lookback = lookback
        seq_len = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attn_mask1 = self.make_attn_mask(seq_len, lookback)
        self.attn_mask2 = self.make_attn_mask(seq_len//2, (lookback+1)//2-1)
        self.attn_mask3 = self.make_attn_mask(seq_len//4, (lookback+1)//4-1)
        if depth==3:
            self.attn_mask = [self.attn_mask1,self.attn_mask2,self.attn_mask3]
        elif depth==4:
            self.attn_mask = [self.attn_mask1,self.attn_mask1,self.attn_mask2,self.attn_mask3]
        elif depth==6:
            self.attn_mask = [self.attn_mask1,self.attn_mask1,self.attn_mask2,self.attn_mask2,self.attn_mask3,self.attn_mask3]
        
        self.down_sample = torch.nn.ModuleList([Mlp(in_features=output_emb_width*2, hidden_features=output_emb_width*4, out_features=output_emb_width, act_layer=nn.GELU, drop=0.) for _ in range(2)])
        self.up_sample = torch.nn.ModuleList([Mlp(in_features=output_emb_width, hidden_features=output_emb_width*4, out_features=output_emb_width*2, act_layer=nn.GELU, drop=0.) for _ in range(2)])


    def make_attn_mask(self, seq_len, lookback):
        mask = np.zeros((seq_len, seq_len), dtype=int)
        # 填充mask
        for i in range(seq_len):
            for j in range(max(0, i - lookback), i + 1):
                mask[i, j] = 1
        attn_mask = torch.tensor(mask,device=self.device).bool()
        return attn_mask


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        x = self.input_liner(x)
        #x_in = self.preprocess(x)
        # Encode
        bs,seq,feat = x.shape
        
        seq_len = seq
        lookback = self.lookback
        mask = np.zeros((seq_len, seq_len), dtype=int)

        # 填充mask
        for i in range(seq_len):
            for j in range(max(0, i - lookback), i + 1):
                mask[i, j] = 1
        attn_mask = torch.tensor(mask).bool().to(x.device)
        
        attn_mask1 = self.make_attn_mask(seq_len, lookback)
        attn_mask2 = self.make_attn_mask(seq_len//2, (lookback+1)//2-1)
        attn_mask3 = self.make_attn_mask(seq_len//4, (lookback+1)//4-1)
        attn_masks = [attn_mask1,attn_mask2,attn_mask3]

        for i, encoder in enumerate(self.encoder):
            attn_mask = attn_masks[i]
            x = encoder(x,attn_mask)
            if i<=1:
                x = (x[:, ::2, :]+x[:, 1::2, :])/2 + self.down_sample[i](torch.concat([x[:, ::2, :], x[:, 1::2, :]], dim=-1))
        # print(x_encoder.shape)
        x_encoder = x.permute(0,2,1).contiguous()
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes

    def forward(self, x):
        x = self.input_liner(x)
        #x_in = self.preprocess(x)
        # Encode
        

        for i, encoder in enumerate(self.encoder):
            attn_mask = self.attn_mask[i]
            x = encoder(x,attn_mask)
            if i<=1:
                x = (x[:, ::2, :]+x[:, 1::2, :])/2 + self.down_sample[i](torch.concat([x[:, ::2, :], x[:, 1::2, :]], dim=-1))

        x_encoder = x.permute(0,2,1).contiguous()
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
        x_quantized = x_quantized.permute(0,2,1).contiguous()
        

        for i, decoder in enumerate(self.decoder):
            attn_mask = self.attn_mask[len(self.attn_mask)-i-1]
            x_quantized = decoder(x_quantized,attn_mask)
            if i<=1:
                x_quantized_combine = self.up_sample[i](x_quantized)
                x_quantized_combine1 = x_quantized_combine[...,:self.code_dim]
                x_quantized_combine2 = x_quantized_combine[...,self.code_dim:]
                x_quantized = x_quantized.repeat_interleave(2, dim=1)
                x_quantized[:, ::2, :] = x_quantized[:, ::2, :] + x_quantized_combine1
                x_quantized[:, 1::2, :] = x_quantized[:, 1::2, :] + x_quantized_combine2
                

        x_out = self.output_liner(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return  {
            'rec_pose': x_out,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
        }

    def forward_once(self, x):
        x = self.input_liner(x)
        #x_in = self.preprocess(x)
        # Encode
        bs,seq,feat = x.shape
        
        seq_len = seq
        lookback = self.lookback
        attn_mask1 = self.make_attn_mask(seq_len, lookback)
        attn_mask2 = self.make_attn_mask(seq_len//2, (lookback+1)//2-1)
        attn_mask3 = self.make_attn_mask(seq_len//4, (lookback+1)//4-1)
        if len(self.attn_mask)==3:
            attn_masks = [attn_mask1,attn_mask2,attn_mask3]
        elif len(self.attn_mask)==4:
            attn_masks = [attn_mask1,attn_mask1,attn_mask2,attn_mask3]
        elif len(self.attn_mask)==6:
            attn_masks = [attn_mask1,attn_mask1,attn_mask2,attn_mask2,attn_mask3,attn_mask3]

        for i, encoder in enumerate(self.encoder):
            attn_mask = attn_masks[i]
            x = encoder(x,attn_mask)
            if i<=1:
                x = (x[:, ::2, :]+x[:, 1::2, :])/2 + self.down_sample[i](torch.concat([x[:, ::2, :], x[:, 1::2, :]], dim=-1))

        x_encoder = x.permute(0,2,1).contiguous()
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
        x_quantized = x_quantized.permute(0,2,1).contiguous()
        for i, decoder in enumerate(self.decoder):
            attn_mask = attn_masks[len(attn_masks)-i-1]
            x_quantized = decoder(x_quantized,attn_mask)
            if i<=1:
                x_quantized_combine = self.up_sample[i](x_quantized)
                x_quantized_combine1 = x_quantized_combine[...,:self.code_dim]
                x_quantized_combine2 = x_quantized_combine[...,self.code_dim:]
                x_quantized = x_quantized.repeat_interleave(2, dim=1)
                x_quantized[:, ::2, :] = x_quantized[:, ::2, :] + x_quantized_combine1
                x_quantized[:, 1::2, :] = x_quantized[:, 1::2, :] + x_quantized_combine2
        
        x_out = self.output_liner(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return  {
            'rec_pose': x_out,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
        }
        
    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0)
        
        bs,seq,feat = x.shape

        
        seq_len = seq*4
        lookback = self.lookback
        attn_mask1 = self.make_attn_mask(seq_len, lookback)
        attn_mask2 = self.make_attn_mask(seq_len//2, (lookback+1)//2-1)
        attn_mask3 = self.make_attn_mask(seq_len//4, (lookback+1)//4-1)
        if len(self.attn_mask)==3:
            attn_masks = [attn_mask1,attn_mask2,attn_mask3]
        elif len(self.attn_mask)==4:
            attn_masks = [attn_mask1,attn_mask1,attn_mask2,attn_mask3]
        elif len(self.attn_mask)==6:
            attn_masks = [attn_mask1,attn_mask1,attn_mask2,attn_mask2,attn_mask3,attn_mask3]
        

        x_quantized = x

        for i, decoder in enumerate(self.decoder):
            attn_mask = attn_masks[len(attn_masks)-i-1]
            x_quantized = decoder(x_quantized,attn_mask)
            if i<=1:
                x_quantized_combine = self.up_sample[i](x_quantized)
                x_quantized_combine1 = x_quantized_combine[...,:self.code_dim]
                x_quantized_combine2 = x_quantized_combine[...,self.code_dim:]
                x_quantized = x_quantized.repeat_interleave(2, dim=1)
                x_quantized[:, ::2, :] = x_quantized[:, ::2, :] + x_quantized_combine1
                x_quantized[:, 1::2, :] = x_quantized[:, 1::2, :] + x_quantized_combine2

        x_out = self.output_liner(x_quantized)
        return x_out
    
    def map2latent(self,x):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        x_encoder = x_encoder.permute(0,2,1)
        return x_encoder

    def latent2origin(self,x):
        x = x.permute(0,2,1)
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x, sample_codebook_temp=0.5)
        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return x_out, commit_loss, perplexity


class AttentionPoolingMLP4_AE(nn.Module):
    """
    非因果的 AE 版本，使用双向 attention mask（前后各看 lookback 个位置）
    移除了 VQ 量化部分
    """
    def __init__(self,
                 args,
                 input_width=263,
                 code_dim=512,
                 output_emb_width=512,
                 depth=3,
                 lookback=15,
                 ):

        super().__init__()
        
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        
        self.input_liner = nn.Linear(input_width, output_emb_width)
        self.output_liner = nn.Linear(output_emb_width, input_width)
        # 非因果 attention
        self.encoder = nn.ModuleList([
            Block(dim=output_emb_width, num_heads=8, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, drop_path=0, is_causal=False)
                for _ in range(depth)])
        self.decoder = nn.ModuleList([
            Block(dim=output_emb_width, num_heads=8, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, drop_path=0, is_causal=False)
                for _ in range(depth)])
        
        self.lookback = lookback
        seq_len = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 双向 attention mask
        self.attn_mask1 = self.make_bidirectional_attn_mask(seq_len, lookback)
        self.attn_mask2 = self.make_bidirectional_attn_mask(seq_len//2, (lookback+1)//2-1)
        self.attn_mask3 = self.make_bidirectional_attn_mask(seq_len//4, (lookback+1)//4-1)
        if depth==3:
            self.attn_mask = [self.attn_mask1, self.attn_mask2, self.attn_mask3]
        elif depth==4:
            self.attn_mask = [self.attn_mask1, self.attn_mask1, self.attn_mask2, self.attn_mask3]
        elif depth==6:
            self.attn_mask = [self.attn_mask1, self.attn_mask1, self.attn_mask2, self.attn_mask2, self.attn_mask3, self.attn_mask3]
        
        self.down_sample = torch.nn.ModuleList([Mlp(in_features=output_emb_width*2, hidden_features=output_emb_width*4, out_features=output_emb_width, act_layer=nn.GELU, drop=0.) for _ in range(2)])
        self.up_sample = torch.nn.ModuleList([Mlp(in_features=output_emb_width, hidden_features=output_emb_width*4, out_features=output_emb_width*2, act_layer=nn.GELU, drop=0.) for _ in range(2)])

    def make_bidirectional_attn_mask(self, seq_len, lookback):
        """
        创建双向 attention mask，每个位置可以看前后各 lookback 个位置
        """
        mask = np.zeros((seq_len, seq_len), dtype=int)
        # 填充mask：位置 i 可以看 [i-lookback, i+lookback] 范围内的位置
        for i in range(seq_len):
            start = max(0, i - lookback)
            end = min(seq_len, i + lookback + 1)
            for j in range(start, end):
                mask[i, j] = 1
        attn_mask = torch.tensor(mask, device=self.device).bool()
        return attn_mask

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        """编码器：输入 -> latent"""
        x = self.input_liner(x)
        bs, seq, feat = x.shape
        
        seq_len = seq
        lookback = self.lookback
        attn_mask1 = self.make_bidirectional_attn_mask(seq_len, lookback)
        attn_mask2 = self.make_bidirectional_attn_mask(seq_len//2, (lookback+1)//2-1)
        attn_mask3 = self.make_bidirectional_attn_mask(seq_len//4, (lookback+1)//4-1)
        attn_masks = [attn_mask1, attn_mask2, attn_mask3]

        for i, encoder in enumerate(self.encoder):
            attn_mask = attn_masks[i]
            x = encoder(x, attn_mask)
            if i <= 1:
                x = (x[:, ::2, :] + x[:, 1::2, :]) / 2 + self.down_sample[i](torch.concat([x[:, ::2, :], x[:, 1::2, :]], dim=-1))
        
        # x_encoder = x.permute(0, 2, 1).contiguous()
        return x

    def decode(self, x_latent):
        """解码器：latent -> 输出"""
        bs, seq, feat  = x_latent.shape
        
        seq_len = seq * 4  # 因为经过2次下采样，所以原始序列长度是4倍
        lookback = self.lookback
        attn_mask1 = self.make_bidirectional_attn_mask(seq_len, lookback)
        attn_mask2 = self.make_bidirectional_attn_mask(seq_len//2, (lookback+1)//2-1)
        attn_mask3 = self.make_bidirectional_attn_mask(seq_len//4, (lookback+1)//4-1)
        if len(self.attn_mask) == 3:
            attn_masks = [attn_mask1, attn_mask2, attn_mask3]
        elif len(self.attn_mask) == 4:
            attn_masks = [attn_mask1, attn_mask1, attn_mask2, attn_mask3]
        elif len(self.attn_mask) == 6:
            attn_masks = [attn_mask1, attn_mask1, attn_mask2, attn_mask2, attn_mask3, attn_mask3]
        
        # x_latent = x_latent.permute(0, 2, 1).contiguous()  # (bs, seq, feat)
        
        for i, decoder in enumerate(self.decoder):
            attn_mask = attn_masks[len(attn_masks) - i - 1]
            x_latent = decoder(x_latent, attn_mask)
            if i <= 1:
                x_combine = self.up_sample[i](x_latent)
                x_combine1 = x_combine[..., :self.code_dim]
                x_combine2 = x_combine[..., self.code_dim:]
                x_latent = x_latent.repeat_interleave(2, dim=1)
                x_latent[:, ::2, :] = x_latent[:, ::2, :] + x_combine1
                x_latent[:, 1::2, :] = x_latent[:, 1::2, :] + x_combine2

        x_out = self.output_liner(x_latent)
        return x_out

    def forward(self, x):
        x = self.input_liner(x)
        
        # Encode
        for i, encoder in enumerate(self.encoder):
            attn_mask = self.attn_mask[i]
            x = encoder(x, attn_mask)
            if i <= 1:
                x = (x[:, ::2, :] + x[:, 1::2, :]) / 2 + self.down_sample[i](torch.concat([x[:, ::2, :], x[:, 1::2, :]], dim=-1))

        x_latent = x  # 这里是 latent 表示
        
        # Decode
        for i, decoder in enumerate(self.decoder):
            attn_mask = self.attn_mask[len(self.attn_mask) - i - 1]
            x_latent = decoder(x_latent, attn_mask)
            if i <= 1:
                x_combine = self.up_sample[i](x_latent)
                x_combine1 = x_combine[..., :self.code_dim]
                x_combine2 = x_combine[..., self.code_dim:]
                x_latent = x_latent.repeat_interleave(2, dim=1)
                x_latent[:, ::2, :] = x_latent[:, ::2, :] + x_combine1
                x_latent[:, 1::2, :] = x_latent[:, 1::2, :] + x_combine2
                
        x_out = self.output_liner(x_latent)
        return {
            'rec_pose': x_out,
        }

    def forward_once(self, x):
        """支持任意长度序列的前向传播"""
        x = self.input_liner(x)
        bs, seq, feat = x.shape
        
        seq_len = seq
        lookback = self.lookback
        attn_mask1 = self.make_bidirectional_attn_mask(seq_len, lookback)
        attn_mask2 = self.make_bidirectional_attn_mask(seq_len//2, (lookback+1)//2-1)
        attn_mask3 = self.make_bidirectional_attn_mask(seq_len//4, (lookback+1)//4-1)
        if len(self.attn_mask) == 3:
            attn_masks = [attn_mask1, attn_mask2, attn_mask3]
        elif len(self.attn_mask) == 4:
            attn_masks = [attn_mask1, attn_mask1, attn_mask2, attn_mask3]
        elif len(self.attn_mask) == 6:
            attn_masks = [attn_mask1, attn_mask1, attn_mask2, attn_mask2, attn_mask3, attn_mask3]

        for i, encoder in enumerate(self.encoder):
            attn_mask = attn_masks[i]
            x = encoder(x, attn_mask)
            if i <= 1:
                x = (x[:, ::2, :] + x[:, 1::2, :]) / 2 + self.down_sample[i](torch.concat([x[:, ::2, :], x[:, 1::2, :]], dim=-1))

        x_latent = x
        
        for i, decoder in enumerate(self.decoder):
            attn_mask = attn_masks[len(attn_masks) - i - 1]
            x_latent = decoder(x_latent, attn_mask)
            if i <= 1:
                x_combine = self.up_sample[i](x_latent)
                x_combine1 = x_combine[..., :self.code_dim]
                x_combine2 = x_combine[..., self.code_dim:]
                x_latent = x_latent.repeat_interleave(2, dim=1)
                x_latent[:, ::2, :] = x_latent[:, ::2, :] + x_combine1
                x_latent[:, 1::2, :] = x_latent[:, 1::2, :] + x_combine2
        
        x_out = self.output_liner(x_latent)
        return {
            'rec_pose': x_out,
        }


class CausalAttentionPoolingMLP8_RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=4,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 lookback=15,
                 ):

        super().__init__()
        
        # 暂时增大depth
        # depth = 12
        
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        # self.quant = args.quantizer
        
        self.input_liner = nn.Linear(input_width, output_emb_width)
        self.output_liner = nn.Linear(output_emb_width, input_width)
        self.encoder = nn.ModuleList([
            Block(dim=output_emb_width, num_heads=8, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, drop_path=0,is_causal=True)
                for _ in range(depth)])
        self.decoder = nn.ModuleList([
            Block(dim=output_emb_width, num_heads=8, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, drop_path=0,is_causal=True)
                for _ in range(depth)])
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)
        
        self.lookback = lookback
        seq_len = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attn_mask1 = self.make_attn_mask(seq_len, lookback)
        self.attn_mask2 = self.make_attn_mask(seq_len//2, (lookback+1)//2-1)
        self.attn_mask3 = self.make_attn_mask(seq_len//4, (lookback+1)//4-1)
        self.attn_mask4 = self.make_attn_mask(seq_len//8, (lookback+1)//8-1)

        if depth==4:
            self.attn_mask = [self.attn_mask1,self.attn_mask2,self.attn_mask3,self.attn_mask4]

        self.down_sample = torch.nn.ModuleList([Mlp(in_features=output_emb_width*2, hidden_features=output_emb_width*4, out_features=output_emb_width, act_layer=nn.GELU, drop=0.) for _ in range(3)])
        self.up_sample = torch.nn.ModuleList([Mlp(in_features=output_emb_width, hidden_features=output_emb_width*4, out_features=output_emb_width*2, act_layer=nn.GELU, drop=0.) for _ in range(3)])


    def make_attn_mask(self, seq_len, lookback):
        mask = np.zeros((seq_len, seq_len), dtype=int)
        # 填充mask
        for i in range(seq_len):
            for j in range(max(0, i - lookback), i + 1):
                mask[i, j] = 1
        attn_mask = torch.tensor(mask,device=self.device).bool()
        return attn_mask


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        x = self.input_liner(x)
        #x_in = self.preprocess(x)
        # Encode
        bs,seq,feat = x.shape
        
        seq_len = seq
        lookback = self.lookback
        mask = np.zeros((seq_len, seq_len), dtype=int)

        # 填充mask
        for i in range(seq_len):
            for j in range(max(0, i - lookback), i + 1):
                mask[i, j] = 1
        attn_mask = torch.tensor(mask).bool().to(x.device)
        
        attn_mask1 = self.make_attn_mask(seq_len, lookback)
        attn_mask2 = self.make_attn_mask(seq_len//2, (lookback+1)//2-1)
        attn_mask3 = self.make_attn_mask(seq_len//4, (lookback+1)//4-1)
        attn_mask4 = self.make_attn_mask(seq_len//8, (lookback+1)//8-1)
        attn_masks = [attn_mask1,attn_mask2,attn_mask3,attn_mask4]

        for i, encoder in enumerate(self.encoder):
            attn_mask = attn_masks[i]
            x = encoder(x,attn_mask)
            if i<=2:
                x = (x[:, ::2, :]+x[:, 1::2, :])/2 + self.down_sample[i](torch.concat([x[:, ::2, :], x[:, 1::2, :]], dim=-1))
        # print(x_encoder.shape)
        x_encoder = x.permute(0,2,1).contiguous()
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes

    def forward(self, x):
        x = self.input_liner(x)
        #x_in = self.preprocess(x)
        # Encode
        

        for i, encoder in enumerate(self.encoder):
            attn_mask = self.attn_mask[i]
            x = encoder(x,attn_mask)
            if i<=2:
                x = (x[:, ::2, :]+x[:, 1::2, :])/2 + self.down_sample[i](torch.concat([x[:, ::2, :], x[:, 1::2, :]], dim=-1))

        x_encoder = x.permute(0,2,1).contiguous()
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
        x_quantized = x_quantized.permute(0,2,1).contiguous()
        

        for i, decoder in enumerate(self.decoder):
            attn_mask = self.attn_mask[len(self.attn_mask)-i-1]
            x_quantized = decoder(x_quantized,attn_mask)
            if i<=2:
                x_quantized_combine = self.up_sample[i](x_quantized)
                x_quantized_combine1 = x_quantized_combine[...,:self.code_dim]
                x_quantized_combine2 = x_quantized_combine[...,self.code_dim:]
                x_quantized = x_quantized.repeat_interleave(2, dim=1)
                x_quantized[:, ::2, :] = x_quantized[:, ::2, :] + x_quantized_combine1
                x_quantized[:, 1::2, :] = x_quantized[:, 1::2, :] + x_quantized_combine2
                

        x_out = self.output_liner(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return  {
            'rec_pose': x_out,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
        }

    def forward_once(self, x):
        x = self.input_liner(x)
        #x_in = self.preprocess(x)
        # Encode
        bs,seq,feat = x.shape
        
        seq_len = seq
        lookback = self.lookback
        attn_mask1 = self.make_attn_mask(seq_len, lookback)
        attn_mask2 = self.make_attn_mask(seq_len//2, (lookback+1)//2-1)
        attn_mask3 = self.make_attn_mask(seq_len//4, (lookback+1)//4-1)
        attn_mask4 = self.make_attn_mask(seq_len//8, (lookback+1)//8-1) 
        if len(self.attn_mask)==4:
            attn_masks = [attn_mask1,attn_mask2,attn_mask3,attn_mask4]


        for i, encoder in enumerate(self.encoder):
            attn_mask = attn_masks[i]
            x = encoder(x,attn_mask)
            if i<=2:
                x = (x[:, ::2, :]+x[:, 1::2, :])/2 + self.down_sample[i](torch.concat([x[:, ::2, :], x[:, 1::2, :]], dim=-1))

        x_encoder = x.permute(0,2,1).contiguous()
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
        x_quantized = x_quantized.permute(0,2,1).contiguous()
        for i, decoder in enumerate(self.decoder):
            attn_mask = attn_masks[len(attn_masks)-i-1]
            x_quantized = decoder(x_quantized,attn_mask)
            if i<=2:
                x_quantized_combine = self.up_sample[i](x_quantized)
                x_quantized_combine1 = x_quantized_combine[...,:self.code_dim]
                x_quantized_combine2 = x_quantized_combine[...,self.code_dim:]
                x_quantized = x_quantized.repeat_interleave(2, dim=1)
                x_quantized[:, ::2, :] = x_quantized[:, ::2, :] + x_quantized_combine1
                x_quantized[:, 1::2, :] = x_quantized[:, 1::2, :] + x_quantized_combine2
        
        x_out = self.output_liner(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return  {
            'rec_pose': x_out,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
        }
        
    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0)
        
        bs,seq,feat = x.shape

        
        seq_len = seq*8
        lookback = self.lookback
        attn_mask1 = self.make_attn_mask(seq_len, lookback)
        attn_mask2 = self.make_attn_mask(seq_len//2, (lookback+1)//2-1)
        attn_mask3 = self.make_attn_mask(seq_len//4, (lookback+1)//4-1)
        attn_mask4 = self.make_attn_mask(seq_len//8, (lookback+1)//8-1)
        if len(self.attn_mask)==4:
            attn_masks = [attn_mask1,attn_mask2,attn_mask3,attn_mask4]
        

        x_quantized = x

        for i, decoder in enumerate(self.decoder):
            attn_mask = attn_masks[len(attn_masks)-i-1]
            x_quantized = decoder(x_quantized,attn_mask)
            if i<=2:
                x_quantized_combine = self.up_sample[i](x_quantized)
                x_quantized_combine1 = x_quantized_combine[...,:self.code_dim]
                x_quantized_combine2 = x_quantized_combine[...,self.code_dim:]
                x_quantized = x_quantized.repeat_interleave(2, dim=1)
                x_quantized[:, ::2, :] = x_quantized[:, ::2, :] + x_quantized_combine1
                x_quantized[:, 1::2, :] = x_quantized[:, 1::2, :] + x_quantized_combine2

        x_out = self.output_liner(x_quantized)
        return x_out
    
    def map2latent(self,x):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        x_encoder = x_encoder.permute(0,2,1)
        return x_encoder

    def latent2origin(self,x):
        x = x.permute(0,2,1)
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x, sample_codebook_temp=0.5)
        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return x_out, commit_loss, perplexity


class AttentionPoolingMLP8_AE(nn.Module):
    """
    非因果的 AE 版本，使用双向 attention mask（前后各看 lookback 个位置）
    8倍下采样，移除了 VQ 量化部分
    """
    def __init__(self,
                 args,
                 input_width=263,
                 code_dim=512,
                 output_emb_width=512,
                 depth=4,
                 lookback=15,
                 ):

        super().__init__()
        
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        
        self.input_liner = nn.Linear(input_width, output_emb_width)
        self.output_liner = nn.Linear(output_emb_width, input_width)
        # 非因果 attention
        self.encoder = nn.ModuleList([
            Block(dim=output_emb_width, num_heads=8, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, drop_path=0, is_causal=False)
                for _ in range(depth)])
        self.decoder = nn.ModuleList([
            Block(dim=output_emb_width, num_heads=8, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, drop_path=0, is_causal=False)
                for _ in range(depth)])
        
        self.lookback = lookback
        seq_len = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 双向 attention mask
        self.attn_mask1 = self.make_bidirectional_attn_mask(seq_len, lookback)
        self.attn_mask2 = self.make_bidirectional_attn_mask(seq_len//2, (lookback+1)//2-1)
        self.attn_mask3 = self.make_bidirectional_attn_mask(seq_len//4, (lookback+1)//4-1)
        self.attn_mask4 = self.make_bidirectional_attn_mask(seq_len//8, (lookback+1)//8-1)

        if depth == 4:
            self.attn_mask = [self.attn_mask1, self.attn_mask2, self.attn_mask3, self.attn_mask4]
        
        self.down_sample = torch.nn.ModuleList([Mlp(in_features=output_emb_width*2, hidden_features=output_emb_width*4, out_features=output_emb_width, act_layer=nn.GELU, drop=0.) for _ in range(3)])
        self.up_sample = torch.nn.ModuleList([Mlp(in_features=output_emb_width, hidden_features=output_emb_width*4, out_features=output_emb_width*2, act_layer=nn.GELU, drop=0.) for _ in range(3)])

    def make_bidirectional_attn_mask(self, seq_len, lookback):
        """
        创建双向 attention mask，每个位置可以看前后各 lookback 个位置
        """
        mask = np.zeros((seq_len, seq_len), dtype=int)
        # 填充mask：位置 i 可以看 [i-lookback, i+lookback] 范围内的位置
        for i in range(seq_len):
            start = max(0, i - lookback)
            end = min(seq_len, i + lookback + 1)
            for j in range(start, end):
                mask[i, j] = 1
        attn_mask = torch.tensor(mask, device=self.device).bool()
        return attn_mask

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        """编码器：输入 -> latent"""
        x = self.input_liner(x)
        bs, seq, feat = x.shape
        
        seq_len = seq
        lookback = self.lookback
        attn_mask1 = self.make_bidirectional_attn_mask(seq_len, lookback)
        attn_mask2 = self.make_bidirectional_attn_mask(seq_len//2, (lookback+1)//2-1)
        attn_mask3 = self.make_bidirectional_attn_mask(seq_len//4, (lookback+1)//4-1)
        attn_mask4 = self.make_bidirectional_attn_mask(seq_len//8, (lookback+1)//8-1)
        attn_masks = [attn_mask1, attn_mask2, attn_mask3, attn_mask4]

        for i, encoder in enumerate(self.encoder):
            attn_mask = attn_masks[i]
            x = encoder(x, attn_mask)
            if i <= 2:
                x = (x[:, ::2, :] + x[:, 1::2, :]) / 2 + self.down_sample[i](torch.concat([x[:, ::2, :], x[:, 1::2, :]], dim=-1))
        
        return x

    def decode(self, x_latent):
        """解码器：latent -> 输出"""
        bs, seq, feat = x_latent.shape
        
        seq_len = seq * 8  # 因为经过3次下采样，所以原始序列长度是8倍
        lookback = self.lookback
        attn_mask1 = self.make_bidirectional_attn_mask(seq_len, lookback)
        attn_mask2 = self.make_bidirectional_attn_mask(seq_len//2, (lookback+1)//2-1)
        attn_mask3 = self.make_bidirectional_attn_mask(seq_len//4, (lookback+1)//4-1)
        attn_mask4 = self.make_bidirectional_attn_mask(seq_len//8, (lookback+1)//8-1)
        if len(self.attn_mask) == 4:
            attn_masks = [attn_mask1, attn_mask2, attn_mask3, attn_mask4]
        
        for i, decoder in enumerate(self.decoder):
            attn_mask = attn_masks[len(attn_masks) - i - 1]
            x_latent = decoder(x_latent, attn_mask)
            if i <= 2:
                x_combine = self.up_sample[i](x_latent)
                x_combine1 = x_combine[..., :self.code_dim]
                x_combine2 = x_combine[..., self.code_dim:]
                x_latent = x_latent.repeat_interleave(2, dim=1)
                x_latent[:, ::2, :] = x_latent[:, ::2, :] + x_combine1
                x_latent[:, 1::2, :] = x_latent[:, 1::2, :] + x_combine2

        x_out = self.output_liner(x_latent)
        return x_out

    def forward(self, x):
        x = self.input_liner(x)
        
        # Encode
        for i, encoder in enumerate(self.encoder):
            attn_mask = self.attn_mask[i]
            x = encoder(x, attn_mask)
            if i <= 2:
                x = (x[:, ::2, :] + x[:, 1::2, :]) / 2 + self.down_sample[i](torch.concat([x[:, ::2, :], x[:, 1::2, :]], dim=-1))

        x_latent = x  # 这里是 latent 表示
        
        # Decode
        for i, decoder in enumerate(self.decoder):
            attn_mask = self.attn_mask[len(self.attn_mask) - i - 1]
            x_latent = decoder(x_latent, attn_mask)
            if i <= 2:
                x_combine = self.up_sample[i](x_latent)
                x_combine1 = x_combine[..., :self.code_dim]
                x_combine2 = x_combine[..., self.code_dim:]
                x_latent = x_latent.repeat_interleave(2, dim=1)
                x_latent[:, ::2, :] = x_latent[:, ::2, :] + x_combine1
                x_latent[:, 1::2, :] = x_latent[:, 1::2, :] + x_combine2
                
        x_out = self.output_liner(x_latent)
        return {
            'rec_pose': x_out,
        }

    def forward_once(self, x):
        """支持任意长度序列的前向传播"""
        x = self.input_liner(x)
        bs, seq, feat = x.shape
        
        seq_len = seq
        lookback = self.lookback
        attn_mask1 = self.make_bidirectional_attn_mask(seq_len, lookback)
        attn_mask2 = self.make_bidirectional_attn_mask(seq_len//2, (lookback+1)//2-1)
        attn_mask3 = self.make_bidirectional_attn_mask(seq_len//4, (lookback+1)//4-1)
        attn_mask4 = self.make_bidirectional_attn_mask(seq_len//8, (lookback+1)//8-1)
        if len(self.attn_mask) == 4:
            attn_masks = [attn_mask1, attn_mask2, attn_mask3, attn_mask4]

        for i, encoder in enumerate(self.encoder):
            attn_mask = attn_masks[i]
            x = encoder(x, attn_mask)
            if i <= 2:
                x = (x[:, ::2, :] + x[:, 1::2, :]) / 2 + self.down_sample[i](torch.concat([x[:, ::2, :], x[:, 1::2, :]], dim=-1))

        x_latent = x
        
        for i, decoder in enumerate(self.decoder):
            attn_mask = attn_masks[len(attn_masks) - i - 1]
            x_latent = decoder(x_latent, attn_mask)
            if i <= 2:
                x_combine = self.up_sample[i](x_latent)
                x_combine1 = x_combine[..., :self.code_dim]
                x_combine2 = x_combine[..., self.code_dim:]
                x_latent = x_latent.repeat_interleave(2, dim=1)
                x_latent[:, ::2, :] = x_latent[:, ::2, :] + x_combine1
                x_latent[:, 1::2, :] = x_latent[:, 1::2, :] + x_combine2
        
        x_out = self.output_liner(x_latent)
        return {
            'rec_pose': x_out,
        }









import argparse
def get_causal_conv_vqvae_model(ckpt_path = '/mnt/data/cbh/SynTalker/output_beatx2_causal_attention/RVQVAE_whole_trans/net_20000.pth',
                                            dim_pose = 315,
                                            vq_act = 'relu',
                                            depth_res_blocks = 3,
                                            num_downsampling_stages = 2,
                                            dropout = 0.2,
                                            ):
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args, _ = parser.parse_known_args()
    
    args.num_quantizers = 1
    args.shared_codebook =  False
    args.quantize_dropout_prob = 0.2
    args.mu = 0.99

    args.nb_code = 512
    args.code_dim = 128
    args.down_t = 2
    args.stride_t = 2
    args.width = 512
    args.depth = 3
    args.dilation_growth_rate = 3
    args.vq_act = vq_act
    args.vq_norm = None

    model = CausalConvVQVAE(args,
                dim_pose,
                activation = vq_act,
                depth_res_blocks=depth_res_blocks,
                num_downsampling_stages = num_downsampling_stages,
                dropout = dropout,
    )
    re = model.load_state_dict(torch.load(ckpt_path)['net'],strict=False)
    print(f"load model from {ckpt_path}, load result: {re}")
    model.cuda().eval()
    return model



def get_causal_attn_PoolingMLP4_rvqvae_model(ckpt_path = '/mnt/data/cbh/rta_conv_motorica_xx_encodec/output_30fps_attn_pool_wandb/RVQVAE_PoolingMLP_whole_trans_commit-0.5_loss-pos-l1-0.02_loss-pos-vel-l1-0.2_loss-pos-acc-l1-0.2_loss-trans-vel-l1_smooth-1_depth-3_loss-foot-contact-label-l1-0.3_loss-foot-pos-l1-0.05_num_downsampling_stages-2_dropout-0/net_20000.pth',
                                            dim_pose = 531,
                                            rvq_layer_num=1,
                                            nb_code=1024,
                                            lookback=15,
                                            ):
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args, _ = parser.parse_known_args()
    
    args.num_quantizers = rvq_layer_num
    args.shared_codebook =  False
    args.quantize_dropout_prob = 0.2
    args.mu = 0.99

    args.nb_code = nb_code
    args.code_dim = 512
    args.down_t = 2
    args.stride_t = 2
    args.width = 512
    args.depth = 3
    args.vq_norm = None

    model = CausalAttentionPoolingMLP4_RVQVAE(
                args,
                dim_pose,
                nb_code=nb_code,
                lookback=lookback,
    )
    re = model.load_state_dict(torch.load(ckpt_path)['net'],strict=False)
    print(f"load model from {ckpt_path}, load result: {re}")
    model.cuda().eval()
    return model




def get_causal_attn_Pooling4_rvqvae_model(ckpt_path = '/mnt/data/cbh/rta_conv_motorica_xx_encodec/output_30fps_attn_pool_wandb/RVQVAE_PoolingMLP_whole_trans_commit-0.5_loss-pos-l1-0.02_loss-pos-vel-l1-0.2_loss-pos-acc-l1-0.2_loss-trans-vel-l1_smooth-1_depth-3_loss-foot-contact-label-l1-0.3_loss-foot-pos-l1-0.05_num_downsampling_stages-2_dropout-0/net_20000.pth',
                                            dim_pose = 531,
                                            rvq_layer_num=1,
                                            nb_code=1024,
                                            
                                            ):
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args, _ = parser.parse_known_args()
    
    args.num_quantizers = rvq_layer_num
    args.shared_codebook =  False
    args.quantize_dropout_prob = 0.2
    args.mu = 0.99

    args.nb_code = nb_code
    args.code_dim = 512
    args.down_t = 2
    args.stride_t = 2
    args.width = 512
    args.depth = 3
    args.vq_norm = None

    model = CausalAttentionPooling4_RVQVAE(
                args,
                dim_pose,
                nb_code=nb_code,
    )
    re = model.load_state_dict(torch.load(ckpt_path)['net'],strict=False)
    print(f"load model from {ckpt_path}, load result: {re}")
    model.cuda().eval()
    return model



def get_causal_attn_PoolingMLP4_rvqvae_model_cpu(ckpt_path = '/mnt/data/cbh/rta_conv_motorica_xx_encodec/output_30fps_attn_pool_wandb/RVQVAE_PoolingMLP_whole_trans_commit-0.5_loss-pos-l1-0.02_loss-pos-vel-l1-0.2_loss-pos-acc-l1-0.2_loss-trans-vel-l1_smooth-1_depth-3_loss-foot-contact-label-l1-0.3_loss-foot-pos-l1-0.05_num_downsampling_stages-2_dropout-0/net_20000.pth',
                                            dim_pose = 531,
                                            rvq_layer_num=1,
                                            nb_code=1024,
                                            
                                            ):
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args, _ = parser.parse_known_args()
    
    args.num_quantizers = rvq_layer_num
    args.shared_codebook =  False
    args.quantize_dropout_prob = 0.2
    args.mu = 0.99

    args.nb_code = nb_code
    args.code_dim = 512
    args.down_t = 2
    args.stride_t = 2
    args.width = 512
    args.depth = 3
    args.vq_norm = None

    model = CausalAttentionPoolingMLP4_RVQVAE(
                args,
                dim_pose,
                nb_code=nb_code,
    )
    re = model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['net'], strict=False)
    print(f"load model from {ckpt_path}, load result: {re}")
    return model


def get_attn_PoolingMLP4_ae_model(ckpt_path = "/mnt/data/cbh/rta_attn_zm_xx_encodec_meco_v7/output_bp_30fps_attn_pool_wandb/AE_PoolingMLP_bp_whole_nb-code_512_commit-0.5_loss-pos-l1-0.02_loss-pos-vel-l1-0.2_loss-pos-acc-l1-0.2_loss-trans-vel-l1_smooth-10_depth-3_loss-foot-contact-label-l1-0.3_loss-foot-pos-l1-0.05_dropout-0_num_quantizers-6_lookback-15/net_300000.pth",
                                            dim_pose = 531,
                                            lookback = 15,
                                            depth= 3,
                                            ):
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args, _ = parser.parse_known_args()
    
    model = AttentionPoolingMLP4_AE(
                args,
                input_width = dim_pose,
                lookback = lookback,
                depth= depth,
    )
    re = model.load_state_dict(torch.load(ckpt_path)['net'],strict=False)
    print(f"load model from {ckpt_path}, load result: {re}")
    model.cuda().eval()
    return model



def get_causal_attn_PoolingMLP8_rvqvae_model(ckpt_path = '/mnt/data/cbh/rta_attn_zm_xx_encodec/output_60fps_attn_pool_wandb/RVQVAE_PoolingMLP_whole_trans_commit-0.5_loss-pos-l1-0.02_loss-pos-vel-l1-0.3_loss-pos-acc-l1-1_loss-trans-vel-l1_smooth-5_depth-4_loss-foot-contact-label-l1-1_loss-foot-pos-l1-0.05_num_downsampling_stages-2_dropout-0_num_quantizers-1_lookback-15/net_22000.pth',
                                            dim_pose = 531,
                                            rvq_layer_num=1,
                                            nb_code=512,
                                            lookback=15,
                                            ):
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args, _ = parser.parse_known_args()
    
    args.num_quantizers = rvq_layer_num
    args.shared_codebook =  False
    args.quantize_dropout_prob = 0.2
    args.mu = 0.99

    args.nb_code = nb_code
    args.code_dim = 128
    args.down_t = 2
    args.stride_t = 2
    args.width = 512
    args.depth = 4
    args.vq_norm = None

    model = CausalAttentionPoolingMLP8_RVQVAE(
                args,
                dim_pose,
                nb_code=nb_code,
                lookback = lookback,
    )
    re = model.load_state_dict(torch.load(ckpt_path)['net'],strict=False)
    print(f"load model from {ckpt_path}, load result: {re}")
    model.cuda().eval()
    return model