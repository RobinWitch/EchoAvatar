## Difference from 12.1: 12.1 used Mimi as the audio encoder; this script uses only EnCodec2, using the first two EnCodec RVQ layers as codes.
import os
## Do not force CUDA_VISIBLE_DEVICES inside the script, so body/face can run in parallel on multiple GPUs (set it via command-line environment variables if needed).
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys
sys.path.append('./')
import socket
import pickle
import soundfile as sf
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, AutoConfig
import re
from models.vq.casual_vqvae import get_causal_attn_PoolingMLP4_rvqvae_model
from utils import rotation_conversions as rc
import argparse
import time
import contextlib
from safetensors.torch import load_file
from vllm import LLM,SamplingParams
from transformers import MimiModel, AutoFeatureExtractor, EncodecFeatureExtractor
import json
from utils.anim import quat
from scipy.spatial.transform import Rotation as R
import threading
import queue
import traceback
import librosa
from models.dit_flowmatching_mhubert_v3 import Audio2FaceGPT, pad_audio
from diffusers import FlowMatchEulerDiscreteScheduler
from refers.encodec.encodec.model import EncodecModel
import orjson
# ==================== Lightweight profiling helpers ====================
# Enable with: PROFILE_TIMING=1 (default on), disable with PROFILE_TIMING=0
# Optional: PROFILE_SYNC=1 to cuda-synchronize inside some timed blocks for more accurate GPU timings
PROFILE_TIMING = os.getenv("PROFILE_TIMING", "1") == "1"
PROFILE_SYNC = os.getenv("PROFILE_SYNC", "0") == "1"
MOTION_SERVER_HOST = '10.76.5.190' # change it to your server's IP
PROFILE_TIMING = 0
PROFILE_SYNC = 0
PORT = 12345
MOTION_SERVER_PORT = 12346


@contextlib.contextmanager
def _prof(name: str, extra: str = "", sync_cuda: bool = False):
    if not PROFILE_TIMING:
        yield
        return
    if sync_cuda and torch.cuda.is_available() and PROFILE_SYNC:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if sync_cuda and torch.cuda.is_available() and PROFILE_SYNC:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if extra:
            print(f"[prof] {name}: {dt:.4f}s | {extra}")
        else:
            print(f"[prof] {name}: {dt:.4f}s")

# Enable deterministic mode
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# ==================== Face streaming (Audio2FaceGPT) ====================
FACE_FPS = 30
FACE_AUDIO_RATE_RECV = 24000
FACE_AUDIO_RATE_MODEL = 16000
FACE_CONTEXT_FRAMES = 63  # Requires 63 context frames
FACE_BLENDSHAPE_DIM = 52  # Current face model output dimension (arkit52)


# ==================== CUDA Graph Optimization Classes ====================
# All CUDA Graph classes use dedicated CUDA streams and the correct device context to avoid conflicts in multi-GPU environments

class CUDAGraphAudioEncoder:
    """
    Apply CUDA Graph optimization to the Audio Encoder (HuBERT + audio_proj)
    Use a dedicated CUDA stream to avoid conflicts in multi-GPU environments
    """
    def __init__(self, model, audio_len, frame_num=63, device='cuda'):
        self.model = model
        self.audio_len = audio_len
        self.frame_num = frame_num
        self.device = device
        
        self.graph = None
        self.static_audio = None
        self.static_output = None
        
        # Create a dedicated CUDA stream
        self.stream = torch.cuda.Stream(device=device)
        
        self._create_graph()
    
    def _create_graph(self):
        """Create the CUDA Graph for the Audio Encoder"""
        print(f"  [CUDA Graph] Creating Audio Encoder CUDA Graph (audio_len={self.audio_len}, frame_num={self.frame_num})...")
        
        # Ensure operations run on the correct device
        with torch.cuda.device(self.device):
            # Create static inputs
            self.static_audio = torch.randn(1, self.audio_len, device=self.device)
            
            # Warm up (on the dedicated stream)
            with torch.cuda.stream(self.stream):
                for _ in range(3):
                    with torch.no_grad():
                        padded = pad_audio(self.static_audio)
                        hidden_states = self.model.audio_encoder(padded).last_hidden_state
                        hidden_states = hidden_states.transpose(1, 2)
                        hidden_states = F.interpolate(hidden_states, size=self.frame_num, align_corners=False, mode='linear')
                        hidden_states = hidden_states.transpose(1, 2)
                        _ = self.model.audio_proj(hidden_states)
            
            self.stream.synchronize()
            
            # Capture the CUDA Graph (on the dedicated stream)
            self.graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.stream(self.stream):
                with torch.cuda.graph(self.graph, stream=self.stream):
                    with torch.no_grad():
                        padded = pad_audio(self.static_audio)
                        hidden_states = self.model.audio_encoder(padded).last_hidden_state
                        hidden_states = hidden_states.transpose(1, 2)
                        hidden_states = F.interpolate(hidden_states, size=self.frame_num, align_corners=False, mode='linear')
                        hidden_states = hidden_states.transpose(1, 2)
                        self.static_output = self.model.audio_proj(hidden_states)
            
            print(f"  [CUDA Graph] Audio Encoder CUDA Graph created")
    
    def forward(self, audio):
        """Run audio encoding with CUDA Graph"""
        # Copy inputs into the static buffer
        self.static_audio.copy_(audio)
        
        # Replay the graph
        self.graph.replay()
        
        # Return a copy of the output
        return self.static_output.clone()


class CUDAGraphGPTBlocks:
    """
    Apply CUDA Graph optimization to GPT blocks
    Use a dedicated CUDA stream to avoid conflicts in multi-GPU environments
    """
    def __init__(self, model, pre_motion_len, gen_len, device='cuda'):
        self.model = model
        self.pre_motion_len = pre_motion_len
        self.gen_len = gen_len
        self.device = device
        self.hidden_size = model.hidden_size
        
        # Create a graph for each possible T
        self.graphs = {}
        self.static_inputs = {}
        self.static_outputs = {}
        
        # Create a dedicated CUDA stream
        self.stream = torch.cuda.Stream(device=device)
        
        self._create_graphs()
    
    def _create_graphs(self):
        """Create CUDA Graphs for each sequence length T"""
        print(f"  [CUDA Graph] Creating GPT Blocks CUDA Graphs (T: {self.pre_motion_len} ~ {self.pre_motion_len + self.gen_len - 1})...")
        
        # Ensure operations run on the correct device
        with torch.cuda.device(self.device):
            # Pre-create the maximum-size causal mask
            max_T = self.pre_motion_len + self.gen_len
            causal_mask_full = self.model.generate_causal_mask(max_T, self.device)
            cross_mask_full = self.model.generate_cross_causal_mask(max_T, self.device)
            
            for frame_idx in range(self.gen_len):
                T = self.pre_motion_len + frame_idx
                
                # Create static inputs
                bs = 2  # CFG
                static_x = torch.randn(bs, T, self.hidden_size, device=self.device)
                static_audio = torch.randn(bs, T, self.hidden_size, device=self.device)
                static_causal_mask = causal_mask_full[:T, :T].clone()
                static_cross_mask = cross_mask_full[:T, :T].clone()
                
                # Warm up (on the dedicated stream)
                with torch.cuda.stream(self.stream):
                    for _ in range(3):
                        with torch.no_grad():
                            x = static_x.clone()
                            for block in self.model.blocks:
                                x = block(x, static_audio, static_causal_mask, static_cross_mask)
                
                self.stream.synchronize()
                
                # Capture the CUDA Graph (on the dedicated stream)
                g = torch.cuda.CUDAGraph()
                
                with torch.cuda.stream(self.stream):
                    with torch.cuda.graph(g, stream=self.stream):
                        with torch.no_grad():
                            x = static_x
                            for block in self.model.blocks:
                                x = block(x, static_audio, static_causal_mask, static_cross_mask)
                            static_output = x
                
                self.graphs[T] = g
                self.static_inputs[T] = {
                    'x': static_x,
                    'audio': static_audio,
                    'causal_mask': static_causal_mask,
                    'cross_mask': static_cross_mask,
                }
                self.static_outputs[T] = static_output
            
            print(f"  [CUDA Graph] GPT Blocks CUDA Graphs created ({len(self.graphs)} graphs)")
    
    def forward(self, T, x, audio):
        """Run GPT blocks with the CUDA Graph for the corresponding T"""
        # Copy inputs into the static buffer
        self.static_inputs[T]['x'].copy_(x)
        self.static_inputs[T]['audio'].copy_(audio)
        # causal_mask and cross_mask are fixed, so they do not need to be copied each time
        
        # Replay the graph
        self.graphs[T].replay()
        
        # Return a copy of the output
        return self.static_outputs[T].clone()


class CUDAGraphDiffusionHead:
    """
    Apply CUDA Graph optimization to DiffusionHead
    Use a dedicated CUDA stream to avoid conflicts in multi-GPU environments
    """
    def __init__(self, diffusion_head, time_embed, face_dim=52, hidden_size=64, 
                 num_inference_steps=5, device='cuda'):
        self.diffusion_head = diffusion_head
        self.time_embed = time_embed
        self.face_dim = face_dim
        self.hidden_size = hidden_size
        self.num_inference_steps = num_inference_steps
        self.device = device
        
        # Create a dedicated CUDA stream
        self.stream = torch.cuda.Stream(device=device)
        
        # Create the scheduler
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_karras_sigmas=False
        )
        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)
        
        # Pre-cache timesteps and sigmas
        self.timesteps = self.noise_scheduler.timesteps.to(device)
        self.sigmas = self.noise_scheduler.sigmas.to(device)
        
        # Static inputs/outputs
        self.static_latent = None
        self.static_gpt_output = None
        self.static_time_embedding = None
        self.static_output = None
        
        # Create a CUDA Graph for each timestep
        self.graphs = {}
        self.static_inputs = {}
        self.static_outputs = {}
        
        self._create_graphs()
    
    def _create_graphs(self):
        """Create CUDA Graphs for each diffusion timestep"""
        print(f"  [CUDA Graph] Creating Diffusion Head CUDA Graphs ({self.num_inference_steps} steps)...")
        
        # Ensure operations run on the correct device
        with torch.cuda.device(self.device):
            for step_idx, timestep in enumerate(self.timesteps):
                # Create static inputs
                static_latent = torch.randn(2, 1, self.face_dim, device=self.device)  # bs=2 for CFG
                static_gpt_output = torch.randn(2, 1, self.face_dim, device=self.device)
                
                # Time embedding
                t_batch = torch.full((1,), timestep, device=self.device, dtype=torch.long)
                static_time_embedding = self.time_embed(t_batch).unsqueeze(1)
                
                # Warm up (on the dedicated stream)
                with torch.cuda.stream(self.stream):
                    for _ in range(3):
                        with torch.no_grad():
                            _ = self.diffusion_head(static_latent, static_gpt_output, temb=static_time_embedding)
                
                self.stream.synchronize()
                
                # Capture the CUDA Graph (on the dedicated stream)
                g = torch.cuda.CUDAGraph()
                
                with torch.cuda.stream(self.stream):
                    with torch.cuda.graph(g, stream=self.stream):
                        with torch.no_grad():
                            static_output = self.diffusion_head(static_latent, static_gpt_output, temb=static_time_embedding)
                
                self.graphs[step_idx] = g
                self.static_inputs[step_idx] = {
                    'latent': static_latent,
                    'gpt_output': static_gpt_output,
                    'time_embedding': static_time_embedding,
                }
                self.static_outputs[step_idx] = static_output
            
            print(f"  [CUDA Graph] Diffusion Head CUDA Graphs created")
    
    def forward_step(self, step_idx: int, latent: torch.Tensor, gpt_output: torch.Tensor):
        """Run one diffusion step with CUDA Graph"""
        # Copy inputs into the static buffer
        self.static_inputs[step_idx]['latent'].copy_(latent)
        self.static_inputs[step_idx]['gpt_output'].copy_(gpt_output)
        
        # Replay the graph
        self.graphs[step_idx].replay()
        
        # Return a copy of the output
        return self.static_outputs[step_idx].clone()



def resample_audio(audio_24k: np.ndarray, src_sr: int = FACE_AUDIO_RATE_RECV,
                  target_sr: int = FACE_AUDIO_RATE_MODEL) -> np.ndarray:
    """Resample 24 kHz audio to 16 kHz"""
    with _prof("face.resample_audio(librosa)", extra=f"n={len(audio_24k)} src={src_sr} tgt={target_sr}"):
        audio_24k = np.asarray(audio_24k, dtype=np.float32).flatten()
        audio_16k = librosa.resample(audio_24k, orig_sr=src_sr, target_sr=target_sr)
        return audio_16k


class StreamingFaceInfer:
    """
    Streaming face inference runner (using CUDA Graph optimization)
    Each input contains contextual audio and outputs the blendshape for the current 0.4 seconds
    """

    def __init__(self, ckpt_path: str, device: str = 'cuda:0', fps: int = FACE_FPS, 
                 num_inference_steps: int = 3):
        self.device = device
        self.fps = fps
        self.num_inference_steps = num_inference_steps

        # Load the model
        self.model = Audio2FaceGPT(diffusion_num_layers=3, hidden_size=64).to(self.device)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(ckpt['gpt'], strict=True)
        print(f'[face] Loaded checkpoint from {ckpt_path}.')
        self.model.eval()

        # Load the mean and standard deviation
        mean_std_path = f"./stats/arkit_mean_std_30fps.npy"
        npy_data = np.load(mean_std_path, allow_pickle=True)[None][0]
        mean = npy_data['mean']
        std = npy_data['std']
        self.arkit_mean = torch.tensor(mean).float().to(self.device)
        self.arkit_std = torch.tensor(std).float().to(self.device)

        self.audio_unit = FACE_AUDIO_RATE_MODEL // self.fps  # Audio samples per frame

        # Initialize historical motion tokens for autoregression
        self.pre_motion_token = torch.zeros([64, FACE_BLENDSHAPE_DIM]).to(self.device)
        self.stride = 12  # Generate 12 frames each time (0.4 seconds at 30 fps)
        
        # Fixed audio length: 63 frames at 16 kHz
        self.audio_len = FACE_CONTEXT_FRAMES * (FACE_AUDIO_RATE_MODEL // self.fps)  # 63 * (16000 // 30) = 33600
        self.pre_motion_len = 64 - self.stride  # 52
        
        # ===== Create CUDA Graph optimization components =====
        # Use a dedicated CUDA stream to avoid conflicts in multi-GPU environments
        print(f"[face] Initializing CUDA Graph optimization components...")
        
        # Audio Encoder CUDA Graph
        self.cuda_graph_audio = CUDAGraphAudioEncoder(
            model=self.model,
            audio_len=self.audio_len,
            frame_num=FACE_CONTEXT_FRAMES,
            device=self.device
        )
        
        # GPT Blocks CUDA Graph (created for each T)
        self.cuda_graph_gpt = CUDAGraphGPTBlocks(
            model=self.model,
            pre_motion_len=self.pre_motion_len,
            gen_len=self.stride,
            device=self.device
        )
        
        # Diffusion Head CUDA Graph
        self.cuda_graph_diffusion = CUDAGraphDiffusionHead(
            diffusion_head=self.model.diffusion_head,
            time_embed=self.model.time_embed,
            face_dim=self.model.face_dim,
            hidden_size=self.model.hidden_size,
            num_inference_steps=self.num_inference_steps,
            device=self.device
        )
        
        # Pre-create the scheduler (used during inference)
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_karras_sigmas=False
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        self.timesteps = self.noise_scheduler.timesteps.to(self.device)
        self.sigmas = self.noise_scheduler.sigmas.to(self.device)
        
        print(f"[face] CUDA Graph initialization complete!")

    def reset(self):
        """Reset state"""
        self.pre_motion_token = torch.zeros([64, FACE_BLENDSHAPE_DIM]).to(self.device)

    @torch.no_grad()
    def infer_chunk(self, audio_16k: np.ndarray) -> np.ndarray:
        """
        Inference method optimized with CUDA Graph
        
        Args:
            audio_16k: 16 kHz audio data as a NumPy array; must provide 63 frames of context
        
        Returns:
            blendshape: NumPy array with shape (stride, FACE_BLENDSHAPE_DIM)
        """
        # Convert to tensor
        audio = torch.from_numpy(np.asarray(audio_16k, dtype=np.float32)).float().to(self.device)
        
        # Normalize
        audio = (audio - audio.mean()) / (audio.std() + 1e-5)
        audio = audio.unsqueeze(0)  # [1, audio_len]
        
        # Ensure the audio length is correct (pad or truncate)
        if audio.shape[1] < self.audio_len:
            pad_len = self.audio_len - audio.shape[1]
            audio = F.pad(audio, (pad_len, 0), mode='constant', value=0)
        elif audio.shape[1] > self.audio_len:
            audio = audio[:, -self.audio_len:]
        
        gen_len = self.stride
        pre_motion = self.pre_motion_token[-(64 - gen_len):]  # Take the last 52 frames as context
        
        # ===== Run inference with CUDA Graph =====
        
        # 1. Extract audio features with CUDA Graph
        audio_feat = self.cuda_graph_audio.forward(audio)
        audio_hidden = torch.cat([audio_feat * 0, audio_feat], dim=0)  # CFG: [2, 63, hidden_size]
        
        # Preallocate output
        gen_tokens = torch.empty(gen_len, self.model.face_dim, device=self.device, dtype=pre_motion.dtype)
        
        all_motion = pre_motion.clone()
        cfg_audio = self.model.cfg_audio
        
        # Autoregressively generate each frame
        for frame_idx in range(gen_len):
            T = self.pre_motion_len + frame_idx
            motion_feat = self.model.face_embed(all_motion)
            
            # Build GPT inputs
            x = motion_feat.unsqueeze(0).expand(2, -1, -1)
            
            # 2. Run GPT blocks with CUDA Graph (CUDA Graph optimized)
            x = self.cuda_graph_gpt.forward(T, x, audio_hidden[:, :T, :])
            
            x = self.model.output_norm(x[:, -1:])
            gpt_output_t = self.model.output_proj(x)
            
            # Initialize noise
            latent_t = torch.randn(1, 1, self.model.face_dim, device=self.device, dtype=gpt_output_t.dtype)
            latent_t = torch.cat([latent_t, latent_t], dim=0)  # CFG requires 2
            
            self.noise_scheduler._step_index = None
            
            # 3. Run diffusion steps with CUDA Graph (CUDA Graph optimized)
            for step_idx, timestep in enumerate(self.timesteps):
                # Run the diffusion step with CUDA Graph
                output_batch = self.cuda_graph_diffusion.forward_step(step_idx, latent_t, gpt_output_t)
                
                noise_pred_uncond, noise_pred_cond = output_batch.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + cfg_audio * (noise_pred_cond - noise_pred_uncond)
                
                sigma_idx = self.noise_scheduler.step_index
                if sigma_idx is None:
                    self.noise_scheduler._init_step_index(timestep)
                    sigma_idx = self.noise_scheduler.step_index
                sigma = self.sigmas[sigma_idx]
                
                velocity = (latent_t[:1] - noise_pred) / (sigma + 1e-9)
                latent_t_new = self.noise_scheduler.step(velocity, timestep, latent_t[:1], return_dict=False)[0]
                latent_t = torch.cat([latent_t_new, latent_t_new], dim=0)
            
            gen_tokens[frame_idx] = latent_t[0].squeeze()
            all_motion = torch.cat([all_motion, latent_t[0]], dim=0)
        
        motion_token = gen_tokens
        
        # Update historical motion tokens
        self.pre_motion_token = torch.cat([self.pre_motion_token, motion_token], dim=0)
        if self.pre_motion_token.shape[0] > 128:
            self.pre_motion_token = self.pre_motion_token[-128:]
        
        # Denormalize
        motion_coef = motion_token * self.arkit_std + self.arkit_mean

        motion_coef = convert_52_to_47_fast(motion_coef.detach().cpu().numpy())
        return motion_coef


class BlendshapeSmoother:
    """
    Blendshape smoother: within-block SavGol smoothing + block-boundary blending + 30fps-to-60fps linear interpolation
    """

    def __init__(self, window_length: int = 3, poly_order: int = 2,
                 boundary_blend_frames: int = 3, upsample: bool = True):
        self.window_length = window_length
        self.poly_order = poly_order
        self.blend_frames = boundary_blend_frames
        self.upsample = upsample
        self.prev_block_end = None  # Last few frames of the previous block

    def process(self, blendshape_block: np.ndarray) -> np.ndarray:
        # Step 1: Smooth
        blendshape_smooth = self._smooth_block(blendshape_block)

        # Step 2: Interpolate 30fps -> 60fps
        if self.upsample:
            return self._interpolate_30to60(blendshape_smooth)
        return blendshape_smooth

    def _smooth_block(self, blendshape_block: np.ndarray) -> np.ndarray:
        from scipy.signal import savgol_filter

        blendshape_block = np.asarray(blendshape_block, dtype=np.float32)
        n_frames = blendshape_block.shape[0]

        # SavGol smoothing
        if n_frames >= self.window_length:
            blendshape_smooth = savgol_filter(blendshape_block, self.window_length,
                                              self.poly_order, axis=0)
        else:
            blendshape_smooth = blendshape_block.copy()

        # Block-boundary blending
        if self.prev_block_end is not None:
            blendshape_smooth = self._blend_boundary(self.prev_block_end, blendshape_smooth)

        # Save the end of the current block for the next boundary blend
        self.prev_block_end = blendshape_smooth[-self.blend_frames:].copy()

        return blendshape_smooth

    def _interpolate_30to60(self, blendshape: np.ndarray) -> np.ndarray:
        """Output 2*N frames"""
        n_frames = blendshape.shape[0]
        blendshape_60fps = np.zeros((2 * n_frames,) + blendshape.shape[1:], dtype=blendshape.dtype)

        # Place original frames at even indices [0, 2, 4, ...]
        blendshape_60fps[0::2] = blendshape

        # Place interpolated frames at odd indices [1, 3, 5, ...]
        blendshape_60fps[1:-1:2] = (blendshape[:-1] + blendshape[1:]) / 2
        blendshape_60fps[-1] = blendshape[-1]

        return blendshape_60fps

    def _blend_boundary(self, prev_blendshape: np.ndarray, curr_blendshape: np.ndarray) -> np.ndarray:
        """Smooth the transition at block boundaries"""
        blend_len = min(self.blend_frames, len(curr_blendshape))

        for i in range(blend_len):
            t = (i + 1) / (blend_len + 1)
            curr_blendshape[i] = (1 - t) * prev_blendshape[-1] + t * curr_blendshape[i]

        return curr_blendshape

    def reset(self):
        self.prev_block_end = None

select_bone_names = "Root_M Hip_R HipPart1_R Knee_R KneePart1_R Ankle_R Toes_R ToesEnd_R Heel_R HeelEnd_R Spine1_M Spine1Part1_M Chest_M Scapula_R Shoulder_R ShoulderPart1_R Elbow_R ElbowPart1_R Wrist_R MiddleFinger1_R MiddleFinger2_R MiddleFinger3_R MiddleFinger4_R ThumbFinger1_R ThumbFinger2_R ThumbFinger3_R ThumbFinger4_R IndexFinger1_R IndexFinger2_R IndexFinger3_R IndexFinger4_R Cup_R PinkyFinger1_R PinkyFinger2_R PinkyFinger3_R PinkyFinger4_R RingFinger1_R RingFinger2_R RingFinger3_R RingFinger4_R Neck_M NeckPart1_M Head_M Head_angleFix L_eye_jnt L_eScale_jnt Eye_L L_pupil_jnt R_eye_jnt R_eScale_jnt Eye_R R_pupil_jnt Scapula_L Shoulder_L ShoulderPart1_L Elbow_L ElbowPart1_L Wrist_L MiddleFinger1_L MiddleFinger2_L MiddleFinger3_L MiddleFinger4_L ThumbFinger1_L ThumbFinger2_L ThumbFinger3_L ThumbFinger4_L IndexFinger1_L IndexFinger2_L IndexFinger3_L IndexFinger4_L Cup_L PinkyFinger1_L PinkyFinger2_L PinkyFinger3_L PinkyFinger4_L RingFinger1_L RingFinger2_L RingFinger3_L RingFinger4_L Hip_L HipPart1_L Knee_L KneePart1_L Ankle_L Toes_L ToesEnd_L Heel_L HeelEnd_L".split()


state_dict = {
    "idle": [7, 10, 28, 280],
    "old": [93, 156, 497, 511],
    "angry": [2, 4, 7, 9, 11, 15, 75, 100, 106, 149, 173, 186, 208, 252, 293, 339, 358, 368, 387, 389, 416, 433, 437, 441, 490, 493],
    "raise up left hand":  [116, 224, 433],
    "raise up right hand":  [123, 260, 300, 406],
    "raise up both hands":[27, 59, 86, 174, 179,495-512 * 6],
    "look around":[43, 210],
    #"thinking":  [187, 406],
    # "thinking":  [504],
    # "disagree": [4, 6, 7, 15, 42, 86, 125, 271, 323, 330, 389, 404, 431, 437, 453, 488, 496, 498],
    "disagree": [27, 174, 399, 453, 488],
    "give up":[316, 348],
    "raise up both hands higher": [316, 348],
    "point to left":[61, 72, 79, 85, 100, 173, 373, 422, 503],
    "point to right":[391],
    "relax stand":[2, 208, 495-512 * 6],
}

style_system_prompt = "none_tem0"
# style_system_prompt = "disagree"

style_system_prompt = "none_tem1"

def get_system_prompt(name: str):
    """
    motion-example system prompt:
    - Inject a set of example motion tokens into the system message (placed here in the upper codebook segment: +6*512)
    - Also return the real vocab IDs for these tokens (156690 + 6*512 + token) for logit_bias
    """
    print(f"DEBUG1: {name}")
    if name in state_dict:
        print(f"DEBUG2: {name}")
        system_prompt = state_dict[name]
        upper_motion_example_tokens_set_str = "".join([f"<|motion_{(token + 512 * 6):04d}|>" for token in system_prompt])
        upper_motion_example_tokens_set = [token + 156690 + 6 * 512 for token in system_prompt]
        return upper_motion_example_tokens_set, upper_motion_example_tokens_set_str
    return None, None



from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
class CyclicMotionTokenProcessorV1_BP(AdapterLogitsProcessor):
    """
    Periodic motion-token constraint LogitsProcessor for vLLM v1
    Supports body-part-specific VQVAE with use_motion_rvq_num * 3 codebook cycles in total.
    The order at each timestep is: lower (6) -> upper (6) -> hands (6)
    """

    _nb_code: int = int(os.environ.get('MOTION_NB_CODE', 512))
    _use_motion_rvq_num: int = 6  # Number of RVQ layers per body part
    _total_rvq_num: int = 18  # Three body parts, 18 layers in total
    _motion_vocab_start_index: int = 156690

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        super().__init__(vllm_config, device, is_pin_memory)
        self.device = device
        self.vllm_config = vllm_config
        self.is_pin_memory = is_pin_memory

        # Precompute the token ID range for each codebook
        # The cycle period is total_rvq_num (that is, use_motion_rvq_num * 3 = 18)
        self._codebook_ranges: list[tuple[int, int]] = []
        for q in range(self._total_rvq_num):
            start = self._motion_vocab_start_index + q * self._nb_code
            end = start + self._nb_code
            self._codebook_ranges.append((start, end))

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ):
        def _mask_logits(output_ids: list[int], logits_row: torch.Tensor
                         ) -> torch.Tensor:
            step = len(output_ids)
            # Use total_rvq_num as the cycle period
            current_quantizer_idx = step % self._total_rvq_num
            start_token_id, end_token_id = self._codebook_ranges[current_quantizer_idx]
            neg_inf = float("-inf")

            if start_token_id > 0:
                logits_row[:start_token_id] = neg_inf
            if end_token_id < logits_row.shape[-1]:
                logits_row[end_token_id:] = neg_inf
            return logits_row

        return _mask_logits

    def is_argmax_invariant(self) -> bool:
        return False

    @classmethod
    def configure(cls, nb_code: int, use_motion_rvq_num: int,
                  motion_vocab_start_index: int):
        cls._nb_code = nb_code
        cls._use_motion_rvq_num = use_motion_rvq_num
        cls._total_rvq_num = use_motion_rvq_num * 3  # Three body parts
        cls._motion_vocab_start_index = motion_vocab_start_index
        
        # Recompute codebook ranges
        cls._codebook_ranges = []
        for q in range(cls._total_rvq_num):
            start = motion_vocab_start_index + q * nb_code
            end = start + nb_code
            cls._codebook_ranges.append((start, end))
        
        print(f"CyclicMotionTokenProcessorV1_BP configured: nb_code={nb_code}, "
              f"rvq_num={use_motion_rvq_num}, total_rvq_num={cls._total_rvq_num}, "
              f"start_index={motion_vocab_start_index}")

    @classmethod
    def get_config(cls) -> dict:
        return {
            "nb_code": cls._nb_code,
            "use_motion_rvq_num": cls._use_motion_rvq_num,
            "total_rvq_num": cls._total_rvq_num,
            "motion_vocab_start_index": cls._motion_vocab_start_index,
        }



HOST = '0.0.0.0'

##################### Start the text server and receive style_system_prompt #####################

# This line starts its own server thread, so it can be started at any time


TEXT_SERVER_PORT = 12346

def handle_text_message(text):
    """Handle the received text message"""
    global style_system_prompt
    style_system_prompt = text
    print(f"Processing text: {text}")

def text_server_thread():
    """Standalone text server thread"""
    text_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    text_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    text_server.bind((HOST, TEXT_SERVER_PORT))
    text_server.listen(1)
    print(f"Text server listening on port {TEXT_SERVER_PORT}...")
    
    while True:
        try:
            text_client, addr = text_server.accept()
            print(f"Text client connected from: {addr}")
            
            while True:
                # Receive the length header
                length_bytes = text_client.recv(4)
                if not length_bytes:
                    break
                msg_length = int.from_bytes(length_bytes, byteorder='big')
                # Receive text data
                data = b''
                while len(data) < msg_length:
                    chunk = text_client.recv(min(msg_length - len(data), 4096))
                    if not chunk:
                        break
                    data += chunk
                
                text_message = data.decode('utf-8')
                print(f"Received text: {text_message}")
                handle_text_message(text_message)
                
        except Exception as e:
            print(f"Text server error: {e}")
            try:
                text_client.close()
            except:
                pass

# Start the text server thread
text_thread = threading.Thread(target=text_server_thread, daemon=True)
text_thread.start()

################################################################

from scipy.signal import savgol_filter
import numpy as np

class BlockQuaternionSmoother:
    """
    Within-block bidirectional SavGol smoothing + causal block-boundary blending + 30fps-to-60fps SLERP interpolation
    """
    def __init__(self, window_length=5, poly_order=2, boundary_blend_frames=3, upsample=True):
        """
        window_length: SavGol window (odd number)
        poly_order: Polynomial order
        boundary_blend_frames: Number of boundary transition frames
        upsample: Whether to run 30fps-to-60fps interpolation
        """
        self.window_length = window_length
        self.poly_order = poly_order
        self.blend_frames = boundary_blend_frames
        self.upsample = upsample
        self.prev_block_end = None  # Last few frames of the previous block
    
    def process(self, quat_block, trans_block,smooth=True):
        """
        Full processing flow: smoothing + interpolation
        
        Args:
            quat_block: shape (N, num_bones, 4)  # N=12 for 0.4s @ 30fps
            trans_block: shape (N, 3)
        
        Returns:
            quat_out: shape (2*N-1, num_bones, 4) if upsample else (N, ...)
            trans_out: shape (2*N-1, 3) if upsample else (N, 3)
        """
        # Step 1: Smooth
        if smooth:
            quat_smooth, trans_smooth = self._smooth_block(quat_block, trans_block)
        else:
            quat_smooth, trans_smooth = quat_block, trans_block
        
        # Step 2: Interpolate 30fps -> 60fps
        if self.upsample:
            quat_out, trans_out = self._interpolate_30to60(quat_smooth, trans_smooth)
        else:
            quat_out, trans_out = quat_smooth, trans_smooth
        
        return quat_out, trans_out
    
    def _smooth_block(self, quat_block, trans_block):
        """Within-block smoothing + boundary blending"""
        n_frames, num_bones, _ = quat_block.shape
        
        # ============ Within-block bidirectional SavGol smoothing ============
        # Apply SavGol directly to translation
        if n_frames >= self.window_length:
            trans_smooth = savgol_filter(trans_block, self.window_length, 
                                         self.poly_order, axis=0)
        else:
            trans_smooth = trans_block.copy()
        
        # Quaternions: align signs first, then filter, then normalize
        quat_aligned = self._align_quaternions(quat_block)
        
        if n_frames >= self.window_length:
            quat_smooth = savgol_filter(quat_aligned, self.window_length,
                                        self.poly_order, axis=0)
        else:
            quat_smooth = quat_aligned.copy()
        
        # Normalize quaternions
        quat_smooth = quat_smooth / np.linalg.norm(quat_smooth, axis=-1, keepdims=True)
        
        # ============ Block-boundary blending ============
        if self.prev_block_end is not None:
            quat_smooth, trans_smooth = self._blend_boundary(
                self.prev_block_end['quat'], 
                self.prev_block_end['trans'],
                quat_smooth, 
                trans_smooth
            )
        
        # Save the end of the current block for the next boundary blend
        self.prev_block_end = {
            'quat': quat_smooth[-self.blend_frames:].copy(),
            'trans': trans_smooth[-self.blend_frames:].copy()
        }
        
        return quat_smooth, trans_smooth
    
    def _interpolate_30to60(self, quat, trans):
        """
        30fps -> 60fps, with the frame count exactly doubled
        N frames -> 2*N frames
        """
        n_frames = quat.shape[0]
        result_frames = 2 * n_frames  # Exactly doubled!
        
        quat_60fps = np.zeros((result_frames,) + quat.shape[1:], dtype=quat.dtype)
        trans_60fps = np.zeros((result_frames,) + trans.shape[1:], dtype=trans.dtype)
        
        # Place original frames at even indices: 0, 2, 4, ..., 2*(N-1)
        quat_60fps[0::2] = quat
        trans_60fps[0::2] = trans
        
        # Place intermediate frames at odd indices: 1, 3, 5, ..., 2*N-1
        # First N-1 intermediate frames: midpoint with the next frame
        quat_60fps[1:-1:2] = self._slerp_midpoint(quat[:-1], quat[1:])
        trans_60fps[1:-1:2] = (trans[:-1] + trans[1:]) / 2
        
        # Last intermediate frame (index = 2*N-1): copy the last frame
        # Extrapolation could be used, but copying is safer
        quat_60fps[-1] = quat[-1]
        trans_60fps[-1] = trans[-1]
        
        return quat_60fps, trans_60fps
    
    def _slerp_midpoint(self, q0, q1):
        """
        Batch-compute the midpoint between two quaternions (t=0.5)
        q0, q1: shape (..., 4)
        """
        q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
        q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
        
        dot = np.sum(q0 * q1, axis=-1, keepdims=True)
        q1 = np.where(dot < 0, -q1, q1)
        
        result = q0 + q1
        return result / np.linalg.norm(result, axis=-1, keepdims=True)
    
    def _slerp(self, q0, q1, t):
        """Batch SLERP for any t value"""
        q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
        q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
        
        dot = np.sum(q0 * q1, axis=-1, keepdims=True)
        q1 = np.where(dot < 0, -q1, q1)
        
        result = (1 - t) * q0 + t * q1
        return result / np.linalg.norm(result, axis=-1, keepdims=True)
    
    def _align_quaternions(self, quats):
        """Align quaternion signs to avoid jumps"""
        aligned = quats.copy()
        for i in range(1, len(aligned)):
            dot = np.sum(aligned[i-1] * aligned[i], axis=-1, keepdims=True)
            aligned[i] = np.where(dot < 0, -aligned[i], aligned[i])
        return aligned
    
    def _blend_boundary(self, prev_quat, prev_trans, curr_quat, curr_trans):
        """Smooth the transition at block boundaries"""
        blend_len = min(self.blend_frames, len(curr_quat))
        
        for i in range(blend_len):
            t = (i + 1) / (blend_len + 1)
            curr_quat[i] = self._slerp(prev_quat[-1], curr_quat[i], t)
            curr_trans[i] = (1 - t) * prev_trans[-1] + t * curr_trans[i]
        
        return curr_quat, curr_trans
    
    def reset(self):
        """Reset state (call for a new session)"""
        self.prev_block_end = None


class Converter_unity:
    # Converts BVH rotation and translation data into a Unity-compatible format
    def __init__(
        self,
        transforms_file="./stats/magic_transforms.json",
        rest_matrices_file="./stats/magic_rest_matrices.json",
    ):
        self.transforms = {}
        self.transforms_inv = {}
        self._load_transforms(transforms_file)

        a = json.load(open(rest_matrices_file))
        rest_pose_matrix = []
        for name in select_bone_names:
            rest_pose_matrix.append(a[name])

        # Use float32 consistently to avoid unnecessary float64 values
        self.rest_pose_matrix = np.asarray(rest_pose_matrix, dtype=np.float32)[:, :3, :3]
        self.rest_pose_matrix_inv = np.linalg.inv(self.rest_pose_matrix).astype(np.float32)

        # Pre-cache as torch CPU tensors (rc only accepts torch tensors)
        self.rest_pose_matrix_t = torch.from_numpy(self.rest_pose_matrix)          # (J,3,3)
        self.rest_pose_matrix_inv_t = torch.from_numpy(self.rest_pose_matrix_inv)  # (J,3,3)

        self.bone_rest_matrix_inv = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.bone_rest_matrix_inv_t = torch.from_numpy(self.bone_rest_matrix_inv)

    def _load_transforms(self, filename):
        if not os.path.exists(filename):
            print(f"Error: Transforms file not found at '{filename}'")
            return
        with open(filename, "r") as f:
            data = json.load(f)

        for name, matrix_list in data.items():
            transform_matrix = np.asarray(matrix_list, dtype=np.float32)
            transform_inv = np.linalg.inv(transform_matrix).astype(np.float32)

            # Store torch tensors directly to avoid per-frame conversion
            self.transforms[name] = torch.from_numpy(transform_matrix)      # (3,3)
            self.transforms_inv[name] = torch.from_numpy(transform_inv)     # (3,3)

        print(f"Converter initialized with {len(self.transforms)} pre-computed transforms.")

    def convert_rest(self, pose_data):
        """
        pose_data: (T, J, 4)  quaternion wxyz
        return:    (T, J, 4)  quaternion wxyz
        """
        pose_data = np.asarray(pose_data, dtype=np.float32)
        seq_len, num_bones, _ = pose_data.shape

        q = torch.from_numpy(pose_data.reshape(seq_len * num_bones, 4))  # wxyz
        Rm = rc.quaternion_to_matrix(q).reshape(seq_len, num_bones, 3, 3)

        # (J,3,3) @ (T,J,3,3) @ (J,3,3) -> (T,J,3,3)
        Rm = torch.matmul(self.rest_pose_matrix_inv_t, torch.matmul(Rm, self.rest_pose_matrix_t))

        q_out = rc.matrix_to_quaternion(Rm.reshape(seq_len * num_bones, 3, 3)).reshape(seq_len, num_bones, 4)
        return q_out.numpy()

    def convert_quat(self, joint_name, bvh_quat):
        """
        bvh_quat: (T,4) wxyz
        return:   (T,4) wxyz
        """
        if joint_name not in self.transforms:
            print(f"Error: No pre-computed transform found for joint '{joint_name}'.")
            return None

        M_transform = self.transforms[joint_name]        # (3,3)
        M_transform_inv = self.transforms_inv[joint_name]

        q = torch.from_numpy(np.asarray(bvh_quat, dtype=np.float32))  # (T,4) wxyz
        M_pose = rc.quaternion_to_matrix(q)                           # (T,3,3)

        M_final = torch.matmul(M_transform, torch.matmul(M_pose, M_transform_inv))  # (T,3,3)
        q_final = rc.matrix_to_quaternion(M_final)                                  # (T,4) wxyz
        return q_final.numpy()

    def convert(self, pose_data, trans):
        pose_data = np.asarray(pose_data, dtype=np.float32)
        trans = np.asarray(trans, dtype=np.float32)

        pose_data = self.convert_rest(pose_data)
        
        # Vectorized: convert all bones at once
        seq_len, num_bones, _ = pose_data.shape
        q = torch.from_numpy(pose_data)  # (T, J, 4)
        
        # Stack all transform matrices into (J, 3, 3)
        M_transform = torch.stack([self.transforms[name] for name in select_bone_names])      # (J, 3, 3)
        M_transform_inv = torch.stack([self.transforms_inv[name] for name in select_bone_names])  # (J, 3, 3)
        
        # Batch-convert quaternions to matrices
        M_pose = rc.quaternion_to_matrix(q.reshape(-1, 4)).reshape(seq_len, num_bones, 3, 3)  # (T, J, 3, 3)
        
        # Batch matrix multiplication: (J,3,3) @ (T,J,3,3) @ (J,3,3) -> (T,J,3,3)
        M_final = torch.einsum('jab,tjbc,jcd->tjad', M_transform, M_pose, M_transform_inv)
        
        # Batch-convert back to quaternions
        q_out = rc.matrix_to_quaternion(M_final.reshape(-1, 3, 3)).reshape(seq_len, num_bones, 4)
        pose_data = q_out.numpy()

        trans = trans @ self.bone_rest_matrix_inv.T
        pose_data = np.concatenate([pose_data[..., 1:], pose_data[..., 0:1]], axis=-1)
        return pose_data, trans


###################### audio to hubert code #######################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Place the audio tokenizer on cuda:1 (same as face/motion) to avoid conflicts with vLLM (cuda:0)
AUDIO_DEVICE = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'

feature_extractor = EncodecFeatureExtractor()

audio_tokenizer = EncodecModel.encodec_model_24khz().eval().to(AUDIO_DEVICE)
print(f"[Audio Tokenizer] moved to {AUDIO_DEVICE}")

n_q = 2
codebook_num = 1024
pre_motion_tokens=27
motion_token_fps = 7.5
audio_token_fps = 75*n_q
motion_fps = 30

use_motion_rvq_num=6
@torch.no_grad()
def wav2hubertcode(wav,sr =24000):

    audio_tokens_list = []
    with _prof("body.wav2hubertcode(total)", extra=f"samples={len(wav)} sr={sr}", sync_cuda=False):
        time1 = time.time()
        with _prof("body.wav2hubertcode.feature_extractor", extra=f"samples={len(wav)}"):
            inputs = feature_extractor(raw_audio=wav[:,0], sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
        time2 = time.time()
        print(f"feature_extractor time: {time2-time1:.4f}s")

        with _prof("body.wav2hubertcode.encodec.encode", sync_cuda=True):
            encoder_outputs = audio_tokenizer.encode(inputs["input_values"].to(AUDIO_DEVICE))[0][0]
        time3 = time.time()
        print(f"encode time: {time3-time2:.4f}s")

        # NOTE: flatten().tolist() will force GPU->CPU sync/copy if encoder_outputs is on CUDA
        with _prof("body.wav2hubertcode.postproc(to_list)", sync_cuda=True):
            audio_tokens = encoder_outputs.squeeze()[:n_q]
            audio_tokens = audio_tokens.transpose(0,1).flatten().tolist()
        time4 = time.time()
        print(f"transpose time: {time4-time3:.4f}s")

        with _prof("body.wav2hubertcode.codebook_offset"):
            for idx in range(len(audio_tokens)):
                q = idx%n_q
                audio_tokens[idx] = audio_tokens[idx] + codebook_num*q
        time5 = time.time()
        print(f"codebook_num time: {time5-time4:.4f}s")

        with _prof("body.wav2hubertcode.to_tokens_str"):
            audio_tokens_list+=audio_tokens
            audio_tokens = [f'<|audio_{num:04d}|>' for num in audio_tokens_list]
        time7 = time.time()
        print(f"audio_tokens time: {time7-time5:.4f}s")
        return audio_tokens





from process_zm_dataset import lower_body_indices_feature_indices, upper_body_indices_feature_indices, hands_body_indices_feature_indices

rvq_layer_num=6

ckpt_path_lower = './ckpts/body_rvqvae/RVQVAE_PoolingMLP_bp_lower_nb-code_512_commit-0.5_loss-pos-l1-0.02_loss-pos-vel-l1-0.2_loss-pos-acc-l1-0.2_loss-trans-vel-l1_smooth-10_depth-3_loss-foot-contact-label-l1-0.3_loss-foot-pos-l1-0.05_dropout-0_num_quantizers-6_lookback-15/net_65000.pth'
ckpt_path_upper = './ckpts/body_rvqvae/RVQVAE_PoolingMLP_bp_upper_nb-code_512_commit-0.5_loss-pos-l1-0.02_loss-pos-vel-l1-0.2_loss-pos-acc-l1-0.2_loss-trans-vel-l1_smooth-10_depth-3_loss-foot-contact-label-l1-0.3_loss-foot-pos-l1-0.05_dropout-0_num_quantizers-6_lookback-15/net_54000.pth'
ckpt_path_hands = './ckpts/body_rvqvae/RVQVAE_PoolingMLP_bp_hands_nb-code_512_commit-0.5_loss-pos-l1-0.02_loss-pos-vel-l1-0.2_loss-pos-acc-l1-0.2_loss-trans-vel-l1_smooth-10_depth-3_loss-foot-contact-label-l1-0.3_loss-foot-pos-l1-0.05_dropout-0_num_quantizers-6_lookback-15/net_60000.pth'

nb_code_lower = int(re.search(r'nb-code_(\d+)', ckpt_path_lower).group(1))
nb_code_upper = int(re.search(r'nb-code_(\d+)', ckpt_path_upper).group(1))
nb_code_hands = int(re.search(r'nb-code_(\d+)', ckpt_path_hands).group(1))

# Place the motion tokenizer on cuda:1 (same as face) to avoid conflicts with vLLM (cuda:0)
MOTION_DEVICE = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'

motion_tokenizer_lower = get_causal_attn_PoolingMLP4_rvqvae_model(
                                                            ckpt_path_lower,
                                                            dim_pose=len(lower_body_indices_feature_indices),
                                                            rvq_layer_num=rvq_layer_num, 
                                                            nb_code=int(re.search(r'nb-code_(\d+)', ckpt_path_lower).group(1)))

motion_tokenizer_upper = get_causal_attn_PoolingMLP4_rvqvae_model(
                                                            ckpt_path_upper,
                                                            dim_pose=len(upper_body_indices_feature_indices),
                                                            rvq_layer_num=rvq_layer_num, 
                                                            nb_code=int(re.search(r'nb-code_(\d+)', ckpt_path_upper).group(1)))
motion_tokenizer_hands = get_causal_attn_PoolingMLP4_rvqvae_model(
                                                            ckpt_path_hands,
                                                            dim_pose=len(hands_body_indices_feature_indices),
                                                            rvq_layer_num=rvq_layer_num, 
                                                            nb_code=int(re.search(r'nb-code_(\d+)', ckpt_path_hands).group(1)))

# Move to MOTION_DEVICE
motion_tokenizer_lower = motion_tokenizer_lower.to(MOTION_DEVICE)
motion_tokenizer_upper = motion_tokenizer_upper.to(MOTION_DEVICE)
motion_tokenizer_hands = motion_tokenizer_hands.to(MOTION_DEVICE)
print(f"[Motion Tokenizers] moved to {MOTION_DEVICE}")


# ==================== Motion Decoder CUDA Graph Optimization ====================
class CUDAGraphMotionDecoder:
    """
    Apply CUDA Graph optimization to a single Motion Tokenizer Decoder
    Use a dedicated CUDA stream to support parallel execution
    Pre-cache attention masks to avoid memory allocation during CUDA Graph capture
    """
    def __init__(self, decoder_model, seq_len=30, rvq_num=6, device='cuda:0', name='decoder'):
        self.decoder = decoder_model
        self.seq_len = seq_len
        self.rvq_num = rvq_num
        self.device = device
        self.name = name
        
        self.graph = None
        self.static_input = None
        self.static_output = None
        
        # Create a dedicated CUDA stream
        self.stream = torch.cuda.Stream(device=device)
        
        # Pre-cache attention masks (avoid creating tensors during CUDA Graph capture)
        self._precompute_attn_masks()
        
        self._create_graph()
    
    def _precompute_attn_masks(self):
        """Precompute and cache all required attention masks"""
        lookback = self.decoder.lookback
        full_seq_len = self.seq_len * 4  # In forward_decoder: seq_len = seq * 4
        
        def make_attn_mask_np(seq_len, lb):
            mask = np.zeros((seq_len, seq_len), dtype=np.float32)
            for i in range(seq_len):
                for j in range(max(0, i - lb), i + 1):
                    mask[i, j] = 1
            return mask
        
        # Precompute masks at three sizes
        mask1 = make_attn_mask_np(full_seq_len, lookback)
        mask2 = make_attn_mask_np(full_seq_len // 2, (lookback + 1) // 2 - 1)
        mask3 = make_attn_mask_np(full_seq_len // 4, (lookback + 1) // 4 - 1)
        
        # Convert to tensors and cache
        with torch.cuda.device(self.device):
            self._cached_mask1 = torch.tensor(mask1, device=self.device).bool()
            self._cached_mask2 = torch.tensor(mask2, device=self.device).bool()
            self._cached_mask3 = torch.tensor(mask3, device=self.device).bool()
        
        # Save the original method
        self._original_make_attn_mask = self.decoder.make_attn_mask
        
        # Capture outer self and full_seq_len in a closure
        outer_self = self
        cached_full_seq_len = full_seq_len
        
        # Note: must receive the decoder_self argument because this is an instance-method call
        def cached_make_attn_mask(seq_len, lookback):
            if seq_len == cached_full_seq_len:
                return outer_self._cached_mask1
            elif seq_len == cached_full_seq_len // 2:
                return outer_self._cached_mask2
            elif seq_len == cached_full_seq_len // 4:
                return outer_self._cached_mask3
            else:
                # fallback: use the original method
                return outer_self._original_make_attn_mask(seq_len, lookback)
        
        self.decoder.make_attn_mask = cached_make_attn_mask
        print(f"    [CUDA Graph] Pre-cached {self.name} attention masks: {full_seq_len}, {full_seq_len//2}, {full_seq_len//4}")
    
    def _create_graph(self):
        """Create the CUDA Graph for the Decoder"""
        print(f"  [CUDA Graph] Creating Motion Decoder ({self.name}) CUDA Graph (seq_len={self.seq_len}, rvq_num={self.rvq_num})...")
        
        with torch.cuda.device(self.device):
            # Create static inputs - shape: (1, seq_len, rvq_num)
            self.static_input = torch.zeros(1, self.seq_len, self.rvq_num, dtype=torch.long, device=self.device)
            
            # Warm up (on the dedicated stream)
            with torch.cuda.stream(self.stream):
                for _ in range(3):
                    with torch.no_grad():
                        _ = self.decoder.forward_decoder(self.static_input)
            
            self.stream.synchronize()
            
            # Capture the CUDA Graph (on the dedicated stream)
            self.graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.stream(self.stream):
                with torch.cuda.graph(self.graph, stream=self.stream):
                    with torch.no_grad():
                        self.static_output = self.decoder.forward_decoder(self.static_input)
            
            print(f"  [CUDA Graph] Motion Decoder ({self.name}) CUDA Graph created")
    
    def forward(self, tokens):
        """Run decoding with CUDA Graph"""
        # Copy inputs into the static buffer
        self.static_input.copy_(tokens)
        
        # Replay the graph
        self.graph.replay()
        
        # Return a copy of the output
        return self.static_output.clone()
    
    def forward_async(self, tokens):
        """Run decoding asynchronously (on the dedicated stream); manual synchronization is required"""
        self.static_input.copy_(tokens)
        with torch.cuda.stream(self.stream):
            self.graph.replay()
        # Do not synchronize; return the output reference (caller must synchronize before use)
        return self.static_output


class ParallelMotionDecoders:
    """
    CUDA Graph optimizer that runs three Motion Decoders in parallel
    Use three independent CUDA streams for true parallelism
    """
    def __init__(self, decoder_lower, decoder_upper, decoder_hands, 
                 seq_len=30, rvq_num=6, device='cuda:0'):
        self.device = device
        self.seq_len = seq_len
        self.rvq_num = rvq_num
        
        print(f"[Motion Decoders] Initializing parallel CUDA Graph optimization...")
        
        # Create three independent CUDA Graph decoders
        self.cuda_decoder_lower = CUDAGraphMotionDecoder(
            decoder_lower, seq_len, rvq_num, device, name='lower'
        )
        self.cuda_decoder_upper = CUDAGraphMotionDecoder(
            decoder_upper, seq_len, rvq_num, device, name='upper'
        )
        self.cuda_decoder_hands = CUDAGraphMotionDecoder(
            decoder_hands, seq_len, rvq_num, device, name='hands'
        )
        
        print(f"[Motion Decoders] Parallel CUDA Graph initialization complete!")
    
    def forward_parallel(self, tokens_lower, tokens_upper, tokens_hands):
        """
        Decode three body parts in parallel
        
        Args:
            tokens_lower: (1, seq_len, rvq_num) long tensor
            tokens_upper: (1, seq_len, rvq_num) long tensor
            tokens_hands: (1, seq_len, rvq_num) long tensor
        
        Returns:
            (rec_lower, rec_upper, rec_hands) Three decoded results
        """
        # Launch the three decoders asynchronously (on their own streams)
        out_lower = self.cuda_decoder_lower.forward_async(tokens_lower)
        out_upper = self.cuda_decoder_upper.forward_async(tokens_upper)
        out_hands = self.cuda_decoder_hands.forward_async(tokens_hands)
        
        # Synchronize all streams
        self.cuda_decoder_lower.stream.synchronize()
        self.cuda_decoder_upper.stream.synchronize()
        self.cuda_decoder_hands.stream.synchronize()
        
        # Return a copy of the output
        return out_lower.clone(), out_upper.clone(), out_hands.clone()



def convert_52_to_47_fast(arkit_frames):
    """
    Input: (Batch, 52) matrix, in the exact order of the provided CURRENT_ORDER
    Output: (Batch, 48) matrix, corresponding to Faceunity indices 0 to 47
    """
    src = np.array(arkit_frames)
    if src.ndim == 1:
        src = src[None, :] # Handle the single-frame case

    # -----------------------------------------------------------
    # 1. Simple mapping group (precomputed indices)
    # Corresponding FU indices: [1...35] (35 total)
    # -----------------------------------------------------------
    idx_part1 = [
        0, 7,   # 1,2   EyeBlink L/R
        5, 12,  # 3,4   EyeSquint L/R
        1, 8,   # 5,6   EyeDown L/R
        2, 9,   # 7,8   EyeIn L/R
        6, 13,  # 9,10  EyeOpen L/R (Wide)
        3, 10,  # 11,12 EyeOut L/R
        4, 11,  # 13,14 EyeUp L/R
        41, 42, # 15,16 BrowsD L/R
        43,     # 17    BrowsU C
        44, 45, # 18,19 BrowsU L/R
        14,     # 20    JawFwd
        15,     # 21    JawLeft
        17,     # 22    JawOpen (Note: original list index 17 is jawOpen)
        16,     # 23    JawRight
        21, 22, # 24,25 MouthLeft/Right
        25, 26, # 26,27 MouthFrown L/R
        23, 24, # 28,29 MouthSmile L/R
        27, 28, # 30,31 MouthDimple L/R
        29, 30, # 32,33 MouthStretch L/R
        32, 31  # 34,35 LipsUpperClose/LowerClose (Roll)
    ]

    # -----------------------------------------------------------
    # 2. Simple Mapping Group Part 2
    # Corresponding FU indices: [38, 39, 40, 41] (4 total)
    # -----------------------------------------------------------
    idx_part2 = [
        34,     # 38 MouthUp (ShrugUpper)
        19,     # 39 LipsFunnel
        20,     # 40 LipsPucker
        33      # 41 ChinLowerRaise (ShrugLower)
    ]

    # -----------------------------------------------------------
    # 3. Simple Mapping Group Part 3
    # Corresponding FU indices: [44, 45, 46, 47] (4 total)
    # -----------------------------------------------------------
    idx_part3 = [
        46,     # 44 Puff
        47, 48, # 45,46 CheekSquint L/R
        18      # 47 MouthOpenClose (Close)
    ]

    # -----------------------------------------------------------
    # Run extraction
    # -----------------------------------------------------------
    
    # Extract all simple-mapping columns
    p1 = src[:, idx_part1]
    p2 = src[:, idx_part2]
    p3 = src[:, idx_part3]

    # Compute merged columns (take max of left/right)
    # 36: LipsUpperUp (UpperUp L/R: 39, 40)
    c36 = np.maximum(src[:, 39], src[:, 40])[:, None]
    
    # 37: LipsLowerDown (LowerDown L/R: 37, 38)
    c37 = np.maximum(src[:, 37], src[:, 38])[:, None]
    
    # 42: ChinUpperRaise (Press L/R: 35, 36)
    c42 = np.maximum(src[:, 35], src[:, 36])[:, None]
    
    # 43: Sneer (Sneer L/R: 49, 50)
    c43 = np.maximum(src[:, 49], src[:, 50])[:, None]


    # -----------------------------------------------------------
    # Concatenate output (strictly in 0-47 order)
    # -----------------------------------------------------------
    return np.hstack([
        p1,   # 1-35
        c36,  # 36
        c37,  # 37
        p2,   # 38-41
        c42,  # 42
        c43,  # 43
        p3    # 44-47
    ])

def parse_args():
    parser = argparse.ArgumentParser(
        description="Streaming vLLM Unity server for body/face motion inference."
    )
    parser.add_argument(
        "--model_name",
        "--model-name",
        default="./ckpts/body_g_d",
        help="Body LLM model checkpoint path or Hugging Face model name.",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    model_name = args.model_name
    print(f"[body] model_name={model_name}")
    logits_processors = [CyclicMotionTokenProcessorV1_BP]
    llm = LLM(model=model_name,
                gpu_memory_utilization=0.6, 
                logits_processors=logits_processors,
                )   

    tokenizer = AutoTokenizer.from_pretrained(model_name)


    default_qwen_system_prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"


    allow_motion_tokens = list(range(156690, 156690+nb_code_lower*use_motion_rvq_num*3))


    past_key_values = None

    def get_gpt_generation_result(audio_tokens, motion_tokens,pre_motion_tokens=3):
        global past_key_values
        audio_seg_num = 30*n_q
        prefix_audio_num = (300-30)*n_q
        seg_num = (len(audio_tokens)-prefix_audio_num) // audio_seg_num
        motion_tokens_str = "".join(motion_tokens[-pre_motion_tokens:])
        for i in range(seg_num):
            with _prof("body.vllm.build_prompt", extra=f"seg={i+1}/{seg_num}"):
                motion_seed_tokens = motion_tokens_str[-15*pre_motion_tokens:]    #One motion token is 15 characters; take 3*15=45 characters as the seed
                audio_input = "".join(audio_tokens[audio_seg_num*i:prefix_audio_num+audio_seg_num*(i+1)])
                sample_allow_motion_tokens = list(set(allow_motion_tokens))
            example_tok_ids, example_tok_ids_str = get_system_prompt(style_system_prompt)
            if example_tok_ids is not None:
                text = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": example_tok_ids_str},
                        {"role": "user", "content": audio_input},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                text = text + motion_seed_tokens
            else:
                text = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": audio_input},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                text = text[len(default_qwen_system_prompt):] + motion_seed_tokens

            print(f"style_system_prompt: {style_system_prompt}")
            temperature = 1
            if style_system_prompt == "none_tem0":
                temperature = 0
            if style_system_prompt == "none_tem2":
                temperature = 2

            sampling_params = SamplingParams(
                max_tokens=30 * use_motion_rvq_num * 3 - pre_motion_tokens,
                temperature = temperature,
                logit_bias={tid: 20.0 for tid in set(example_tok_ids)} if example_tok_ids is not None else None,
                # frequency_penalty = 2 if style_system_prompt != "none_tem0" else 0,
                # logits_processors=[cyclic_processor]
            )
            with _prof("body.vllm.generate", extra=f"seg={i+1}/{seg_num}", sync_cuda=False):
                outputs = llm.generate(text,sampling_params=sampling_params)
            response = outputs[0].outputs[0].text
            
            # end_time = time.time()
            # print(f"Time to generate motion segment {i}: {end_time-start_time:.2f}s")
            
            motion_tokens_str+=response
        
        matches = re.findall(r'<\|motion_\d{4}\|>', motion_tokens_str)
        motion_tokens_list = [match for match in matches]
        return motion_tokens_list


    mean, std = np.load('./stats/body_mean_std_30fps.npy')
    std = std + 1e-10
    pose_mean = mean
    pose_std = std


    # Place on the same device as the motion tokenizer
    mean_tensor = torch.from_numpy(pose_mean).float().to(MOTION_DEVICE)
    std_tensor = torch.from_numpy(pose_std).float().to(MOTION_DEVICE)


    default_meta_info_path = './stats/zm_meta_info.npz'
    default_meta_info = np.load(default_meta_info_path)
    default_skeleton = default_meta_info['offsets']
    default_skeleton = torch.tensor(default_skeleton).float().to(MOTION_DEVICE)

    parents = default_meta_info['parents']
    bone_names = default_meta_info['names']


    from process_zm_dataset import forward_kinematics



    last_record_trans_x = torch.tensor(0.0).to(MOTION_DEVICE)
    last_record_trans_y = torch.tensor(0.0).to(MOTION_DEVICE)
    last_record_z_rot = torch.tensor(0.0).to(MOTION_DEVICE)


    from process_zm_dataset import repre_to_bvh
    all_results = []    
    #This is for inspection at a debug breakpoint
    # Then use
    #    pred_motion = np.concatenate(all_results,axis=1)[0]
    #    unnormal_motion = pred_motion*pose_std + pose_mean   
    #    repre_to_bvh(unnormal_motion.copy(),f"test.bvh")
    # to obtain BVH

    
    def get_joint_pos(pred_motion):
        
        bs,seq,_ = pred_motion.shape
        device = pred_motion.device
        
        out_poses = pred_motion* std_tensor + mean_tensor
        
        rot6d = out_poses[..., :-3].reshape(bs,seq,88,6)
        rot_mat = rc.rotation_6d_to_matrix(rot6d)
        
        rot_mat= rot_mat.reshape(bs*seq,88,3,3)
        
        rot_quat = rc.matrix_to_quaternion(rot_mat).reshape(bs*seq,88,4)
        
        trans_vx = out_poses[..., -3:-2]
        trans_vy = out_poses[..., -2:-1]
        trans_z = out_poses[..., -1:]
        
        trans_x = torch.cumsum(trans_vx, dim = -2)
        trans_y = torch.cumsum(trans_vy, dim = -2)
        
        trans = torch.concat([trans_x,trans_y,trans_z],dim=-1)
        
        
        return rot_quat,trans


    from utils.cylindrical_hand_ik1_fast_lbfgs import realtime_hands_ik_solver_lbfgs
    myrealtime_hands_ik_solver = realtime_hands_ik_solver_lbfgs(iterations=10, use_compile=False)
    
    from utils.swing_only_ik import realtime_hands_ik_solver_analytical

    myrealtime_hands_ik_solver = realtime_hands_ik_solver_analytical()

    # Initialize parallel Motion Decoder CUDA Graphs (placed on MOTION_DEVICE to avoid vLLM conflicts)
    parallel_motion_decoders = ParallelMotionDecoders(
        decoder_lower=motion_tokenizer_lower,
        decoder_upper=motion_tokenizer_upper,
        decoder_hands=motion_tokenizer_hands,
        seq_len=30,  # Generate 30 frames each time
        rvq_num=use_motion_rvq_num,
        device=MOTION_DEVICE
    )
    
    def motion_token_to_motion_axix_angle(motion_tokens,pre_motion_tokens):
        global last_record_trans_x
        global last_record_trans_y
        with _prof("body.motion_token_to_motion(total)", extra=f"tokens={len(motion_tokens)}"):
            convert_to_tokenlist= motion_tokens
            with _prof("body.motion_token_to_motion.token_offset"):
                for i in range(len(convert_to_tokenlist)):
                    token_num = convert_to_tokenlist[i]
                    new_token_num = token_num - (i%(use_motion_rvq_num*3)*nb_code_lower)
                    convert_to_tokenlist[i] = new_token_num

            convert_to_tokenlist_lower = []
            convert_to_tokenlist_upper = []
            convert_to_tokenlist_hands = []

            with _prof("body.motion_token_to_motion.split_parts"):
                for i in range(len(convert_to_tokenlist)):
                    if i%18<6:
                        convert_to_tokenlist_lower.append(convert_to_tokenlist[i])
                    elif i%18<12:
                        convert_to_tokenlist_upper.append(convert_to_tokenlist[i])
                    elif i%18<18:
                        convert_to_tokenlist_hands.append(convert_to_tokenlist[i])
        
            # Decode with parallel CUDA Graphs (on MOTION_DEVICE)
            with _prof("body.motion_token_to_motion.tensorize", sync_cuda=False):
                tokens_lower = torch.tensor(convert_to_tokenlist_lower, device=MOTION_DEVICE).reshape(1, -1, use_motion_rvq_num)
                tokens_upper = torch.tensor(convert_to_tokenlist_upper, device=MOTION_DEVICE).reshape(1, -1, use_motion_rvq_num)
                tokens_hands = torch.tensor(convert_to_tokenlist_hands, device=MOTION_DEVICE).reshape(1, -1, use_motion_rvq_num)
        
            time1 = time.time()
            with _prof("body.motion_decoder.forward_parallel", sync_cuda=True):
                rec_lower_pose, rec_upper_pose, rec_hands_pose = parallel_motion_decoders.forward_parallel(
                    tokens_lower, tokens_upper, tokens_hands
                )
            torch.cuda.synchronize()
            time2 = time.time()
            print(f"forward parallel time: {time2-time1:.4f}s")
            rec_normal_pose = torch.zeros((1, rec_lower_pose.shape[1], 531), device=MOTION_DEVICE)


            with _prof("body.motion_decoder.scatter_merge"):
                rec_normal_pose[..., lower_body_indices_feature_indices] = rec_lower_pose
                rec_normal_pose[..., upper_body_indices_feature_indices] = rec_upper_pose
                rec_normal_pose[..., hands_body_indices_feature_indices] = rec_hands_pose

            with _prof("body.motion_decoder.debug_cpu_copy(all_results)", sync_cuda=True):
                all_results.append(rec_normal_pose[:,-12:].detach().cpu().numpy())
            with _prof("body.get_joint_pos", sync_cuda=False):
                pose_quat,trans = get_joint_pos(rec_normal_pose[:,-12:])
        

            pose_quat = pose_quat
            trans = trans[0]
            
            trans[:,0] = trans[:,0] + last_record_trans_x
            trans[:,1] = trans[:,1] + last_record_trans_y
        
            last_record_trans_x = trans[-1,0]
            last_record_trans_y = trans[-1,1]

            with _prof("body.ik.detach_to_cpu", sync_cuda=True):
                pose_quat = pose_quat.detach().cpu()
                trans = trans.detach().cpu()
            try:
                with _prof("body.ik.solve(lbfgs)", extra="iterations=10"):
                    pose_quat = myrealtime_hands_ik_solver.solve(rc.quaternion_to_matrix(pose_quat), trans)
            except Exception as e:
                print(f"[ERROR] IK solve failed: {e}")
                traceback.print_exc()
                # Use the original pose_quat as a fallback
                pose_quat = rc.matrix_to_quaternion(rc.quaternion_to_matrix(pose_quat))
        
            pose_quat = rc.matrix_to_quaternion(pose_quat)
            with _prof("body.motion_token_to_motion.final_numpy"):
                return pose_quat.numpy(), trans.numpy(), None



    ###################################################################


    # Server settings



    # Audio settings
    rate = 24000
    channels = 1
    output_file = 'received_audio.wav'

    # Create an audio buffer to accumulate enough audio data for processing
    BUFFER_SIZE = rate*2//5  # Number of samples for 1 second
    audio_buffer = deque(maxlen=BUFFER_SIZE)
    hubert_codes = []
    motion_codes = [f"<|motion_{32+nb_code_lower*i:04d}|>" for i in range(0, 18)]*27#<|motion_{32+nb_code_lower:04d}|><|motion_{32+nb_code_lower*2:04d}|><|motion_{32+nb_code_lower*3:04d}|><|motion_{32+nb_code_lower*4:04d}|><|motion_{32+nb_code_lower*5:04d}|>"]*27
    def setup_motion_client():
        """Establish a persistent connection to the motion server"""
        try:
            motion_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            motion_client.connect((MOTION_SERVER_HOST, MOTION_SERVER_PORT))
            print(f"Connected to motion server {MOTION_SERVER_HOST}:{MOTION_SERVER_PORT}")
            return motion_client
        except Exception as e:
            print(f"Failed to connect to motion server: {e}")
            return None

    def save_motion_to_json_append(motion_axis_angle, filepath="saved_motion_data.json"):
        """
        Append motion_axis_angle data to a JSONL file
        Method 2: append line by line (suitable for large datasets)
        """
        try:
            # Prepare the data to save
            new_data = {
                "motion_axis_angle": motion_axis_angle.tolist() if isinstance(motion_axis_angle, np.ndarray) else motion_axis_angle
            }
            
            # Append to file
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                
            print(f"Motion data appended to {filepath}")
            
        except Exception as e:
            print(f"Error while saving motion data: {e}")

    converter_unity = Converter_unity()

    smoother = BlockQuaternionSmoother(window_length=5, poly_order=2, 
                                        boundary_blend_frames=3, upsample=True)


    def send_motion_data(client_socket, motion_axis_angle, trans, expression,audio):
        """
        Send motion data to the server
        """ 
        try:
            time1 = time.time()
            motion_axis_angle,trans = converter_unity.convert(motion_axis_angle,trans)
            time2 = time.time()
            print(f"converter_unity time: {time2-time1:.4f}s")
            
            
            with _prof("sender.body_smoother.process"):
                motion_axis_angle, trans = smoother.process(motion_axis_angle, trans,smooth=False)
            # motion_axis_angle = np.repeat(motion_axis_angle, 2, axis=0)
            # trans = np.repeat(trans, 2, axis=0)

            # expression: face blendshape (expected to be aligned to 60 fps, 24 frames)
            if expression is None:
                expression = np.zeros((motion_axis_angle.shape[0], FACE_BLENDSHAPE_DIM), dtype=np.float32)
            expression = np.asarray(expression, dtype=np.float32)

            # If the expression frame count does not match, align it to motion_axis_angle as much as possible
            with _prof("sender.align_expression", extra=f"expr={expression.shape[0]} pose={motion_axis_angle.shape[0]}"):
                if expression.shape[0] != motion_axis_angle.shape[0]:
                    if expression.shape[0] * 2 == motion_axis_angle.shape[0]:
                        # 30fps -> 60fps (exactly doubled)
                        expr_60 = np.zeros((motion_axis_angle.shape[0],) + expression.shape[1:], dtype=expression.dtype)
                        expr_60[0::2] = expression
                        expr_60[1:-1:2] = (expression[:-1] + expression[1:]) / 2
                        expr_60[-1] = expression[-1]
                        expression = expr_60
                    elif expression.shape[0] < motion_axis_angle.shape[0]:
                        pad = np.repeat(expression[-1:], motion_axis_angle.shape[0] - expression.shape[0], axis=0)
                        expression = np.concatenate([expression, pad], axis=0)
                    else:
                        expression = expression[:motion_axis_angle.shape[0]]

            # audio: supports list[np.ndarray], list[float], or np.ndarray
            with _prof("sender.prepare_audio", extra=f"type={type(audio).__name__}"):
                if isinstance(audio, list):
                    if len(audio) == 0:
                        audio_arr = np.zeros((0,), dtype=np.float32)
                    else:
                        try:
                            audio_arr = np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in audio], axis=0)
                        except Exception:
                            audio_arr = np.asarray(audio, dtype=np.float32).reshape(-1)
                else:
                    audio_arr = np.asarray(audio, dtype=np.float32).reshape(-1)

            with _prof("sender.tolist", extra=f"pose={len(motion_axis_angle)} audio={len(audio_arr)}"):
                chunk_data = {
                    'pose': motion_axis_angle.tolist(),
                    'trans': trans.tolist(),
                    'blendshape': expression.tolist(),
                    'audio': audio_arr.tolist(),
                    
                }
            # with _prof("sender.dumps_only"):
            #     message_bytes = json.dumps(chunk_data).encode('utf-8')
            with _prof("sender.dumps_only"):
                message_bytes = orjson.dumps(chunk_data)  # Already bytes, UTF-8 JSON
            # 2. Prepend the message length as a 4-byte integer (big-endian)
            message_header = len(message_bytes).to_bytes(4, byteorder='big')
            # Send JSON data directly without a length prefix
            try:
                with _prof("sender.socket.sendall", extra=f"bytes={len(message_bytes)}"):
                    client_socket.sendall(message_header)
                    client_socket.sendall(message_bytes)
                #save_motion_to_json_append(motion_axis_angle.tolist())
            except socket.error as e:
                print(f"Error while sending data chunk: {e}")
                raise
                
        except Exception as e:
            print(f"Error while sending motion data: {e}")
            try:
                client_socket.close()
            except:
                pass
            raise

        return True

    # ==================== Parallel: body/face inference + sender packaging and sending ====================
    # Initialize the face model
    face_ckpt_path = "./ckpts/face/llf_v3_FaceGPT_CE_audio_norm_dit_flowmatching_mhubert_augment_ws-64_layers-6_heads-8_feat-512_bs-128_diffusion-layers-3_hidden-size-64/gpt_80000.pth"
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        face_device = "cuda:1"
    else:
        face_device = "cuda:0"
    print(f"[face] device={face_device}, fps={FACE_FPS}")
    face_infer = StreamingFaceInfer(face_ckpt_path, device=face_device, fps=FACE_FPS)
    face_smoother = BlendshapeSmoother(window_length=5, poly_order=2, boundary_blend_frames=3, upsample=True)

    # task/result queues
    body_task_q: "queue.Queue[tuple[int, np.ndarray] | None]" = queue.Queue(maxsize=32)
    face_task_q: "queue.Queue[tuple[int, np.ndarray] | None]" = queue.Queue(maxsize=32)
    result_q: "queue.Queue[tuple]" = queue.Queue(maxsize=128)

    # motion socket holder (for reconnecting inside threads)
    motion_client_box = {"sock": None}

    def body_worker():
        """Only runs body inference and outputs 30fps (12-frame) pose/trans plus the corresponding 0.4-second audio chunk"""
        hubert_codes_local: list[str] = []
        motion_codes_local: list[str] = [f"<|motion_{32 + nb_code_lower * i:04d}|>" for i in range(0, 18)] * pre_motion_tokens
        history_wav_local: list[float] = []
        last_t = time.time()

        while True:
            item = body_task_q.get()
            if item is None:
                result_q.put(("done", "body"))
                return

            time1 = time.time()
            chunk_id, audio_chunk_24k = item
            audio_chunk_24k = np.asarray(audio_chunk_24k, dtype=np.float32).flatten()
            time2 = time.time()
            print(f"get item time: {time2-time1:.4f}s")

            # Maintain 4 seconds of history (for EnCodec token extraction)
            history_wav_local.extend(audio_chunk_24k.tolist())
            max_hist = int(rate * 4)
            if len(history_wav_local) > max_hist:
                history_wav_local = history_wav_local[-max_hist:]
            time3 = time.time()
            print(f"extend history time: {time3-time2:.4f}s")
            # EnCodec tokens: take tokens for the last 0.4 seconds of history and append them to hubert_codes
            wav_2d = np.asarray(history_wav_local, dtype=np.float32).reshape(-1, 1)
            code_all = wav2hubertcode(wav_2d)
            code_last = code_all[-int(0.4 * audio_token_fps):]
            hubert_codes_local += code_last
            time4 = time.time()
            print(f"wav2hubertcode time: {time4-time3:.4f}s")
            # Start producing body results only after 4 seconds of tokens are ready
            if len(hubert_codes_local) < 4 * 75 * n_q:
                continue

            t0 = time.time()
            try:
                motion_tokens = get_gpt_generation_result(
                    hubert_codes_local[-300 * n_q:],
                    motion_codes_local[-pre_motion_tokens * use_motion_rvq_num * 3:],
                    pre_motion_tokens * use_motion_rvq_num * 3
                )
                re_motion = [int(match[9:13]) for match in motion_tokens]
                motion_axix_angle, trans, _ = motion_token_to_motion_axix_angle(
                    re_motion, pre_motion_tokens * use_motion_rvq_num * 3
                )
            except Exception:
                print(f"[body] chunk {chunk_id} inference exception:")
                traceback.print_exc()
                continue

            t1 = time.time()
            print(f"[body] chunk={chunk_id} motion consuming {t1 - t0:.3f}s, after {t1 - last_t:.3f}s")
            last_t = t1

            # Output to sender: body (30fps) plus the corresponding 0.4-second audio chunk (sender packages it with face)
            result_q.put(("body", chunk_id, motion_axix_angle, trans, audio_chunk_24k))

            # Update context
            motion_codes_local = motion_tokens[-pre_motion_tokens * use_motion_rvq_num * 3:]
            hubert_codes_local = hubert_codes_local[int(-pre_motion_tokens * audio_token_fps // motion_token_fps):]

    def face_worker():
        """Only runs face inference and outputs 60fps (24-frame) blendshape"""
        history_audio_local: list[float] = []
        context_audio_len_24k = FACE_CONTEXT_FRAMES * (FACE_AUDIO_RATE_RECV // FACE_FPS)  # 63 * 800 = 50400
        last_t = time.time()

        while True:
            item = face_task_q.get()
            if item is None:
                result_q.put(("done", "face"))
                return

            chunk_id, audio_chunk_24k = item
            audio_chunk_24k = np.asarray(audio_chunk_24k, dtype=np.float32).flatten()

            history_audio_local.extend(audio_chunk_24k.tolist())
            max_hist = int(FACE_AUDIO_RATE_RECV * 10)  # 10 seconds
            if len(history_audio_local) > max_hist:
                history_audio_local = history_audio_local[-max_hist:]

            # Take 63 context frames (zero-pad if insufficient)
            if len(history_audio_local) >= context_audio_len_24k:
                audio_for_infer_24k = np.asarray(history_audio_local[-context_audio_len_24k:], dtype=np.float32)
            else:
                audio_for_infer_24k = np.zeros((context_audio_len_24k,), dtype=np.float32)
                hist_arr = np.asarray(history_audio_local, dtype=np.float32)
                audio_for_infer_24k[-len(hist_arr):] = hist_arr

            try:
                audio_for_infer_16k = resample_audio(audio_for_infer_24k, FACE_AUDIO_RATE_RECV, FACE_AUDIO_RATE_MODEL)
                t0 = time.time()
                with _prof("face.infer_chunk", extra=f"chunk={chunk_id}", sync_cuda=True):
                    blendshape_30 = face_infer.infer_chunk(audio_for_infer_16k)
                with _prof("face.smoother.process", extra=f"chunk={chunk_id}"):
                    blendshape_60 = face_smoother.process(blendshape_30)
                t1 = time.time()
                print(f"[face] chunk={chunk_id} consuming {t1 - t0:.3f}s, after {t1 - last_t:.3f}s")
                last_t = t1
            except Exception:
                print(f"[face] chunk {chunk_id} inference exception:")
                traceback.print_exc()
                continue

            result_q.put(("face", chunk_id, blendshape_60))

    def sender_worker():
        """Synchronize and package by chunk_id: body (pose/trans) + face (blendshape) + audio (0.4s)"""
        body_buf: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        face_buf: dict[int, np.ndarray] = {}
        done = set()
        started = False
        next_id = None

        while True:
            msg = result_q.get()
            if msg[0] == "done":
                done.add(msg[1])
                # After both workers finish, flush as much sendable data as possible and exit
                if done == {"body", "face"}:
                    if started and next_id is not None:
                        while next_id in body_buf and next_id in face_buf:
                            motion_axix_angle, trans, audio_chunk = body_buf.pop(next_id)
                            blendshape = face_buf.pop(next_id)
                            sock = motion_client_box["sock"]
                            if sock is not None:
                                try:
                                    send_motion_data(sock, motion_axix_angle, trans, blendshape, audio_chunk)
                                except Exception:
                                    pass
                            next_id += 1
                    return
                continue

            kind = msg[0]
            if kind == "body":
                _, chunk_id, motion_axix_angle, trans, audio_chunk = msg
                body_buf[int(chunk_id)] = (motion_axix_angle, trans, audio_chunk)
            elif kind == "face":
                _, chunk_id, blendshape = msg
                face_buf[int(chunk_id)] = blendshape
            else:
                continue

            # First send: find the first chunk_id that has both body and face data
            if not started:
                ready = sorted(set(body_buf.keys()) & set(face_buf.keys()))
                if not ready:
                    continue
                next_id = ready[0]
                # Discard earlier unaligned results from the warmup stage
                for k in list(body_buf.keys()):
                    if k < next_id:
                        body_buf.pop(k, None)
                for k in list(face_buf.keys()):
                    if k < next_id:
                        face_buf.pop(k, None)
                started = True

            # Send in order
            while started and next_id in body_buf and next_id in face_buf:
                motion_axix_angle, trans, audio_chunk = body_buf.pop(next_id)
                blendshape = face_buf.pop(next_id)

                sock = motion_client_box["sock"]
                if sock is None:
                    print("[sender] motion_client is empty; skipping send")
                    next_id += 1
                    continue

                try:
                    ok = send_motion_data(sock, motion_axix_angle, trans, blendshape, audio_chunk)
                except Exception:
                    ok = False

                if not ok:
                    print("[sender] Send failed; trying to reconnect to the motion server...")
                    try:
                        sock.close()
                    except Exception:
                        pass
                    motion_client_box["sock"] = setup_motion_client()
                    sock2 = motion_client_box["sock"]
                    if sock2 is not None:
                        try:
                            send_motion_data(sock2, motion_axix_angle, trans, blendshape, audio_chunk)
                        except Exception:
                            pass

                next_id += 1

    # Start worker threads
    t_body = threading.Thread(target=body_worker, daemon=True)
    t_face = threading.Thread(target=face_worker, daemon=True)
    t_sender = threading.Thread(target=sender_worker, daemon=True)
    t_body.start()
    t_face.start()
    t_sender.start()

    # ==================== Network receive: single audio entry point ====================
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"Waiting for connection on port {PORT}...")

    # Establish a persistent connection to the motion server
    motion_client_box["sock"] = setup_motion_client()
    if motion_client_box["sock"] is None:
        print("Unable to connect to the motion server; exiting")
        server_socket.close()
        exit(1)

    client_socket, address = server_socket.accept()
    print(f"Accepted connection from {address}")

    chunk_id = 0
    recv_cache: list[float] = []

    try:
        with sf.SoundFile(output_file, mode='w', samplerate=rate, channels=channels) as file:
            while True:
                size_bytes = client_socket.recv(4)
                if not size_bytes or len(size_bytes) < 4:
                    print("Connection closed")
                    break
                size = int.from_bytes(size_bytes, byteorder='big')

                if size == 0:
                    print("Transfer finished")
                    break

                data_bytes = b''
                while len(data_bytes) < size:
                    chunk = client_socket.recv(size - len(data_bytes))
                    if not chunk:
                        break
                    data_bytes += chunk

                data = pickle.loads(data_bytes)
                data = np.asarray(data, dtype=np.float32).flatten()

                # # Save raw received audio (for debugging)
                # try:
                #     file.write(data)
                # except Exception:
                #     pass

                recv_cache.extend(data.tolist())

                while len(recv_cache) >= BUFFER_SIZE:
                    audio_chunk = np.asarray(recv_cache[:BUFFER_SIZE], dtype=np.float32)
                    del recv_cache[:BUFFER_SIZE]

                    # Send the same chunk_id to the body and face workers at the same time
                    body_task_q.put((chunk_id, audio_chunk))
                    face_task_q.put((chunk_id, audio_chunk))
                    chunk_id += 1

    except KeyboardInterrupt:
        print("User interrupted")
    except Exception:
        print("Error in main thread:")
        traceback.print_exc()
    finally:
        try:
            client_socket.close()
        except Exception:
            pass
        try:
            server_socket.close()
        except Exception:
            pass

        # Notify workers to exit
        try:
            body_task_q.put(None)
            face_task_q.put(None)
        except Exception:
            pass

        # Wait for threads to finish (sender is a daemon, so it is not forcibly joined here)
        try:
            t_body.join(timeout=2.0)
            t_face.join(timeout=2.0)
        except Exception:
            pass

        # Close the motion connection
        try:
            if motion_client_box["sock"] is not None:
                motion_client_box["sock"].close()
        except Exception:
            pass

        print("All connections closed")
        
