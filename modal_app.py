"""
Modal application for SeedVR2 Video Upscaler
Runs video upscaling on Modal.com with GPU support
"""

import os
import sys
import tempfile
from pathlib import Path

import modal

# Add project root to sys.path for src module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Set CUDA memory allocation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

# Create Modal app
app = modal.App("seedvr2-upscaler")

# Define the image with all dependencies
# Clone the repository during image build
REPO_URL = "https://github.com/mzsbz/SeedVR2_modal.git"  # Update this to your repo URL if different

image = (
    modal.Image.debian_slim(python_version="3.12.9")
    .apt_install("ffmpeg", "libsm6", "libxext6", "libxrender-dev", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        "safetensors",
        "numpy",
        "tqdm",
        "psutil",
        "einops",
        "omegaconf>=2.3.0",
        "diffusers>=0.33.1",
        "rotary_embedding_torch>=0.5.3",
        "opencv-python",
        "gguf",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync"})
    .run_commands(
        f"git clone {REPO_URL} /workspace",
        "cd /workspace && rm -rf .git __pycache__ .pytest_cache node_modules .venv venv _output _test .cursor modelsSEEDVR2",
    )
)

# Mount the hf-hub-cache volume
hf_cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=False)

# Define model directory path in the volume
MODEL_DIR = "/cache/seedvr2_models"


def extract_frames_from_video_bytes(video_bytes: bytes, debug: bool = False, skip_first_frames: int = 0, load_cap: int = None):
    """
    Extract frames from video bytes and convert to tensor format
    
    Args:
        video_bytes (bytes): Video file content as bytes
        debug (bool): Enable debug logging
        skip_first_frames (int): Skip the first N frames
        load_cap (int): Maximum number of frames to load (None for all)
        
    Returns:
        tuple: (frames_tensor, fps) where frames_tensor is [T, H, W, C] Float16
    """
    import cv2
    import numpy as np
    import torch
    
    if debug:
        print(f"🎬 Extracting frames from video bytes")
    
    # Write bytes to temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(video_bytes)
    
    try:
        # Open video
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if debug:
            print(f"📊 Video info: {frame_count} frames, {width}x{height}, {fps:.2f} FPS")
            if skip_first_frames:
                print(f"⏭️ Will skip first {skip_first_frames} frames")
            if load_cap:
                print(f"🔢 Will load maximum {load_cap} frames")
        
        frames = []
        frame_idx = 0
        frames_loaded = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip first frames if requested
            if frame_idx < skip_first_frames:
                frame_idx += 1
                continue
            
            # Check load cap
            if load_cap is not None and load_cap > 0 and frames_loaded >= load_cap:
                if debug:
                    print(f"🔢 Reached load cap of {load_cap} frames")
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to float32 and normalize to 0-1
            frame = frame.astype(np.float32) / 255.0
            
            frames.append(frame)
            frame_idx += 1
            frames_loaded += 1
            
            if debug and frames_loaded % 100 == 0:
                total_to_load = min(frame_count, load_cap) if load_cap else frame_count
                print(f"📹 Extracted {frames_loaded}/{total_to_load} frames")
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video")
        
        if debug:
            print(f"✅ Extracted {len(frames)} frames")
        
        # Convert to tensor [T, H, W, C] and cast to Float16
        frames_tensor = torch.from_numpy(np.stack(frames)).to(torch.float16)
        
        if debug:
            print(f"📊 Frames tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}")
        
        return frames_tensor, fps
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_frames_to_video_bytes(frames_tensor, fps: float = 30.0, debug: bool = False) -> bytes:
    """
    Save frames tensor to video bytes
    
    Args:
        frames_tensor: Frames in format [T, H, W, C] (Float16, 0-1)
        fps (float): Output video FPS
        debug (bool): Enable debug logging
        
    Returns:
        bytes: Video file content as bytes
    """
    import cv2
    import numpy as np
    
    if debug:
        print(f"🎬 Saving {frames_tensor.shape[0]} frames to video bytes")
    
    # Convert tensor to numpy and denormalize
    frames_np = frames_tensor.cpu().numpy()
    frames_np = (frames_np * 255.0).astype(np.uint8)
    
    # Get video properties
    T, H, W, C = frames_np.shape
    
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp_path, fourcc, fps, (W, H))
        
        if not out.isOpened():
            raise ValueError(f"Cannot create video writer")
        
        # Write frames
        for i, frame in enumerate(frames_np):
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
            if debug and (i + 1) % 100 == 0:
                print(f"💾 Saved {i + 1}/{T} frames")
        
        out.release()
        
        if debug:
            print(f"✅ Video saved successfully")
        
        # Read back as bytes
        with open(tmp_path, 'rb') as f:
            video_bytes = f.read()
        
        return video_bytes
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# Try to get HF secret if it exists (optional)
hf_secret = None
try:
    hf_secret = modal.Secret.from_name("hf-secret")
except Exception:
    pass  # Secret doesn't exist, that's fine

@app.function(
    image=image,
    gpu="A10G",  # Use A10G for good performance, can be changed to T4 or A100
    volumes={"/cache": hf_cache_volume},
    timeout=3600,  # 1 hour timeout
    secrets=[hf_secret] if hf_secret else [],  # Optional HF token
)
def upscale_video(
    video_bytes: bytes,
    model: str = "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    seed: int = 100,
    resolution: int = 1072,
    batch_size: int = 1,
    skip_first_frames: int = 0,
    load_cap: int = 0,
    preserve_vram: bool = False,
    debug: bool = False,
) -> bytes:
    """
    Upscale video using SeedVR2 model on Modal
    
    Args:
        video_bytes: Input video file as bytes
        model: Model name to use
        seed: Random seed for generation
        resolution: Target resolution of the short side
        batch_size: Number of frames per batch
        skip_first_frames: Skip the first N frames
        load_cap: Maximum number of frames to load (0 for all)
        preserve_vram: Enable VRAM preservation mode
        debug: Enable debug logging
        
    Returns:
        bytes: Upscaled video file as bytes
    """
    # Change to workspace directory and add to path for imports
    os.chdir("/workspace")
    if "/workspace" not in sys.path:
        sys.path.insert(0, "/workspace")
    
    from src.utils.downloads import download_weight
    from src.core.model_manager import configure_runner
    from src.core.generation import generation_loop
    
    # Ensure model directory exists in volume
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Download model if needed
    if debug:
        print(f"📥 Checking/downloading model: {model}")
    download_weight(model, MODEL_DIR)
    
    # Extract frames from video
    if debug:
        print(f"🎬 Extracting frames from video...")
    frames_tensor, original_fps = extract_frames_from_video_bytes(
        video_bytes, debug, skip_first_frames, load_cap if load_cap > 0 else None
    )
    
    if debug:
        print(f"📊 Extracted {len(frames_tensor)} frames")
    
    # Configure runner
    if debug:
        print(f"🔄 Configuring runner...")
    runner = configure_runner(model, MODEL_DIR, preserve_vram, debug)
    
    # Run generation
    if debug:
        print(f"🚀 Starting upscaling generation...")
    result = generation_loop(
        runner=runner,
        images=frames_tensor,
        cfg_scale=1.0,
        seed=seed,
        res_w=resolution,
        batch_size=batch_size,
        preserve_vram=preserve_vram,
        temporal_overlap=0,
        debug=debug,
    )
    
    if debug:
        print(f"✅ Generation completed")
        print(f"📊 Result shape: {result.shape}, dtype: {result.dtype}")
    
    # Save to video bytes
    if debug:
        print(f"💾 Saving upscaled video...")
    output_video_bytes = save_frames_to_video_bytes(result, original_fps, debug)
    
    # Commit volume changes (model downloads)
    hf_cache_volume.commit()
    
    if debug:
        print(f"✅ Upscaling completed successfully!")
    
    return output_video_bytes

