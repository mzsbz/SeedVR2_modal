#!/usr/bin/env python3
"""
Modal CLI wrapper for SeedVR2 Video Upscaler
Calls Modal functions remotely while maintaining the same interface as inference_cli.py
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    import modal
except ImportError:
    print("❌ Error: modal package not installed. Install with: uv pip install modal")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SeedVR2 Video Upscaler CLI (Modal)")
    
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--seed", type=int, default=100,
                        help="Random seed for generation (default: 100)")
    parser.add_argument("--resolution", type=int, default=1072,
                        help="Target resolution of the short side (default: 1072)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of frames per batch (default: 1)")
    parser.add_argument("--model", type=str, default="seedvr2_ema_3b_fp8_e4m3fn.safetensors",
                        choices=[
                            "seedvr2_ema_3b_fp16.safetensors",
                            "seedvr2_ema_3b_fp8_e4m3fn.safetensors", 
                            "seedvr2_ema_7b_fp16.safetensors",
                            "seedvr2_ema_7b_fp8_e4m3fn.safetensors"
                        ],
                        help="Model to use (default: 3B FP8)")
    parser.add_argument("--skip_first_frames", type=int, default=0,
                        help="Skip the first frames during processing")
    parser.add_argument("--load_cap", type=int, default=0,
                        help="Maximum number of frames to load from video (default: 0 = load all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: auto-generated)")
    parser.add_argument("--output_format", type=str, default="video", choices=["video", "png"],
                        help="Output format: 'video' (mp4) or 'png' images (default: video)")
    parser.add_argument("--preserve_vram", action="store_true",
                        help="Enable VRAM preservation mode")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    return parser.parse_args()


def save_frames_to_png_from_bytes(video_bytes: bytes, output_dir: str, base_name: str, fps: float, debug: bool = False):
    """
    Extract frames from video bytes and save as PNG images.
    Note: This is a simplified version that requires re-encoding.
    For PNG output, it's better to return frames from Modal.
    """
    import cv2
    import numpy as np
    
    # Write video bytes to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(video_bytes)
    
    try:
        # Extract frames
        cap = cv2.VideoCapture(tmp_path)
        os.makedirs(output_dir, exist_ok=True)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            filename = f"{base_name}_{frame_idx:05d}.png"
            file_path = os.path.join(output_dir, filename)
            cv2.imwrite(file_path, frame)
            
            if debug and (frame_idx + 1) % 100 == 0:
                print(f"💾 Saved {frame_idx + 1} PNGs")
            
            frame_idx += 1
        
        cap.release()
        
        if debug:
            print(f"✅ PNG saving completed: {frame_idx} files in '{output_dir}'")
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    """Main CLI function"""
    print(f"🚀 SeedVR2 Video Upscaler CLI (Modal) started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse arguments
    args = parse_arguments()
    
    if args.debug:
        print(f"📋 Arguments:")
        for key, value in vars(args).items():
            print(f"   {key}: {value}")
    
    try:
        # Read input video
        if not os.path.exists(args.video_path):
            raise FileNotFoundError(f"Video file not found: {args.video_path}")
        
        print(f"📥 Reading video file: {args.video_path}")
        with open(args.video_path, 'rb') as f:
            video_bytes = f.read()
        
        print(f"📊 Video size: {len(video_bytes) / 1024 / 1024:.2f} MB")
        
        # Connect to Modal app
        print(f"🔌 Connecting to Modal...")
        app = modal.App.lookup("seedvr2-upscaler", create_if_missing=False)
        upscale_func = app.upscale_video
        
        # Call Modal function
        print(f"☁️ Uploading video and starting upscaling on Modal...")
        print(f"⏳ This may take a while depending on video length and model size...")
        
        result_bytes = upscale_func.remote(
            video_bytes=video_bytes,
            model=args.model,
            seed=args.seed,
            resolution=args.resolution,
            batch_size=args.batch_size,
            skip_first_frames=args.skip_first_frames,
            load_cap=args.load_cap if args.load_cap > 0 else 0,
            preserve_vram=args.preserve_vram,
            debug=args.debug,
        )
        
        print(f"✅ Upscaling completed!")
        print(f"📊 Result size: {len(result_bytes) / 1024 / 1024:.2f} MB")
        
        # Determine output path
        if args.output_format == "png":
            output_path_obj = Path(args.output) if args.output else Path(args.video_path).parent
            if output_path_obj.suffix:
                output_path_obj = output_path_obj.with_suffix('')
            output_dir = str(output_path_obj)
            base_name = Path(args.video_path).stem + "_upscaled"
            
            print(f"🖼️ Saving PNG frames to directory: {output_dir}")
            save_frames_to_png_from_bytes(result_bytes, output_dir, base_name, 30.0, args.debug)
            print(f"📁 PNG frames saved in directory: {output_dir}")
        else:
            # Save video
            if args.output:
                output_path_obj = Path(args.output)
                if not output_path_obj.suffix or args.output.endswith(os.sep) or args.output.endswith('/'):
                    os.makedirs(args.output, exist_ok=True)
                    video_name = Path(args.video_path).stem + "_upscaled.mp4"
                    output_path = str(Path(args.output) / video_name)
                else:
                    output_dir = os.path.dirname(args.output)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    output_path = args.output
            else:
                video_name = Path(args.video_path).stem + "_upscaled.mp4"
                output_path = str(Path(args.video_path).parent / video_name)
            
            print(f"💾 Saving upscaled video to: {output_path}")
            with open(output_path, 'wb') as f:
                f.write(result_bytes)
            
            print(f"📁 Output saved to video: {output_path}")
        
        print(f"✅ Processing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

