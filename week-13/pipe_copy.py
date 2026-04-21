"""
pipeline.py — Main Stable Diffusion image generation pipeline for Fashion Outfit Generator.

Integrates SD 1.5 + ControlNet (OpenPose) + IP-Adapter into a single
pipeline that generates full-body outfit images while preserving
the reference person's identity and pose.
"""

import torch
import gc
import os
from PIL import Image
from typing import List, Optional

from config import (
    SD_MODEL_ID,
    CONTROLNET_MODEL_ID,
    IP_ADAPTER_REPO,
    IP_ADAPTER_SUBFOLDER,
    IP_ADAPTER_WEIGHT_NAME,
    IP_ADAPTER_IMAGE_ENCODER_SUBFOLDER,
    DEVICE,
    DTYPE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_CONTROLNET_SCALE,
    DEFAULT_IP_ADAPTER_SCALE,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_SEED,
)
from prompt_engine import generate_structured_prompt, generate_naive_prompt, get_negative_prompt
from control import extract_and_prepare, validate_input_image


class FashionPipeline:
    """
    Unified pipeline for generating fashion outfit images using
    Stable Diffusion 1.5 + ControlNet (OpenPose) + IP-Adapter.
    """

    def __init__(self, device: str = None, dtype=None):
        """
        Initialize the pipeline.

        Models are loaded lazily on first call to load_models().

        Args:
            device: Target device ("cuda", "mps", "cpu"). Auto-detected if None.
            dtype: Torch dtype. Auto-selected if None.
        """
        self.device = device or DEVICE
        self.dtype = dtype or DTYPE
        self.pipe = None
        self.is_loaded = False
        self._load_progress_callback = None

    def set_progress_callback(self, callback):
        """Set a callback function for model loading progress updates."""
        self._load_progress_callback = callback

    def _report_progress(self, message: str):
        """Report progress through callback if set."""
        if self._load_progress_callback:
            self._load_progress_callback(message)
        else:
            print(message)

    def load_models(self):
        """
        Load all models: ControlNet, Stable Diffusion pipeline, and IP-Adapter.

        This downloads models from HuggingFace on first run (~6-8GB).
        Subsequent runs use the cached versions.
        """
        if self.is_loaded:
            self._report_progress("Models already loaded.")
            return

        from diffusers import (
            StableDiffusionControlNetPipeline,
            ControlNetModel,
            UniPCMultistepScheduler,
        )

        # Step 1: Load ControlNet (OpenPose)
        self._report_progress("Loading ControlNet (OpenPose)...")
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL_ID,
            torch_dtype=self.dtype,
        )

        # Step 2: Load Stable Diffusion + ControlNet pipeline
        self._report_progress("Loading Stable Diffusion 1.5 pipeline...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            SD_MODEL_ID,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # Use a faster scheduler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Step 3: Load IP-Adapter
        self._report_progress("Loading IP-Adapter for identity preservation...")
        self.pipe.load_ip_adapter(
            IP_ADAPTER_REPO,
            subfolder=IP_ADAPTER_SUBFOLDER,
            weight_name=IP_ADAPTER_WEIGHT_NAME,
            image_encoder_folder=IP_ADAPTER_IMAGE_ENCODER_SUBFOLDER,
        )

        # Step 4: Move to device with memory optimization
        if self.device == "cuda":
            self._report_progress("Enabling CUDA GPU memory optimization...")
            self.pipe.enable_model_cpu_offload()
            # Try xformers on CUDA
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                self._report_progress("xformers memory-efficient attention enabled.")
            except Exception:
                self._report_progress("xformers not available, continuing without it.")
        elif self.device == "mps":
            self._report_progress("Moving pipeline to MPS (Apple Silicon)...")
            self.pipe = self.pipe.to(self.device)
            self._report_progress("Running natively on MPS without attention slicing as it conflicts with IP-Adapter.")
        else:
            self._report_progress(f"Moving pipeline to {self.device}...")
            self.pipe = self.pipe.to(self.device)

        self.is_loaded = True
        self._report_progress("All models loaded successfully!")

    def generate(
        self,
        reference_image: Image.Image,
        occasion: str,
        style: str,
        color_palette: str,
        num_images: int = 1,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        controlnet_scale: float = DEFAULT_CONTROLNET_SCALE,
        ip_adapter_scale: float = DEFAULT_IP_ADAPTER_SCALE,
        width: int = DEFAULT_IMAGE_WIDTH,
        height: int = DEFAULT_IMAGE_HEIGHT,
        seed: Optional[int] = DEFAULT_SEED,
        use_naive_prompt: bool = False,
        outfit_override: str = None,
    ) -> dict:
        """
        Generate fashion outfit images from a reference image and structured inputs.

        Args:
            reference_image: PIL Image of the reference person.
            occasion: Event type (e.g., "wedding").
            style: Design language (e.g., "formal").
            color_palette: Color description.
            num_images: Number of images to generate.
            num_inference_steps: Diffusion steps (higher = better quality, slower).
            guidance_scale: Classifier-free guidance scale.
            controlnet_scale: ControlNet conditioning strength (0-1).
            ip_adapter_scale: IP-Adapter identity strength (0-1).
            width: Output image width.
            height: Output image height.
            seed: Random seed for reproducibility. None = random.
            use_naive_prompt: If True, use naive (baseline) prompt instead of structured.
            outfit_override: Optional manual outfit description.

        Returns:
            Dictionary with:
                - 'images': List of generated PIL Images
                - 'prompt': The prompt used
                - 'negative_prompt': The negative prompt used
                - 'pose_image': The extracted pose map
                - 'reference_image': The preprocessed reference image
                - 'parameters': Generation parameters used
        """
        if not self.is_loaded:
            self.load_models()

        # Validate input
        validation = validate_input_image(reference_image)
        if not validation["valid"]:
            raise ValueError(f"Invalid input image: {validation['message']}")

        # Extract pose and prepare reference
        self._report_progress("Extracting pose and preparing reference image...")
        control_data = extract_and_prepare(
            reference_image,
            target_width=width,
            target_height=height,
        )

        pose_image = control_data["pose_image"]
        prepared_ref = control_data["reference_image"]

        # Generate prompt
        if use_naive_prompt:
            prompt = generate_naive_prompt(occasion)
        else:
            prompt = generate_structured_prompt(
                occasion, style, color_palette, outfit_override
            )
        negative_prompt = get_negative_prompt()

        # Set IP-Adapter scale
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)

        # Set up generator for reproducibility
        # NOTE: MPS and CPU both require generator on "cpu"
        generator = None
        if seed is not None:
            gen_device = "cpu"  # Generator must be on CPU for MPS/CPU compatibility
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        # Run generation
        self._report_progress(f"Generating {num_images} image(s)...")
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pose_image,
            ip_adapter_image=prepared_ref,
            num_images_per_prompt=num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            width=width,
            height=height,
            generator=generator,
        )

        self._report_progress("Generation complete!")

        return {
            "images": result.images,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "pose_image": pose_image,
            "reference_image": prepared_ref,
            "parameters": {
                "occasion": occasion,
                "style": style,
                "color_palette": color_palette,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "controlnet_scale": controlnet_scale,
                "ip_adapter_scale": ip_adapter_scale,
                "width": width,
                "height": height,
                "seed": seed,
                "use_naive_prompt": use_naive_prompt,
            },
        }

    def generate_comparison(
        self,
        reference_image: Image.Image,
        occasion: str,
        style: str,
        color_palette: str,
        num_images: int = 1,
        seed: int = 42,
        **kwargs,
    ) -> dict:
        """
        Generate both naive and structured outputs for side-by-side comparison.

        Uses the same seed for both to ensure a fair comparison.

        Args:
            reference_image: PIL Image of the reference person.
            occasion, style, color_palette: Structured inputs.
            num_images: Number of images per mode.
            seed: Fixed seed for reproducibility.
            **kwargs: Additional generation parameters.

        Returns:
            Dictionary with 'naive' and 'structured' results.
        """
        self._report_progress("Generating naive baseline...")
        naive_result = self.generate(
            reference_image=reference_image,
            occasion=occasion,
            style=style,
            color_palette=color_palette,
            num_images=num_images,
            seed=seed,
            use_naive_prompt=True,
            **kwargs,
        )

        self._report_progress("Generating structured output...")
        structured_result = self.generate(
            reference_image=reference_image,
            occasion=occasion,
            style=style,
            color_palette=color_palette,
            num_images=num_images,
            seed=seed,
            use_naive_prompt=False,
            **kwargs,
        )

        return {
            "naive": naive_result,
            "structured": structured_result,
        }

    def unload_models(self):
        """Unload all models to free memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self.is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._report_progress("Models unloaded.")

    @staticmethod
    def _compute_inpaint_dims(
        image: Image.Image,
        min_size: int = 512,
        max_size: int = 768,
        multiple_of: int = 8,
    ):
        """
        Compute SD-compatible (w, h) from the input image.

        Rules:
        - Preserve aspect ratio.
        - Clamp the shorter side to [min_size, max_size].
        - Round both sides to multiples of `multiple_of` (required by SD VAE).
        - Enforce a minimum of min_size on *both* sides to avoid tiny face crops.
        """
        orig_w, orig_h = image.size
        aspect = orig_w / orig_h

        # Scale so the shorter side == min_size (at least)
        if orig_w <= orig_h:
            new_w = min_size
            new_h = int(round(new_w / aspect))
        else:
            new_h = min_size
            new_w = int(round(new_h * aspect))

        # Clamp the longer side to max_size
        if new_w > max_size:
            new_w = max_size
            new_h = int(round(new_w / aspect))
        if new_h > max_size:
            new_h = max_size
            new_w = int(round(new_h * aspect))

        # Snap to multiples of eight
        new_w = max(multiple_of, (new_w // multiple_of) * multiple_of)
        new_h = max(multiple_of, (new_h // multiple_of) * multiple_of)

        return new_w, new_h

    def generate_inpaint(self, reference_image, occasion, style, color_palette, seed=42, strength=0.95, **kwargs):
        from diffusers import StableDiffusionInpaintPipeline
        from segment import get_clothing_mask
        import gc

        # Unload existing ControlNet pipe first to free MPS memory
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.is_loaded = False
            gc.collect()
            torch.mps.empty_cache()  # MPS-specific cache clear

        # inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-inpainting",
        #     torch_dtype=torch.float16,   # force float16, not float32
        #     safety_checker=None,
        #     variant="fp16",              # load fp16 weights directly (smaller download)
        # ).to(self.device)

        # inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-inpainting",
        #     torch_dtype=torch.float32,
        #     safety_checker=None,
        # ).to("cpu")   # force CPU

        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float32,  # must be float32
            safety_checker=None,
        ).to("mps")

        # Force MPS sync — fixes black frame bug
        inpaint_pipe.unet = inpaint_pipe.unet.to(torch.float32)
        inpaint_pipe.vae = inpaint_pipe.vae.to(torch.float32)

        # Reduce memory during inference
        inpaint_pipe.enable_attention_slicing()

        # ── Dynamic dimensions ───────────────────────────────────────────
        # Derive target resolution from the input image so small images are
        # upscaled to a sensible minimum (avoids face distortion) and large
        # images are downscaled to keep memory usage reasonable.
        w, h = self._compute_inpaint_dims(reference_image)
        self._report_progress(f"Inpainting at {w}×{h} (derived from {reference_image.size[0]}×{reference_image.size[1]})")

        ref = reference_image.resize((w, h), Image.LANCZOS)
        mask = get_clothing_mask(ref)
        prompt = generate_structured_prompt(occasion, style, color_palette)

        generator = torch.Generator("cpu").manual_seed(seed)

        result = inpaint_pipe(
            prompt=prompt,
            negative_prompt=get_negative_prompt(),
            image=ref,
            mask_image=mask,
            width=w, height=h,
            strength=strength,
            num_inference_steps=30,   # reduced from 50
            guidance_scale=7.5,
            generator=generator,
        )

        # Free inpaint pipe after done
        del inpaint_pipe
        gc.collect()
        torch.mps.empty_cache()

        return {"images": result.images, "mask": mask, "prompt": prompt, "reference_image": ref}

# ──────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────
_global_pipeline = None


def get_pipeline(device: str = None) -> FashionPipeline:
    """Get or create the global pipeline singleton."""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = FashionPipeline(device=device)
    return _global_pipeline


# ──────────────────────────────────────────────
# Quick test when run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Dtype: {DTYPE}")
    print("Pipeline module loaded successfully.")
    print("Run `python app.py` to start the Streamlit UI.")
