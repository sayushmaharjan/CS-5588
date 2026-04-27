from pipeline import FashionPipeline
from PIL import Image
import os

def main():
    # 1. Initialize the pipeline
    # Make sure to set device="mps" if you are on Apple Silicon, or "cuda" for Nvidia GPUs
    print("Initializing Fashion Pipeline...")
    pipe = FashionPipeline()
    # pipe.load_models()
    
    # 2. Provide a reference image
    # Note: Replace 'reference_photo.jpg' with the path to your actual image
    image_path = "/Users/sayush/Documents/cs5588/CS-5588/week-13/test_images/m2.jpg"
    
    if not os.path.exists(image_path):
        # Create a dummy image just for demonstration if one doesn't exist
        print(f"File {image_path} not found. Creating a dummy image for testing...")
        ref_image = Image.new('RGB', (512, 768), color=(180, 150, 130))
        ref_image.save(image_path)
    
    print(f"Loading reference image from {image_path}...")
    ref_image = Image.open(image_path).convert("RGB")

    # 3. Generate the images
    print("Generating outfits...")
    # result = pipe.generate(
    #     reference_image=ref_image,
    #     occasion="wedding",
    #     style="formal",
    #     color_palette="navy blue and gold accents",
    #     num_images=1, # Reduced to 1 to save memory on Mac MPS
    #     seed=42, # Optional: fixing seed for reproducibility
    # )
    result = pipe.generate_inpaint(
        reference_image=ref_image,
        occasion="office",
        style="formal",
        color_palette="navy blue and gold accents",
        seed=42,
        strength=0.95,
        num_images=1,
    )
    result["images"][0].save("/Users/sayush/Documents/cs5588/CS-5588/week-13/output_images/recon/output_inpaint_man_old.png")
    result["mask"].save("/Users/sayush/Documents/cs5588/CS-5588/week-13/output_images/mask/mask_m3_old.png")  # verify mask looks right
    
    # 4. Save the results
    print("Generation complete! Saving images...")
    for i, img in enumerate(result["images"]):
        output_filename = f"/Users/sayush/Documents/cs5588/CS-5588/week-13/output_images/variation/output_variation_{i+1}.png"
        img.save(output_filename)
        print(f"Saved {output_filename}")
        
    print(f"Used prompt: {result['prompt']}")

if __name__ == "__main__":
    main()
