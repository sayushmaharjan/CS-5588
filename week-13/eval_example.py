import os
import numpy as np
from PIL import Image
from evaluation import evaluate_single, _simplified_ssim

def main():
    # 1. Provide the paths to your reference image and the newly generated image
    reference_path = "/Users/sayush/Documents/cs5588/CS-5588/week-13/test_images/m2.jpg"
    generated_path = "/Users/sayush/Documents/cs5588/CS-5588/week-13/output_images/recon/output_inpaint_man_old.png"

    # You can paste the exact prompt that was printed out during run_example.py
    # This is needed to calculate the CLIP Score (Prompt Alignment)
    prompt_used = "A full-body photo of a person wearing an elegant tailored navy suit with a white dress shirt, silk tie, and polished oxford shoes, suitable for a wedding, in formal style, with navy blue and gold accents, highly detailed, realistic lighting, professional fashion photography, 8k uhd, high resolution, sharp focus"

    if not os.path.exists(reference_path):
        print(f"Error: {reference_path} not found.")
        return
    if not os.path.exists(generated_path):
        print(f"Error: {generated_path} not found.")
        return

    print("Loading images...")
    ref_image = Image.open(reference_path).convert("RGB")
    gen_image = Image.open(generated_path).convert("RGB")

    print("\nRunning Evaluation...")
    # 2. Run the evaluation
    results = evaluate_single(
        image=gen_image,
        prompt=prompt_used,
        reference_image=ref_image,
        is_naive=False
    )

    # 3. Calculate SSIM directly
    size = (256, 256)
    ref_resized = ref_image.resize(size, Image.LANCZOS).convert("RGB")
    gen_resized = gen_image.resize(size, Image.LANCZOS).convert("RGB")
    
    ref_arr = np.array(ref_resized, dtype=np.float32)
    gen_arr = np.array(gen_resized, dtype=np.float32)
    ssim_val = _simplified_ssim(ref_arr, gen_arr)

    # 4. Print out the metrics
    print(f"\n--- EVALUATION RESULTS ---")
    print(f"CLIP Score (Prompt Alignment):    {results.clip_score:.4f}  (> 0.55 is good)")
    print(f"Identity Preservation Score:      {results.identity_score:.4f}  (> 0.20 is good)")
    print(f"Visual Quality Score:             {results.quality_score:.4f}  (> 0.30 is good)")
    print(f"SSIM Score (Structural Sim):      {ssim_val:.4f}  (> 0.30 is good)")
    
    # We can also use parts of the evaluation pipeline manually for multiple images
    from evaluation import compute_consistency, compute_diversity
    # If you had generated 2 outputs: output_variation_1.png and output_variation_2.png
    # You would load both into a list like this:
    # list_of_gens = [gen_image, Image.open("output_variation_2.png").convert("RGB")]
    # print(f"Consistency: {compute_consistency(list_of_gens):.4f}")
    # print(f"Diversity:   {compute_diversity(list_of_gens):.4f}")

if __name__ == "__main__":
    main()
