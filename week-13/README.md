**Goal:** This hands-on challenge focuses on building a data-driven, controlled image generation system using Stable Diffusion. 

**Students will:**
- design structured prompts from data 
- apply control mechanisms 
- generate and compare outputs 
- evaluate performance using defined metrics 
- effectively leverage AI tools

**Fashion Outfit Generator (Occasion-Based)**

**Concept:**
Generate outfits based on context.

**Input:**
    - text prompt with occasion (wedding, gym, interview), style (formal, streetwear, casual), color preferences + image of person

**Output:**
    - full-body fashion images of the same person in the input image wearing the generated outfit

**Focus:**
    Clothing coherence
    Occasion appropriateness

**Experiment Angle:**
    Naive: “a person wearing formal clothes”
    Structured: detailed fashion descriptors

**Technical Requirements**
Your system must include:

**Stable Diffusion Pipeline**
Use frameworks such as:
    
    diffusers
    https://github.com/huggingface/diffusers

    ControlNet
    https://github.com/lllyasviel/ControlNet

    Stable Diffusion
    https://github.com/CompVis/stable-diffusion

**Control Mechanism (Required)**
Include at least one:
- structured prompt templates
- negative prompts
- ControlNet or other conditioning
 

**Data-to-Prompt Mapping**
Convert structured input into prompts
Clearly define your prompt generation strategy

**Evaluation**
Define and apply metrics such as:
- Prompt alignment
- Consistency
- Diversity
- Visual quality

You must include:
- baseline vs improved comparison
- failure cases and analysis

**Tools & Technologies**
    - Python, PyTorch
    - Stable Diffusion (Diffusers)
    - Prompt engineering
Optional:
    - ControlNet
    - Face alignment tools
 

**Requirements**

- Generate multiple images per input
- Maintain style consistency
- Compare:
    naive vs structured promptss
- Analyze:
    identity/style preservation
    failure cases

**Outfit Dataset**
    https://huggingface.co/datasets/mvasil/polyvore-outfits 

**Reference Project**
    https://github.com/sail-sg/EditAnything (Clothes Editing)
    