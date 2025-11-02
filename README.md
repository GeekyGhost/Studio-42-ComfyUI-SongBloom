# ComfyUI Nodes for SongBloom

Generate complete songs up to 4 minutes long using AI! SongBloom uses an interleaved paradigm of autoregressive sketching and diffusion-based refinement.

## Available Models

| Model | Max Length | Features |
|-------|-----------|----------|
| songbloom_full_150s | 2m30s | Base model |
| songbloom_full_150s_dpo | 2m30s | DPO-optimized model |
| **songbloom_full_240s** | **4m** | **Newest model with extended generation** |

**Important Note**: For the 240s model, each lyric token ([intro], [outro], [inst], etc.) corresponds to 5 seconds of audio, compared to 1 second for 150s models.

Compatible tags:
[verse] [chorus] [bridge] [intro] [outro] [inst] [silence]

![image](https://github.com/user-attachments/assets/b2de9e04-edcf-45ca-a76c-4f51fc151a12)

## Installation
1. Clone the repository to your custom_nodes folder:
  ```
  git clone https://github.com/fredconex/ComfyUI-SongBloom.git
  cd ComfyUI-SongBloom
  ```
2. Install dependencies(Use the correct path for your comfyui portable):
  ```
  C:\ai_stuff\ComfyUI_windows_portable\python_embeded\python.exe -m pip install -r requirements.txt
  ```
OR if you want to use Chinese!
  ```
  C:\ai_stuff\ComfyUI_windows_portable\python_embeded\python.exe -m pip install -r requirements_chinese.txt
  ```

# Model Checkpoints

Download model checkpoints to "ComfyUI/models/checkpoints".

**150s models (up to 2m30s):**
- Base model: https://huggingface.co/fredconex/SongBloom-safetensors/tree/main
- DPO model: https://huggingface.co/fredconex/SongBloom-Safetensors/blob/main/songbloom_full_150s_dpo.safetensors

**240s model (up to 4m) - NEWEST:**
- Download from: https://huggingface.co/CypressYang/SongBloom_long

https://github.com/user-attachments/assets/feffbebf-2cd6-4ab9-8ca6-ea77164ccca7


## Credits

Original Repo:
- https://github.com/tencent-ailab/SongBloom (Official Tencent AI Lab)
- https://github.com/Cypress-Yang/SongBloom (Maintained fork with models)
