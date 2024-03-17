Experiment: improve image quality with AI for image generation with Stable Diffusion.



# Appendix: How to add dffusers for CUDA 11.7
```
poetry add diffusers[torch]
poetry source add torch_cu117 --priority=explicit https://download.pytorch.org/whl/cu117
poetry add torch --source torch_cu117
poetry add transformers
```