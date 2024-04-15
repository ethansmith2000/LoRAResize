# LoRARedo

LoRARedo is a simple tool that allows you to further compress or expand a trained LoRA.
Expansion is useful if you've trained a LoRA then realized you'd like to try a larger rank but don't want to start training over
Compression, I believe speaks for itself.

There are two methods employed here, one uses a least squares optimization to estimate your equivalent lora at a different dimension.
Alternatively, the simple_lora_expand function will just add rows/columns of zeros, however I'm not sure how this will affect training initializing the matrix with zeros, so I would proceed with caution.

```python
from lora_redo import change_lora_rank, simple_expand_lora
import torch

state_dict = torch.load("path/to/your/lora.pth")
new_state_dict = change_lora_rank(state_dict, new_rank=10)
```