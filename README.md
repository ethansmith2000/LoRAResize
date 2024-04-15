# LoRAResize

LoRAResize is a simple tool that allows you to further compress or expand a trained LoRA.
Expansion is useful if you've trained a LoRA then realized you'd like to try a larger rank but don't want to start training over
Compression, I believe speaks for itself.

There are a few methods employed here:
- "zero_pad": simply add rows/columns of zeros to the lora matrix, only for expansion, I'm not sure how this will affect training initializing the matrix with zeros, so I would proceed with caution.
- "svd": fuses the lora components into a full size matrix then uses the singular value decomposition to get back components, although quite speedy at normal dimensions, if using for expansion I would also proceed with caution because expanded dimensions come out to be very small numbers near zero.
- "optimization": randomly initialize the lora components of the desired rank, then uses gradient descent to optimize minimum reconstruction error with the full size matrix.
- "auto": chooses between svd and optimization methods based on device and matrix dimensions, tuned for best possible speed

A couple settings to be aware of when using the gradient optimization method:
- down_name/up_name: the naming convention used for the lora layers in your model
- steps: the number optimization steps
- start_lr/end_lr: a linearly interpolated learning rate used for the optimization
- error_threshold: if any elements of the matrix exhibit a % error above this value, retry
- allowed_num_tries: the number of times to retry the optimization before giving up
- method: read above

you may need to adjust the settings a bit to your needs

```python
from lora_resize import change_lora_rank
import torch

state_dict = torch.load("path/to/your/lora.pth")
new_state_dict = change_lora_rank(state_dict, new_rank=10)
```