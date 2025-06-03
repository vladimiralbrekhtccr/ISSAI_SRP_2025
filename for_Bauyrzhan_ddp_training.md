Node = 1 PC with GPUs


So first you check weather you can connect from 1 PC to another one, using ssh internal IP adress.
Then you run bash script on master node
On second node you change the NODE_rank from 0 to 1 and run same bash script.

And this way you will start DDP training on multi-node.

Guide from pytorch.
https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html

You can test with this model training. It will require some time but it's 1B model so you should be able to run it easily on 2 Nodes Lora training.
https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_full.sh


2 those files on each node.
```bash
#!/bin/bash
# Simple Multi-Node DDP Training
# Usage: ./train.sh [node_rank] [master_ip]

NODE_RANK=${1:-0}           # 0 for master, 1 for worker
MASTER_IP=${2:-"192.168.1.100"}
MASTER_PORT=29500

torchrun \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_IP \
    --master_port=$MASTER_PORT \
    --nproc_per_node=8 \
    train.py
```

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize distributed training
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# Your model
model = YourModel()
model = model.cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

# Your training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass, backward pass, optimizer step
        pass

dist.destroy_process_group()
```
