import os
import paddle

from safetensors.numpy import save_file
from paddlenlp.transformers import YulanForCausalLM, Qwen2ForCausalLM, LlamaForCausalLM, AutoConfig


root_folder = "/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_Base_12B-Alpaca_4096-lr_2e5-bs_device_1-acc_1-gpu_16-warmup_10-src_1024-tgt_1024/checkpoint-96-st"
pdparams_path = os.path.join(root_folder, 'model_state.pdparams')
safetensor_path = os.path.join(root_folder, 'model.safetensors')

state_dict = paddle.load(pdparams_path)
tensors = {}
for item in state_dict:
    print(item, state_dict[item].shape)
    tensors[item] = state_dict[item].numpy()

save_file(tensors, safetensor_path, metadata={"format": "np"})

os.remove(pdparams_path)