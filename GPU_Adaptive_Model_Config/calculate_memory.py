from transformers import AutoModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
path = "OpenGVLab/InternVL2_5-2B-MPO"  ##Change the model as per your requirements
model = AutoModel.from_pretrained(path)
result = estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)
print(result)