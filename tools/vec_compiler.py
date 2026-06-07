import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

def compile_linear_layer(weight_tensor):
    """
    Compiles a linear layer (out_features, in_features) to vec101 dual-rail streams.
    Returns w_pos_stream (int64), w_neg_stream (int64), i_stream (uint32), s_stream (float32).
    """
    out_features, in_features = weight_tensor.shape
    
    # We must chunk along in_features in sizes of 256
    pad_len = (256 - (in_features % 256)) % 256
    if pad_len > 0:
        padding = torch.zeros((out_features, pad_len), dtype=weight_tensor.dtype, device=weight_tensor.device)
        weight_tensor = torch.cat([weight_tensor, padding], dim=1)
        in_features += pad_len
        
    num_blocks_per_row = in_features // 256
    
    # In BitNet b1.58, scaling is typically done row-wise (per out_feature)
    scale = weight_tensor.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
    scaled = weight_tensor / scale
    quantized = torch.round(torch.clamp(scaled, -1.0, 1.0))
    
    pos_mask = (quantized == 1.0).cpu().numpy().astype(np.uint64)
    neg_mask = (quantized == -1.0).cpu().numpy().astype(np.uint64)
    
    # Reshape masks to extract 64-bit chunks
    # (out_features, num_blocks_per_row, 4, 64) -> (-1, 64)
    pos_mask_64 = pos_mask.reshape(-1, 64)
    neg_mask_64 = neg_mask.reshape(-1, 64)
    
    powers = np.array([1 << i for i in range(64)], dtype=np.uint64)
    
    w_pos_u64 = np.sum(pos_mask_64 * powers, axis=1) # shape (-1,)
    w_neg_u64 = np.sum(neg_mask_64 * powers, axis=1) # shape (-1,)
    
    # Safetensors doesn't natively support uint64, so we cast to int64.
    w_pos_np = w_pos_u64.view(np.int64)
    w_neg_np = w_neg_u64.view(np.int64)
    
    out_indices = np.arange(out_features, dtype=np.uint32)
    i_np = np.repeat(out_indices, num_blocks_per_row)
    
    scale_np = scale.squeeze(1).cpu().numpy().astype(np.float32)
    s_np = np.repeat(scale_np, num_blocks_per_row)
    
    return w_pos_np, w_neg_np, i_np, s_np

def compile_model(model_id, output_path):
    print(f"Loading model {model_id}...")
    import os
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=torch.float16
    )
    
    tensors_dict = {}
    
    print("Iterating over model modules to find Linear layers...")
    for name, module in model.named_modules():
        # Check if it has a weight parameter that looks like a linear projection
        # BitNet models might use custom classes (e.g. BitLinear), but they usually have a 'weight' attribute.
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            if module.weight.dim() == 2:
                print(f"Compiling layer: {name} | Shape: {module.weight.shape}")
                w_pos, w_neg, i_str, s_str = compile_linear_layer(module.weight.data)
                
                # We prefix the layer name
                tensors_dict[f"{name}.w_pos_stream"] = torch.from_numpy(w_pos)
                tensors_dict[f"{name}.w_neg_stream"] = torch.from_numpy(w_neg)
                # Safetensors supports I32 but not U32 natively, so view as int32
                tensors_dict[f"{name}.i_stream"] = torch.from_numpy(i_str.view(np.int32))
                tensors_dict[f"{name}.s_stream"] = torch.from_numpy(s_str)
                tensors_dict[f"{name}.num_blocks"] = torch.tensor([len(i_str)], dtype=torch.int32)
                
    if not tensors_dict:
        print("Warning: No linear layers found to compile!")
        return
        
    print(f"Saving compiled layers to {output_path}...")
    save_file(tensors_dict, output_path)
    print("Compilation complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="1bitLLM/bitnet_b1_58-large", help="HuggingFace model ID")
    parser.add_argument("--output", type=str, default="bitnet_compiled.safetensors", help="Output safetensors path")
    args = parser.parse_args()
    
    compile_model(args.model, args.output)
