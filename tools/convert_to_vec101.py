import argparse
import os
import torch
import numpy as np
from safetensors.torch import safe_open, save_file

def pack_bitnet_u8_to_vec101(tensor):
    """
    Takes a U8 tensor (packed 4 weights per byte) and converts to dual-rail format.
    Returns w_pos_np (int64), w_neg_np (int64).
    """
    # tensor shape: [out_features, in_features_packed]
    out_features, in_features_packed = tensor.shape
    in_features = in_features_packed * 4

    # Convert to standard unpacked array
    packed = tensor.cpu().numpy().astype(np.uint8)
    
    # We unpack 4 values per byte. 
    # Usually: 0 -> 0, 1 -> 1, 2 -> -1
    # We need to extract them.
    unpacked = np.zeros((out_features, in_features), dtype=np.int8)
    for i in range(4):
        # Extract the 2 bits for the i-th weight
        val = ((packed >> (2 * i)) & 0b11).astype(np.int8)
        # Map 2 to -1, 1 to 1, 0 to 0
        mapped = np.where(val == 2, -1, val)
        # Some implementations might use 3 as -1. We handle both:
        mapped = np.where(mapped == 3, -1, mapped)
        unpacked[:, i::4] = mapped

    # Pad in_features to multiple of 256
    pad_len = (256 - (in_features % 256)) % 256
    if pad_len > 0:
        padding = np.zeros((out_features, pad_len), dtype=np.int8)
        unpacked = np.concatenate([unpacked, padding], axis=1)
        in_features += pad_len
        
    num_blocks_per_row = in_features // 256

    pos_mask = (unpacked == 1).astype(np.uint64)
    neg_mask = (unpacked == -1).astype(np.uint64)

    # Reshape to 64-bit chunks
    pos_mask_64 = pos_mask.reshape(-1, 64)
    neg_mask_64 = neg_mask.reshape(-1, 64)

    powers = np.array([1 << i for i in range(64)], dtype=np.uint64)

    w_pos_u64 = np.sum(pos_mask_64 * powers, axis=1)
    w_neg_u64 = np.sum(neg_mask_64 * powers, axis=1)

    return w_pos_u64.view(np.int64), w_neg_u64.view(np.int64)

def quantize_q4_0(tensor):
    """
    Quantizes a BF16/FP32 tensor to Q4_0 (32 weights -> 1 f16 scale + 16 bytes).
    Returns a byte tensor.
    """
    # shape [out_features, in_features]
    out_features, in_features = tensor.shape
    
    # Pad to multiple of 32
    pad_len = (32 - (in_features % 32)) % 32
    if pad_len > 0:
        padding = torch.zeros((out_features, pad_len), dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=1)
        in_features += pad_len
        
    tensor = tensor.float()
    tensor_reshaped = tensor.view(-1, 32)
    
    # Q4_0 max abs
    max_abs = tensor_reshaped.abs().max(dim=1, keepdim=True).values
    # Standard Q4_0 scale: d = max / -8
    d = max_abs / -8.0
    # Avoid division by zero
    d = torch.where(d == 0, torch.ones_like(d), d)
    
    q = torch.round(tensor_reshaped / d).clamp(-8, 7).to(torch.int8)
    
    # Pack 32 int8 values (-8..7) into 16 bytes
    # qs[i] = (q[2*i] & 0x0F) | ((q[2*i+1] & 0x0F) << 4)
    q0 = q[:, 0::2] & 0x0F
    q1 = q[:, 1::2] & 0x0F
    qs = (q0 | (q1 << 4)).to(torch.uint8)
    
    d_f16 = d.half().view(torch.uint8) # 2 bytes per block
    
    # BlockQ4_0 is: d (2 bytes) + qs (16 bytes) = 18 bytes
    block_bytes = torch.cat([d_f16, qs], dim=1)
    
    return block_bytes.contiguous()

def compile_bitnet(input_path, output_path):
    print(f"Compiling BitNet from {input_path}...")
    tensors_dict = {}
    with safe_open(input_path, framework="pt") as f:
        keys = f.keys()
        for k in keys:
            t = f.get_tensor(k)
            if t.dtype == torch.uint8 and t.dim() == 2:
                print(f"Compiling layer: {k} | Shape: {t.shape}")
                w_pos, w_neg = pack_bitnet_u8_to_vec101(t)
                tensors_dict[f"{k}.w_pos_stream"] = torch.from_numpy(w_pos)
                tensors_dict[f"{k}.w_neg_stream"] = torch.from_numpy(w_neg)
            else:
                # Keep original (like scales or layernorms)
                tensors_dict[k] = t
                
    save_file(tensors_dict, output_path)
    print(f"Saved {output_path}")

def compile_q4_0(input_path, output_path):
    print(f"Compiling Q4_0 from {input_path}...")
    tensors_dict = {}
    with safe_open(input_path, framework="pt") as f:
        keys = f.keys()
        for k in keys:
            t = f.get_tensor(k)
            if "weight" in k and t.dim() == 2:
                print(f"Quantizing layer: {k} | Shape: {t.shape}")
                q4_bytes = quantize_q4_0(t)
                tensors_dict[f"{k}.q4_0_stream"] = q4_bytes
            else:
                tensors_dict[k] = t
                
    save_file(tensors_dict, output_path)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert models to vec101 safetensors")
    parser.add_argument("--input", type=str, required=True, help="Input safetensors path")
    parser.add_argument("--output", type=str, required=True, help="Output safetensors path")
    parser.add_argument("--mode", type=str, choices=["bitnet", "q4_0"], required=True, help="Conversion mode")
    
    args = parser.parse_args()
    
    if args.mode == "bitnet":
        compile_bitnet(args.input, args.output)
    elif args.mode == "q4_0":
        compile_q4_0(args.input, args.output)
