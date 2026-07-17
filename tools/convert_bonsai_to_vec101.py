import argparse
import os
import torch
import numpy as np
from safetensors.torch import safe_open, save_file

def pack_bonsai_u32_to_vec101(tensor):
    """
    Takes a U32 tensor (packed 16 weights per uint32) and converts to vec101 dual-rail format.
    Returns w_pos_np (int64), w_neg_np (int64).
    """
    out_features, in_features_packed = tensor.shape
    in_features = in_features_packed * 16

    packed = tensor.cpu().numpy().astype(np.uint32)
    
    # We unpack 16 values per uint32. 
    unpacked = np.zeros((out_features, in_features), dtype=np.int8)
    for i in range(16):
        val = ((packed >> (2 * i)) & 0b11).astype(np.int8)
        mapped = np.where(val == 2, -1, val)
        mapped = np.where(mapped == 3, -1, mapped)
        unpacked[:, i::16] = mapped

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

def compile_bonsai(input_path, output_path):
    print(f"Compiling Bonsai from {input_path}...")
    tensors_dict = {}
    with safe_open(input_path, framework="pt") as f:
        keys = f.keys()
        for k in keys:
            t = f.get_tensor(k)
            # Bonsai_27B packs 1.58-bit weights into torch.uint32
            if t.dtype == torch.uint32 and t.dim() == 2:
                print(f"Compiling layer: {k} | Shape: {t.shape}")
                w_pos, w_neg = pack_bonsai_u32_to_vec101(t)
                tensors_dict[f"{k}.w_pos_stream"] = torch.from_numpy(w_pos)
                tensors_dict[f"{k}.w_neg_stream"] = torch.from_numpy(w_neg)
            else:
                # Keep original (like scales or biases)
                tensors_dict[k] = t
                
    save_file(tensors_dict, output_path)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Bonsai to vec101 safetensors")
    parser.add_argument("--input", type=str, required=True, help="Input safetensors path")
    parser.add_argument("--output", type=str, required=True, help="Output safetensors path")
    
    args = parser.parse_args()
    compile_bonsai(args.input, args.output)
