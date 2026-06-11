#include <metal_stdlib>
using namespace metal;

struct vec101_block {
    ulong4 w_pos_bits;
    ulong4 w_neg_bits;
};

struct Vec101SuperBlock {
    half scales[8];
    short offsets[8];
    char _padding[32];
    vec101_block blocks[8];
};

kernel void vec101_gemv(
    device const Vec101SuperBlock* w_stream [[buffer(0)]],
    device const vec101_block* x_stream [[buffer(1)]],
    device const float* s_stream [[buffer(2)]],
    device float* out_buffer [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant float& x_scale [[buffer(5)]],
    constant uint& num_rows [[buffer(6)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint row = pos.x;
    uint batch = pos.y;
    
    device const Vec101SuperBlock* row_w_stream = w_stream + (row * blocks_per_row);
    device const vec101_block* batch_x_stream = x_stream + (batch * blocks_per_row * 8);
    
    float row_sum = 0.0f;
    
    for (uint col = 0; col < blocks_per_row; col++) {
        Vec101SuperBlock w_super = row_w_stream[col];
        
        for (uint sub_blk = 0; sub_blk < 8; sub_blk++) {
            float micro_scale = (float)w_super.scales[sub_blk];
            vec101_block w_blk = w_super.blocks[sub_blk];
            vec101_block x_blk = batch_x_stream[col * 8 + sub_blk];
            
            ulong4 pos_prod = (x_blk.w_pos_bits & w_blk.w_pos_bits) | (x_blk.w_neg_bits & w_blk.w_neg_bits);
            ulong4 neg_prod = (x_blk.w_pos_bits & w_blk.w_neg_bits) | (x_blk.w_neg_bits & w_blk.w_pos_bits);
            
            ulong4 p_counts = popcount(pos_prod);
            ulong4 n_counts = popcount(neg_prod);
            
            int sum_p = (int)(p_counts.x + p_counts.y + p_counts.z + p_counts.w);
            int sum_n = (int)(n_counts.x + n_counts.y + n_counts.z + n_counts.w);
            
            row_sum += (float)(sum_p - sum_n) * micro_scale;
        }
    }
    
    float scale = s_stream[row];
    out_buffer[batch * num_rows + row] = row_sum * scale * x_scale;
}
