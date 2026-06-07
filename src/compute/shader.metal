#include <metal_stdlib>
using namespace metal;

struct vec101_block {
    ulong4 w_pos_bits;
    ulong4 w_neg_bits;
};

kernel void vec101_gemv(
    device const vec101_block* w_stream [[buffer(0)]],
    device const vec101_block* x_stream [[buffer(1)]],
    device const float* s_stream [[buffer(2)]],
    device float* out_buffer [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant float& x_scale [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    uint row = id;
    device const vec101_block* row_w_stream = w_stream + (row * blocks_per_row);
    
    int row_sum = 0;
    
    for (uint i = 0; i < blocks_per_row; i++) {
        vec101_block w_blk = row_w_stream[i];
        vec101_block x_blk = x_stream[i];
        
        // Block contains ulong4 (4x uint64_t)
        // We can use bitwise operations on the vector types!
        ulong4 pos_prod = (x_blk.w_pos_bits & w_blk.w_pos_bits) | (x_blk.w_neg_bits & w_blk.w_neg_bits);
        ulong4 neg_prod = (x_blk.w_pos_bits & w_blk.w_neg_bits) | (x_blk.w_neg_bits & w_blk.w_pos_bits);
        
        // popcount each element in ulong4
        // popcount natively works on scalar integers. For vectors, we extract or use native vector popcount.
        // MSL popcount supports vector types: popcount(ulong4) returns int4
        int4 p_counts = popcount(pos_prod);
        int4 n_counts = popcount(neg_prod);
        
        // horizontally sum the counts
        int sum_p = p_counts.x + p_counts.y + p_counts.z + p_counts.w;
        int sum_n = n_counts.x + n_counts.y + n_counts.z + n_counts.w;
        
        row_sum += (sum_p - sum_n);
    }
    
    // Final scaling
    float w_scale = s_stream[row];
    out_buffer[row] = (float)(row_sum) * w_scale * x_scale;
}
