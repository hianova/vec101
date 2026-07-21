use vec101::{conv2d_compute, im2col, pack_conv_weights};

fn main() {
    println!("=== vec101 1.58-bit Convolution Demo ===");

    // 1. Setup Image (Batch=1, C=3, H=4, W=4)
    let batch = 1;
    let in_channels = 3;
    let in_h = 4;
    let in_w = 4;

    let mut image = vec![0i8; batch * in_channels * in_h * in_w];
    for i in 0..image.len() {
        image[i] = (i as i8 % 5) - 2; // dummy values around -2 to 2
    }

    // 2. Im2Col Transformation (Kernel=3x3, Stride=1, Pad=1)
    let k_h = 3;
    let k_w = 3;
    let stride = 1;
    let pad = 1;

    let out_h = (in_h + 2 * pad - k_h) / stride + 1;
    let out_w = (in_w + 2 * pad - k_w) / stride + 1;

    let inner_dim = in_channels * k_h * k_w;
    let padded_inner_dim = inner_dim.div_ceil(2048) * 2048;

    let mut col_matrix = vec![0i8; batch * out_h * out_w * padded_inner_dim];

    im2col(
        &image,
        batch,
        in_channels,
        in_h,
        in_w,
        k_h,
        k_w,
        stride,
        pad,
        padded_inner_dim,
        &mut col_matrix,
    );

    let num_patches = out_h * out_w;
    println!(
        "Im2Col output shape: [{} patches, {} dimensions]",
        num_patches, inner_dim
    );

    // 3. Setup Weights (OutChannels=8)
    let out_channels = 8;
    let mut weights = vec![0i32; out_channels * inner_dim];
    for i in 0..weights.len() {
        weights[i] = if i % 3 == 0 {
            1
        } else if i % 3 == 1 {
            -1
        } else {
            0
        };
    }

    // 4. Pack Weights into vec101 SuperBlocks
    let blocks = pack_conv_weights(&weights, out_channels, in_channels, k_h, k_w);
    println!("Packed into {} vec101_blocks", blocks.len());

    // 5. Run vec101 Convolution Compute (GEMM)
    let mut out_buffer = vec![0i32; batch * out_channels * out_h * out_w];
    let s_stream = vec![1i32; batch]; // mock scaling factors

    conv2d_compute(
        &image,
        &blocks,
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        k_h,
        k_w,
        stride,
        pad,
        &s_stream,
        &mut col_matrix,
        &mut out_buffer,
    );

    println!(
        "Convolution output shape: [Batch={}, OutChannels={}, H={}, W={}]",
        batch, out_channels, out_h, out_w
    );

    println!("First few output values:");
    for i in 0..10.min(out_buffer.len()) {
        print!("{} ", out_buffer[i]);
    }
    println!("\n=== Demo Finished successfully ===");
}
