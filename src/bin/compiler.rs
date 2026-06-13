use safetensors::SafeTensors;
use std::fs;
use vec101::types::{ModelWeights, LayerWeights, Vec101SuperBlock, vec101_block, f32_to_f16};
use rkyv::ser::{serializers::AllocSerializer, Serializer};
use std::collections::HashMap;

#[inline(always)]
fn quantize_q4_0(weights: &[f32], output: &mut [vec101::types::BlockQ4_0]) {
    for (i, block) in output.iter_mut().enumerate() {
        let block_w = &weights[i * 32..(i + 1) * 32];
        let mut amax = 0.0f32;
        for &w in block_w {
            if w.abs() > amax { amax = w.abs(); }
        }
        let d = amax / 7.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        block.d = f32_to_f16(d);
        
        for j in 0..16 {
            let w0 = block_w[j * 2] * id;
            let w1 = block_w[j * 2 + 1] * id;
            
            let q0 = (w0.round() as i32 + 8).clamp(0, 15) as u8;
            let q1 = (w1.round() as i32 + 8).clamp(0, 15) as u8;
            
            block.qs[j] = q0 | (q1 << 4);
        }
    }
}

fn main() {
    let input_path = std::env::args().nth(1).unwrap_or("../tools/bitnet_b1_58-large/bitnet_compiled.safetensors".to_string());
    let output_path = std::env::args().nth(2).unwrap_or("../tools/bitnet_b1_58-large/bitnet_compiled.rkyv".to_string());
    
    println!("Reading {}...", input_path);
    let data = fs::read(&input_path).expect("Failed to read safetensors file");
    let st = SafeTensors::deserialize(&data).expect("Failed to deserialize safetensors");
    
    let mut layer_map: HashMap<String, (Vec<u64>, Vec<u64>, Vec<f32>, u32)> = HashMap::new();
    let mut q4_layer_map: HashMap<String, Vec<f32>> = HashMap::new();
    
    for (tensor_name, tensor) in st.tensors() {
        let parts: Vec<&str> = tensor_name.rsplitn(2, '.').collect();
        if parts.len() != 2 { 
            // Possibly a simple weight tensor without suffix
            let data_bytes = tensor.data();
            // Try to load as raw f16 and convert to f32 (simplified, assuming fp32 for now)
            // Need actual dtype check, assuming fp32 for demonstration
            continue; 
        }
        let suffix = parts[0];
        let layer_name = parts[1].to_string();
        
        let entry = layer_map.entry(layer_name.clone()).or_insert((Vec::new(), Vec::new(), Vec::new(), 0));
        let data_bytes = tensor.data();
        
        match suffix {
            "w_pos_stream" => {
                let ptr = data_bytes.as_ptr() as *const u64;
                let len = data_bytes.len() / 8;
                entry.0 = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
            },
            "w_neg_stream" => {
                let ptr = data_bytes.as_ptr() as *const u64;
                let len = data_bytes.len() / 8;
                entry.1 = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
            },
            "s_stream" => {
                let ptr = data_bytes.as_ptr() as *const f32;
                let len = data_bytes.len() / 4;
                entry.2 = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
            },
            "num_blocks" => {
                let ptr = data_bytes.as_ptr() as *const i32;
                let len = data_bytes.len() / 4;
                let num_blocks_arr = unsafe { std::slice::from_raw_parts(ptr, len) };
                entry.3 = num_blocks_arr[0] as u32;
            },
            "weight" => {
                // Raw QAT unquantized weights
                let ptr = data_bytes.as_ptr() as *const f32; // Assuming fp32 for now
                let len = data_bytes.len() / 4;
                let weights = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
                q4_layer_map.insert(layer_name.clone(), weights);
            },
            _ => {}
        }
    }
    
    let mut model_weights = ModelWeights { layers: Vec::new() };
    
    for (layer_name, (w_pos, w_neg, s, _num_blocks)) in layer_map {
        let mut super_blocks = Vec::new();
        let num_super_blocks = w_pos.len() / 32; // Each SuperBlock has 8 blocks of 4 u64s = 32 u64s
        
        let mut pos_idx = 0;
        let mut neg_idx = 0;
        let mut s_idx = 0;
        
        for _ in 0..num_super_blocks {
            let mut sb = Vec101SuperBlock {
                scales: [0; 8],
                offsets: [0; 8],
                _padding: [0; 32],
                blocks: [vec101_block { w_pos_bits: [0; 4], w_neg_bits: [0; 4] }; 8],
            };
            
            for i in 0..8 {
                let current_s = if s_idx < s.len() { s[s_idx] } else { 1.0 };
                sb.scales[i] = f32_to_f16(current_s);
                s_idx += 1;
                
                for j in 0..4 {
                    if pos_idx < w_pos.len() {
                        sb.blocks[i].w_pos_bits[j] = w_pos[pos_idx];
                        pos_idx += 1;
                    }
                }
                for j in 0..4 {
                    if neg_idx < w_neg.len() {
                        sb.blocks[i].w_neg_bits[j] = w_neg[neg_idx];
                        neg_idx += 1;
                    }
                }
            }
            super_blocks.push(sb);
        }
        
        model_weights.layers.push(LayerWeights {
            name: layer_name,
            data: vec101::types::LayerData::Bit1_58(super_blocks),
        });
    }

    for (layer_name, weights) in q4_layer_map {
        let num_blocks = weights.len() / 32;
        let mut q4_blocks = vec![vec101::types::BlockQ4_0 { d: 0, qs: [0; 16] }; num_blocks];
        quantize_q4_0(&weights, &mut q4_blocks);
        
        model_weights.layers.push(LayerWeights {
            name: layer_name,
            data: vec101::types::LayerData::Q4_0(q4_blocks),
        });
    }
    
    println!("Serializing {} layers to rkyv format...", model_weights.layers.len());
    let mut serializer = AllocSerializer::<4096>::default();
    serializer.serialize_value(&model_weights).expect("Failed to serialize");
    let bytes = serializer.into_serializer().into_inner();
    fs::write(&output_path, bytes).expect("Failed to save rkyv file");
    println!("Successfully saved zero-copy model to {}", output_path);
}
