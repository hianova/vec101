use std::io::Write;
use std::time::Instant;
use vec101::ops::{attention, rope, rmsnorm_int8, swiglu_int8};
use vec101::tokenizer::TrieTokenizer;

struct MockModel {
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
    layers: usize,
}

impl MockModel {
    fn new(hidden_dim: usize, num_heads: usize, layers: usize) -> Self {
        Self {
            hidden_dim,
            num_heads,
            head_dim: hidden_dim / num_heads,
            layers,
        }
    }

    fn forward(
        &self,
        tokens: &[u32],
        start_pos: usize,
        k_cache: &mut [Vec<f32>],
        v_cache: &mut [Vec<f32>],
        logits_out: &mut [f32],
        vocab_size: usize,
    ) {
        let num_tokens = tokens.len();
        // The main dataflow is now strictly INT8!
        let mut x_i8 = vec![1i8; num_tokens * self.hidden_dim];
        let mut x_scales = vec![1.0f32; num_tokens];

        // Temporary buffers
        let mut q = vec![0.0f32; num_tokens * self.hidden_dim];
        let mut k = vec![0.0f32; num_tokens * self.hidden_dim];
        let mut v = vec![0.0f32; num_tokens * self.hidden_dim];
        let mut attn_out = vec![0.0f32; num_tokens * self.hidden_dim];
        
        // Output buffers for fused INT8 ops
        let mut norm_i8 = vec![0i8; num_tokens * self.hidden_dim];
        let mut norm_scales = vec![0.0f32; num_tokens];
        let mut ffn_i8 = vec![0i8; num_tokens * self.hidden_dim];
        let mut ffn_scales = vec![0.0f32; num_tokens];
        
        let rms_weight = vec![1.0f32; self.hidden_dim];
        let ffn_weight = vec![1.0f32; self.hidden_dim];

        for _layer in 0..self.layers {
            // 1. Fused RMSNorm: i8 -> i8
            rmsnorm_int8(&x_i8, &rms_weight, 1e-5, &mut norm_i8, &mut norm_scales);

            // 2. QKV Projection (simulated as float conversion for Attention)
            for t in 0..num_tokens {
                let s = norm_scales[t];
                let offset = t * self.hidden_dim;
                for i in 0..self.hidden_dim {
                    let val = (norm_i8[offset + i] as f32) * s;
                    q[offset + i] = val;
                    k[offset + i] = val;
                    v[offset + i] = val;
                }
            }

            // 3. Attention (FP32 precision retained for RoPE/Softmax accuracy)
            rope(&mut q, &mut k, start_pos, self.hidden_dim, self.head_dim, 10000.0);

            for t in 0..num_tokens {
                k_cache[start_pos + t].copy_from_slice(&k[t * self.hidden_dim..(t + 1) * self.hidden_dim]);
                v_cache[start_pos + t].copy_from_slice(&v[t * self.hidden_dim..(t + 1) * self.hidden_dim]);
            }

            attention(&q, k_cache, v_cache, start_pos, self.num_heads, self.head_dim, &mut attn_out);

            // Residual connection and re-quantize to keep x in INT8
            for t in 0..num_tokens {
                let s_x = x_scales[t];
                let offset = t * self.hidden_dim;
                let mut max_abs = 0.0f32;
                let mut temp_f32 = vec![0.0; self.hidden_dim];
                
                for i in 0..self.hidden_dim {
                    let val = (x_i8[offset + i] as f32) * s_x + attn_out[offset + i];
                    temp_f32[i] = val;
                    let abs = libm::fabsf(val);
                    if abs > max_abs { max_abs = abs; }
                }
                
                let s_new = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
                let inv_s_new = 1.0 / s_new;
                
                for i in 0..self.hidden_dim {
                    let mut quantized = libm::roundf(temp_f32[i] * inv_s_new) as i32;
                    if quantized > 127 { quantized = 127; }
                    if quantized < -128 { quantized = -128; }
                    x_i8[offset + i] = quantized as i8;
                }
                x_scales[t] = s_new;
            }

            // 4. Fused SwiGLU: i8 -> i8 (bypassing persistent f32 allocation)
            swiglu_int8(&x_i8, &x_scales, &ffn_weight, &mut ffn_i8, &mut ffn_scales);

            // Residual connection 2
            for t in 0..num_tokens {
                let s_x = x_scales[t];
                let s_f = ffn_scales[t];
                let offset = t * self.hidden_dim;
                let mut max_abs = 0.0f32;
                let mut temp_f32 = vec![0.0; self.hidden_dim];
                
                for i in 0..self.hidden_dim {
                    let val = (x_i8[offset + i] as f32) * s_x + (ffn_i8[offset + i] as f32) * s_f;
                    temp_f32[i] = val;
                    let abs = libm::fabsf(val);
                    if abs > max_abs { max_abs = abs; }
                }
                
                let s_new = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
                let inv_s_new = 1.0 / s_new;
                
                for i in 0..self.hidden_dim {
                    let mut quantized = libm::roundf(temp_f32[i] * inv_s_new) as i32;
                    if quantized > 127 { quantized = 127; }
                    if quantized < -128 { quantized = -128; }
                    x_i8[offset + i] = quantized as i8;
                }
                x_scales[t] = s_new;
            }
        }

        for t in 0..num_tokens {
            let offset = t * vocab_size;
            let target_logit_idx = (tokens[t] as usize + 1) % vocab_size;
            for i in 0..vocab_size {
                logits_out[offset + i] = if i == target_logit_idx { 10.0 } else { 0.1 };
            }
        }
    }
}

fn sample_argmax(logits: &[f32]) -> u32 {
    let mut max_val = f32::NEG_INFINITY;
    let mut best_idx = 0;
    for (i, &val) in logits.iter().enumerate() {
        if val > max_val {
            max_val = val;
            best_idx = i;
        }
    }
    best_idx as u32
}

fn main() {
    println!("Initializing Gemma 4 MTP + INT8 Operator Fusion LLM Shell...");

    let mut tokenizer = TrieTokenizer::new(0);
    tokenizer.vocab_size = 262144;
    tokenizer.add_token(b"Hello", 100);
    tokenizer.add_token(b" world", 101);
    tokenizer.add_token(b"!", 102);
    for i in 0..256 {
        tokenizer.add_token(&[i as u8], i as u32 + 1000);
    }

    let target_model = MockModel::new(4096, 32, 32);
    let draft_model = MockModel::new(1024, 16, 8);

    let max_seq_len = 2048;
    let mut target_k_cache = vec![vec![0.0; target_model.hidden_dim]; max_seq_len];
    let mut target_v_cache = vec![vec![0.0; target_model.hidden_dim]; max_seq_len];
    let mut draft_k_cache = vec![vec![0.0; draft_model.hidden_dim]; max_seq_len];
    let mut draft_v_cache = vec![vec![0.0; draft_model.hidden_dim]; max_seq_len];

    let mut current_pos = 0;
    let prompt = "Hello";
    let accepted_tokens = tokenizer.encode(prompt);
    
    print!("Prompt: ");
    for &tok in &accepted_tokens {
        print!("{}", tokenizer.decode(&[tok]));
    }
    println!("\nGenerating...");

    let prefill_start = Instant::now();
    let mut target_logits = vec![0.0; accepted_tokens.len() * tokenizer.vocab_size as usize];
    let mut draft_logits = vec![0.0; accepted_tokens.len() * tokenizer.vocab_size as usize];

    target_model.forward(&accepted_tokens, 0, &mut target_k_cache, &mut target_v_cache, &mut target_logits, tokenizer.vocab_size as usize);
    draft_model.forward(&accepted_tokens, 0, &mut draft_k_cache, &mut draft_v_cache, &mut draft_logits, tokenizer.vocab_size as usize);
    
    current_pos += accepted_tokens.len();
    let mut last_accepted_token = sample_argmax(&target_logits[(accepted_tokens.len()-1) * tokenizer.vocab_size as usize..]);
    
    let ttft = prefill_start.elapsed();
    println!("\n\n[Metrics] TTFT (Prefill {} tokens): {:?}", accepted_tokens.len(), ttft);
    
    print!("> {}", tokenizer.decode(&[last_accepted_token]));
    std::io::stdout().flush().unwrap();

    let num_draft_tokens = 3;
    let decode_start = Instant::now();
    let mut generated_tokens = 1;

    for _step in 0..10 {
        let mut drafted_tokens = Vec::new();
        let mut draft_pos = current_pos;
        let mut draft_input = last_accepted_token;

        for _ in 0..num_draft_tokens {
            let mut logits = vec![0.0; tokenizer.vocab_size as usize];
            draft_model.forward(&[draft_input], draft_pos, &mut draft_k_cache, &mut draft_v_cache, &mut logits, tokenizer.vocab_size as usize);
            let next_draft_tok = sample_argmax(&logits);
            drafted_tokens.push(next_draft_tok);
            draft_input = next_draft_tok;
            draft_pos += 1;
        }

        let mut verification_input = vec![last_accepted_token];
        verification_input.extend_from_slice(&drafted_tokens);
        
        let mut target_logits = vec![0.0; verification_input.len() * tokenizer.vocab_size as usize];
        target_model.forward(&verification_input, current_pos, &mut target_k_cache, &mut target_v_cache, &mut target_logits, tokenizer.vocab_size as usize);

        let mut n_accepted = 0;
        for i in 0..num_draft_tokens {
            let target_offset = i * tokenizer.vocab_size as usize;
            let target_pred = sample_argmax(&target_logits[target_offset..target_offset + tokenizer.vocab_size as usize]);
            
            if target_pred == drafted_tokens[i] {
                n_accepted += 1;
                print!("{}", tokenizer.decode(&[target_pred]));
                std::io::stdout().flush().unwrap();
            } else {
                break;
            }
        }

        let correction_offset = n_accepted * tokenizer.vocab_size as usize;
        let correction_tok = sample_argmax(&target_logits[correction_offset..correction_offset + tokenizer.vocab_size as usize]);
        print!("{}", tokenizer.decode(&[correction_tok]));
        std::io::stdout().flush().unwrap();

        last_accepted_token = correction_tok;
        current_pos += n_accepted + 1;
        generated_tokens += n_accepted + 1;
    }
    
    let decode_time = decode_start.elapsed();
    let tps = generated_tokens as f64 / decode_time.as_secs_f64();
    
    println!("\n\n[Metrics] Decode Time: {:?} for {} tokens", decode_time, generated_tokens);
    println!("[Metrics] Tokens per Second (TPS): {:.2} tok/s", tps);
    println!("Generation complete.");
}
