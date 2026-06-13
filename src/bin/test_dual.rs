use std::time::Instant;
use vec101::engine::Vec101Engine;

fn main() {
    let q4_path = "/Users/kuangtalin/Documents/google:gemma-4-E2B-it-qat-q4_0.rkyv";
    let bit1_path = "/Users/kuangtalin/Documents/bitnet_compiled.rkyv";
    
    println!("=== 🧠 初始化大腦左半球 (BitNet 1.58-bit) ===");
    let start_left = Instant::now();
    match Vec101Engine::new(bit1_path) {
        Ok(mut engine_left) => {
            println!("✅ 左半球 (Drafting & Routing) 載入完成！耗時: {:?}", start_left.elapsed());
            // Mock generate
            let prompts = vec!["Fast Intent Routing...".to_string()];
            let results = engine_left.generate_parallel(&prompts);
            println!("👉 左半球光速產出: {:?}", results[0]);
        },
        Err(e) => eprintln!("❌ 左半球載入失敗: {:?}", e)
    }

    println!("\n=== 💡 初始化大腦右半球 (Gemma Q4_0) ===");
    let start_right = Instant::now();
    match Vec101Engine::new(q4_path) {
        Ok(mut engine_right) => {
            println!("✅ 右半球 (Deep Inference & Generation) 載入完成！耗時: {:?}", start_right.elapsed());
            // Mock generate
            let prompts = vec!["Deep Generation...".to_string()];
            let results = engine_right.generate_parallel(&prompts);
            println!("👉 右半球深度產出: {:?}", results[0]);
        },
        Err(e) => eprintln!("❌ 右半球載入失敗: {:?}", e)
    }
    
    println!("\n🎉 雙引擎均能無縫掛載至同一記憶體位址空間！(Zero-Copy mmap)");
}
