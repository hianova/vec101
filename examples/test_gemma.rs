use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let mut file = File::open("/Users/kuangtalin/Documents/gemma-4-12b-it-qat-q4_0.gguf")?;
    let content = gguf_file::Content::read(&mut file)?;
    for (name, _) in content.tensor_infos.iter() {
        println!("{}", name);
        break;
    }
    println!("GGUF loaded successfully.");
    Ok(())
}
