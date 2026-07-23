pub mod llm_traits;
pub use llm_traits::*;
pub mod builder;
pub mod engine;
pub mod kv_cache;
pub mod sensor;
pub use builder::ComputeContextBuilder;
pub use engine::Vec101Engine;
pub use kv_cache::SharedPrefixManager;
pub use sensor::ContinuousInput;

pub mod borrow_engine;
pub mod sequence_evaluator;

pub use borrow_engine::Vec101EngineBorrow;
pub use sequence_evaluator::LayerSequenceEvaluator;
pub mod math;
pub mod topk;
pub use no_std_tool::vec101_compute::types::{vec101_block, Vec101SuperBlock, QuantType, vec101_context};
#[doc = " The Heterogeneous Compute Hub interface (Dual Engine)"]
#[repr(C, align(64))]
pub struct DualEngineContext<'a> {
    #[doc = " 1. 唯一的一份物理記憶體映射 (Zero-copy)"]
    pub shared_kv_cache: &'a mut [u8],
    pub shared_weights_1_58b: &'a [u8],
    #[doc = " 2. 你發明的無鎖信箱 (Lock-free Mailbox)"]
    pub auto_fill_mailbox: crate::sync::AtomicMailboxU32,
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_core_structs_debug() {
        let block1 = vec101_block {
            w_pos_bits: [0; 4],
            w_neg_bits: [0; 4],
        };
        let _ = alloc::format!("{:?}", block1);
        let sb = Vec101SuperBlock {
            scales: [0; 8],
            offsets: [0; 8],
            _padding: [0; 32],
            blocks: [block1; 8],
        };
        let _ = alloc::format!("{:?}", sb);
        let _ = alloc::format!("{:?}", QuantType::Bit1_58);
    }
    #[test]
    fn test_mailbox_default() {
        let _ = crate::sync::AtomicMailboxU32::default();
    }
}
