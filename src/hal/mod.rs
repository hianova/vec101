use crate::types::vec101_context;

pub mod cpu;

#[cfg(feature = "gpu-metal")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

/// 統一的硬體抽象特徵 (Hardware Abstraction Layer)
pub trait Vec101Backend {
    /// 核心的計算進入點
    fn compute(&self, ctx: &vec101_context);
}
