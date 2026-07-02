use crate::core::vec101_context;

pub mod cpu;

/// 統一的硬體抽象特徵 (Hardware Abstraction Layer)
pub trait Vec101Backend {
    /// 核心的計算進入點
    fn compute(&self, ctx: &vec101_context);
}
