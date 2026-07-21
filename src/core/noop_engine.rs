use crate::core::vec101_context;

/// A safe dummy engine that prevents crashes by skipping execution.
/// Used to replace `mem::zeroed()` patterns that would otherwise segfault.
pub struct NoopEngine {
    #[allow(dead_code)]
    pub(crate) ctx: vec101_context,
}

impl NoopEngine {
    pub(crate) fn new(ctx: vec101_context) -> Self {
        Self { ctx }
    }

    /// Performs no operation, returning safely without dereferencing invalid memory.
    pub fn compute(&mut self) {
        // Safe no-op execution. 
        // Bypasses unsafe `vec101_compute(&self.ctx)` call entirely.
    }
}
