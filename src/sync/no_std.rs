pub use alloc::sync::Arc;
pub use core::sync::atomic::{AtomicUsize, Ordering};
pub use core::hint::spin_loop;

// spawn_thread is not available in no_std
