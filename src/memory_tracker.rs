use core::sync::atomic::{AtomicU32, Ordering};

// Global counters for memory leaks and thread drop checks using AtomicU32
static RESOURCE_COUNT: AtomicU32 = AtomicU32::new(0);
static THREAD_ACTIVE_COUNT: AtomicU32 = AtomicU32::new(0);

/// A scoped resource tracker to ensure everything is dropped correctly.
/// Since we operate in a `no_alloc` environment, this helps track lifetimes
/// of important structures across threads.
pub struct ScopedResource;

impl ScopedResource {
    #[inline]
    pub fn new() -> Self {
        RESOURCE_COUNT.fetch_add(1, Ordering::SeqCst);
        THREAD_ACTIVE_COUNT.fetch_add(1, Ordering::SeqCst);
        Self
    }
}

impl Default for ScopedResource {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ScopedResource {
    #[inline]
    fn drop(&mut self) {
        RESOURCE_COUNT.fetch_sub(1, Ordering::SeqCst);
        THREAD_ACTIVE_COUNT.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Checks if there are any memory leaks (resources not dropped).
/// Returns true if there are no leaks.
#[inline]
pub fn check_memory_leaks() -> bool {
    RESOURCE_COUNT.load(Ordering::SeqCst) == 0
}

/// Checks if all threads/operations have correctly dropped.
/// Returns true if all are dropped.
#[inline]
pub fn check_thread_drops() -> bool {
    THREAD_ACTIVE_COUNT.load(Ordering::SeqCst) == 0
}
