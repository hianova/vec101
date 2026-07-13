pub use no_std_tool::sync::AtomicMailboxU32;

#[cfg(feature = "std")]
pub mod std_impl;
#[cfg(feature = "std")]
pub use std_impl::*;

#[cfg(not(feature = "std"))]
pub mod no_std;
#[cfg(not(feature = "std"))]
pub use no_std::*;
