#[cfg(loom)]
pub mod loom_impl;
#[cfg(loom)]
pub use loom_impl::*;

#[cfg(all(feature = "std", not(loom)))]
pub mod std_impl;
#[cfg(all(feature = "std", not(loom)))]
pub use std_impl::*;

#[cfg(not(any(feature = "std", loom)))]
pub mod no_std;
#[cfg(not(any(feature = "std", loom)))]
pub use no_std::*;
