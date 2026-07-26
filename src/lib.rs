#![no_std]

#[cfg(feature = "std")]
extern crate std;

#[macro_use]
extern crate alloc;

#[macro_use]
extern crate covopt_macro;

pub mod compute;
pub mod core;
pub mod hal;
pub mod sync;

pub mod util;
pub mod compress;

pub use util::components::{attention, tokenizer};
pub use util::conv::{im2col, pack_conv_weights, conv2d_compute, Conv2dParams, Im2ColParams};

pub use compute::vec101_compute;
pub use core::{vec101_block, vec101_context};
pub use no_std_tool::debug::{ScopedResource, check_memory_leaks, check_thread_drops};
pub use util::ffi::vec101_compute_c;
