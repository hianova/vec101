#![cfg_attr(not(feature = "std"), no_std)]
#![allow(non_camel_case_types)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_mut)]
#![allow(dead_code)]
#![allow(unreachable_code)]
#![allow(unexpected_cfgs)]
#![allow(unused_imports)]

extern crate alloc;

pub mod sync;
pub mod core;
pub mod compute;
pub mod hal;
pub mod gpu;

pub mod util;

pub use util::components::{attention, tokenizer};

#[cfg(feature = "std")]
pub mod react;


pub use compute::vec101_compute;
pub use no_std_tool::debug::{check_memory_leaks, check_thread_drops, ScopedResource};
pub use core::{vec101_block, vec101_context};
pub use util::ffi::vec101_compute_c;
