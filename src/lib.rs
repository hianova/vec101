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

pub mod compute;
pub mod core;
pub mod gpu;
pub mod hal;
pub mod sync;

pub mod util;

pub use util::components::{attention, tokenizer};


pub use compute::vec101_compute;
pub use core::{vec101_block, vec101_context};
pub use no_std_tool::debug::{ScopedResource, check_memory_leaks, check_thread_drops};
pub use util::ffi::vec101_compute_c;
