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
pub mod types;
pub mod compute;
pub mod hal;

pub mod components;
pub mod feeder;
pub mod ops;
pub mod ffi;
pub mod math_int;

pub use components::{memory_tracker, attention, tokenizer};


pub use compute::vec101_compute;
pub use components::memory_tracker::{check_memory_leaks, check_thread_drops, ScopedResource};
pub use types::{vec101_block, vec101_context};
pub use ffi::vec101_compute_c;
