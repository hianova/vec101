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

pub mod types;
pub mod compute;
pub mod memory_tracker;
pub mod feeder;
pub mod traits;
pub mod engine;
pub mod attention;
pub mod ops;
pub mod ffi;
pub mod tokenizer;
#[cfg(feature = "std")]
pub mod loader;

pub use compute::vec101_compute;
pub use memory_tracker::{check_memory_leaks, check_thread_drops, ScopedResource};
pub use types::{vec101_block, vec101_context};


