use alloc::vec::Vec;
use core::ptr;

/// Manual KV block manager that allows downstream users to explicitly manage
/// and deduplicate KV cache blocks across batches.
pub struct SharedPrefixManager {
    // We store the actual block data here.
    // The index in the outer Vec serves as the 'block_id'.
    // We use a Vec of Vecs to ensure the pointers to inner Vec's data remain stable.
    blocks: Vec<Option<Vec<i32>>>,
}

impl Default for SharedPrefixManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedPrefixManager {
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Inserts a block into the manager, returning a unique block_id.
    pub fn insert_block(&mut self, data: Vec<i32>) -> usize {
        // Find an empty slot
        if let Some(pos) = self.blocks.iter().position(|b| b.is_none()) {
            self.blocks[pos] = Some(data);
            pos
        } else {
            let pos = self.blocks.len();
            self.blocks.push(Some(data));
            pos
        }
    }

    /// Explicitly insert data at a given block_id.
    /// Useful if downstream wants to manage IDs themselves.
    pub fn insert_block_at(&mut self, block_id: usize, data: Vec<i32>) {
        if block_id >= self.blocks.len() {
            self.blocks.resize(block_id + 1, None);
        }
        self.blocks[block_id] = Some(data);
    }

    /// Returns a raw pointer to the block data, suitable for `engine.set_kv_block`.
    pub fn get_block_ptr(&self, block_id: usize) -> *const i32 {
        if let Some(Some(block)) = self.blocks.get(block_id) {
            block.as_ptr()
        } else {
            ptr::null()
        }
    }

    /// Removes a block from the manager.
    pub fn remove_block(&mut self, block_id: usize) {
        if block_id < self.blocks.len() {
            self.blocks[block_id] = None;
        }
    }
}
