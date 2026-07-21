use alloc::vec::Vec;
use core::ptr;
#[doc = " Manual KV block manager that allows downstream users to explicitly manage"]
#[doc = " and deduplicate KV cache blocks across batches."]
#[repr(C, align(64))]
pub struct SharedPrefixManager {
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
    #[doc = " Inserts a block into the manager, returning a unique block_id."]
    pub fn insert_block(&mut self, data: Vec<i32>) -> usize {
        if let Some(pos) = self.blocks.iter().position(|b| b.is_none()) {
            self.blocks[pos] = Some(data);
            pos
        } else {
            let pos = self.blocks.len();
            self.blocks.push(Some(data));
            pos
        }
    }
    #[doc = " Explicitly insert data at a given block_id."]
    #[doc = " Useful if downstream wants to manage IDs themselves."]
    pub fn insert_block_at(&mut self, block_id: usize, data: Vec<i32>) {
        if block_id >= self.blocks.len() {
            self.blocks.resize(block_id + 1, None);
        }
        self.blocks[block_id] = Some(data);
    }
    #[doc = " Returns a raw pointer to the block data, suitable for `engine.set_kv_block`."]
    pub fn get_block_ptr(&self, block_id: usize) -> *const i32 {
        if let Some(Some(block)) = self.blocks.get(block_id) {
            block.as_ptr()
        } else {
            ptr::null()
        }
    }
    #[doc = " Removes a block from the manager."]
    pub fn remove_block(&mut self, block_id: usize) {
        if block_id < self.blocks.len() {
            self.blocks[block_id] = None;
        }
    }
}
