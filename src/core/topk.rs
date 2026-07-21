#[doc = " Zero-allocation Top-K reducer."]
#[doc = " Designed for high-performance, no_std map-reduce tasks (e.g. Rayon parallelism)."]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct TopK<const K: usize> {
    pub items: [Option<(usize, i32)>; K],
}
impl<const K: usize> Default for TopK<K> {
    fn default() -> Self {
        Self::new()
    }
}
impl<const K: usize> TopK<K> {
    pub fn new() -> Self {
        Self { items: [None; K] }
    }
    #[doc = " Add a new item to the TopK structure, maintaining the top K elements."]
    pub fn insert(mut self, item: (usize, i32)) -> Self {
        let score = item.1;
        let mut insert_idx = None;
        for i in 0..K {
            match self.items[i] {
                Some(existing) => {
                    if score > existing.1 {
                        insert_idx = Some(i);
                        break;
                    }
                }
                None => {
                    insert_idx = Some(i);
                    break;
                }
            }
        }
        if let Some(idx) = insert_idx {
            for j in (idx + 1..K).rev() {
                self.items[j] = self.items[j - 1];
            }
            self.items[idx] = Some(item);
        }
        self
    }
    #[doc = " Merge another TopK structure into this one."]
    pub fn merge(self, other: Self) -> Self {
        let mut merged = self;
        for i in other.items.into_iter().flatten() {
            merged = merged.insert(i);
        }
        merged
    }
}
