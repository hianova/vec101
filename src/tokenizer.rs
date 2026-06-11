use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

#[derive(Default)]
struct TrieNode {
    children: BTreeMap<u8, TrieNode>,
    token_id: Option<u32>,
}

pub struct TrieTokenizer {
    root: TrieNode,
    id_to_token: BTreeMap<u32, Vec<u8>>,
    pub vocab_size: u32,
    pub unk_token: u32,
}

impl TrieTokenizer {
    pub fn new(unk_token: u32) -> Self {
        Self {
            root: TrieNode::default(),
            id_to_token: BTreeMap::new(),
            vocab_size: 0,
            unk_token,
        }
    }

    /// Add a string/byte-sequence to the tokenizer vocabulary.
    pub fn add_token(&mut self, text: &[u8], id: u32) {
        let mut curr = &mut self.root;
        for &b in text {
            curr = curr.children.entry(b).or_default();
        }
        curr.token_id = Some(id);
        self.id_to_token.insert(id, text.to_vec());
        if id >= self.vocab_size {
            self.vocab_size = id + 1;
        }
    }

    /// Encodes a string into token IDs using greedy longest-prefix matching.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut tokens = Vec::new();
        let mut i = 0;

        while i < bytes.len() {
            let mut curr = &self.root;
            let mut best_match = None;
            let mut best_len = 0;

            for (j, _byte) in bytes.iter().enumerate().skip(i) {
                if let Some(next_node) = curr.children.get(&bytes[j]) {
                    curr = next_node;
                    if let Some(id) = curr.token_id {
                        best_match = Some(id);
                        best_len = j - i + 1;
                    }
                } else {
                    break;
                }
            }

            if let Some(id) = best_match {
                tokens.push(id);
                i += best_len;
            } else {
                // If no match, emit unknown token and advance by 1 byte
                tokens.push(self.unk_token);
                i += 1;
            }
        }
        tokens
    }

    /// Decodes a sequence of token IDs back into a string.
    /// Invalid UTF-8 bytes are replaced with the replacement character.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in tokens {
            if let Some(token_bytes) = self.id_to_token.get(&id) {
                bytes.extend_from_slice(token_bytes);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_trie_tokenizer() {
        let mut tokenizer = TrieTokenizer::new(0);
        tokenizer.add_token(b"Hello", 1);
        tokenizer.add_token(b"World", 2);
        tokenizer.add_token(b" ", 3);
        
        let encoded = tokenizer.encode("Hello World");
        assert_eq!(encoded, vec![1, 3, 2]);
        
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, "Hello World");
        
        // Test unknown token
        let encoded_unk = tokenizer.encode("XYZ");
        assert_eq!(encoded_unk, vec![0, 0, 0]); // 3 bytes, 3 unk tokens
    }
}
