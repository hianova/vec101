use memmap2::MmapOptions;
use std::fs::File;
use std::path::Path;
use rkyv::validation::validators::DefaultValidator;
use rkyv::CheckBytes;
use crate::types::ModelWeights;

pub struct ZeroCopyModelLoader {
    // Keep mmap alive to ensure the references remain valid
    _mmap: memmap2::Mmap,
    pub model_weights: &'static crate::types::ArchivedModelWeights,
}

impl ZeroCopyModelLoader {
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        // Ensure memory is aligned by checking byte validation
        let mut validator = DefaultValidator::new(&mmap);
        let archived = match rkyv::check_archived_root::<ModelWeights>(&mmap) {
            Ok(root) => root,
            Err(e) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to validate zero-copy rkyv model format: {:?}", e),
                ));
            }
        };

        // Safety: We keep `_mmap` inside this struct, ensuring it lives as long as `model_weights`.
        // By casting the reference to 'static we can return it. The actual lifetime is bound
        // to the struct itself. For a no_std friendly architecture, we use unsafe pointer cast
        // but it's safe within the `ZeroCopyModelLoader` boundary.
        let model_weights_static = unsafe {
            std::mem::transmute::<&crate::types::ArchivedModelWeights, &'static crate::types::ArchivedModelWeights>(archived)
        };

        Ok(Self {
            _mmap: mmap,
            model_weights: model_weights_static,
        })
    }
}
