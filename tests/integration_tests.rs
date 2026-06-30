mod common;

mod it {
    #[cfg(not(loom))]
    mod correctness;
    #[cfg(loom)]
    mod loom_test;
}
