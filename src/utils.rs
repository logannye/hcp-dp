//! Assorted utilities and helpers.
//!
//! These are intentionally minimal; you can extend or replace them as needed.

/// Compute an integer square root-style block size for T layers.
///
/// This is a heuristic used by [`crate::engine::HcpEngine::new`]
/// to pick a balanced height-compression regime.
#[inline]
pub fn default_block_size(num_layers: usize) -> usize {
    if num_layers <= 1 {
        1
    } else {
        (num_layers as f64).sqrt().ceil() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::default_block_size;

    #[test]
    fn returns_one_for_small_t() {
        assert_eq!(default_block_size(0), 1);
        assert_eq!(default_block_size(1), 1);
    }

    #[test]
    fn sqrt_ceiling_behavior() {
        assert_eq!(default_block_size(2), 2);
        assert_eq!(default_block_size(3), 2);
        assert_eq!(default_block_size(4), 2);
        assert_eq!(default_block_size(5), 3);
        assert_eq!(default_block_size(100), 10);
        assert_eq!(default_block_size(101), 11);
        assert_eq!(default_block_size(10_000), 100);
        assert_eq!(default_block_size(10_001), 101);
    }

    #[test]
    fn monotonic_non_decreasing() {
        let mut prev = 0;
        for t in 0..500 {
            let b = default_block_size(t);
            assert!(b >= prev, "block size decreased at t={t}: {b} < {prev}");
            prev = b;
        }
    }
}
