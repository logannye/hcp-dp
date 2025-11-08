//! Block-level data structures used by the engine.
//!
//! Each `BlockSummary` represents an interval [start, end) of layers and
//! its associated summary Σ[start,end].

/// Summary for a contiguous interval of layers [start, end).
#[derive(Debug, Clone)]
pub struct BlockSummary<S> {
    /// Inclusive start layer index a.
    pub start: usize,
    /// Exclusive end layer index b; the block covers layers [start, end).
    pub end: usize,
    /// Interval summary Σ[start, end].
    pub summary: S,
}

impl<S> BlockSummary<S> {
    /// Length of the block in layers.
    #[inline]
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Returns true if the block is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::BlockSummary;

    #[test]
    fn zero_length_block_is_empty() {
        let b = BlockSummary {
            start: 5,
            end: 5,
            summary: (),
        };
        assert_eq!(b.len(), 0);
        assert!(b.is_empty());
    }

    #[test]
    fn typical_block_len() {
        let b = BlockSummary {
            start: 2,
            end: 7,
            summary: "x",
        };
        assert_eq!(b.len(), 5);
        assert!(!b.is_empty());
    }
}
