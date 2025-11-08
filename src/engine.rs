//! Generic height-compressed DP engine.
//!
//! This module implements the two-phase algorithm:
//! 1. A forward pass that builds block-level summaries.
//! 2. A recursive reconstruction that uses these summaries to recover
//!    an exact optimal path with reduced memory.
//!
//! The engine is completely generic over implementations of [`HcpProblem`].

use crate::blocks::BlockSummary;
use crate::traits::HcpProblem;
use crate::utils::default_block_size;

/// Height-compressed DP engine for a given problem instance `P`.
///
/// Typical usage:
/// ```
/// use hcp_dp::{HcpEngine, problems::lcs::LcsProblem};
///
/// let s = b"ACCGGTCGAGTGCGCGGAAGCCGGCCGAA";
/// let t = b"GTCGTTCGGAATGCCGTTGCTCTGTAAA";
/// let problem = LcsProblem::new(s, t);
/// let engine = HcpEngine::new(problem);
/// let (cost, path) = engine.run();
/// println!("LCS length: {}", cost);
/// println!("Path length: {}", path.len());
/// ```
pub struct HcpEngine<P: HcpProblem> {
    problem: P,
    block_size: usize,
}

impl<P: HcpProblem> HcpEngine<P> {
    /// Create a new engine with a heuristic block size (≈ √T).
    pub fn new(problem: P) -> Self {
        let t = problem.num_layers().max(1);
        let b = default_block_size(t).max(1);
        Self {
            problem,
            block_size: b,
        }
    }

    /// Create a new engine with an explicit block size.
    ///
    /// # Panics
    /// Panics if `block_size == 0`.
    pub fn with_block_size(problem: P, block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be positive");
        Self {
            problem,
            block_size,
        }
    }

    /// Expose immutable reference to the underlying problem.
    pub fn problem(&self) -> &P {
        &self.problem
    }

    /// Expose mutable reference if callers need to adjust configuration.
    pub fn problem_mut(&mut self) -> &mut P {
        &mut self.problem
    }

    /// Return the configured block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Run the full height-compressed DP:
    ///
    /// - Executes the forward block summarization.
    /// - Chooses an optimal terminal boundary.
    /// - Recursively reconstructs an optimal path.
    ///
    /// Returns `(optimal_cost, optimal_path_states)`.
    pub fn run(&self) -> (P::Cost, Vec<P::State>) {
        #[cfg(feature = "tracing")]
        let span = tracing::info_span!("hcp_run");
        #[cfg(feature = "tracing")]
        let _enter = span.enter();

        let (blocks, frontier_t) = {
            #[cfg(feature = "tracing")]
            let span = tracing::info_span!("build_block_summaries");
            #[cfg(feature = "tracing")]
            let _enter = span.enter();
            self.build_block_summaries()
        };

        let beta_0 = self.problem.initial_boundary();
        let beta_t = self.problem.terminal_boundary(&frontier_t);

        let path = {
            #[cfg(feature = "tracing")]
            let span = tracing::info_span!("reconstruct");
            #[cfg(feature = "tracing")]
            let _enter = span.enter();
            self.reconstruct_path_on_blocks(&blocks, &beta_0, &beta_t)
        };
        let cost = self.problem.extract_cost(&frontier_t, &beta_t);

        (cost, path)
    }

    /// Phase I: build summaries for each block along the layer dimension.
    ///
    /// Returns:
    /// - the list of `BlockSummary` items covering [0, T), and
    /// - the final frontier at layer T.
    fn build_block_summaries(&self) -> (Vec<BlockSummary<P::Summary>>, P::Frontier) {
        let t = self.problem.num_layers();
        let b = self.block_size;
        let num_blocks = if t == 0 { 0 } else { t.div_ceil(b) };

        let mut blocks = Vec::with_capacity(num_blocks);
        let mut frontier = self.problem.init_frontier();

        for k in 0..num_blocks {
            let start = k * b;
            let end = ((k + 1) * b).min(t);
            let (frontier_end, sigma) = self.problem.summarize_block(start, end, &frontier);
            blocks.push(BlockSummary {
                start,
                end,
                summary: sigma,
            });
            frontier = frontier_end;
        }

        (blocks, frontier)
    }

    /// Phase II: recursively reconstruct an optimal path across blocks.
    ///
    /// This operates purely on block-level metadata plus problem callbacks.
    fn reconstruct_path_on_blocks(
        &self,
        blocks: &[BlockSummary<P::Summary>],
        beta_a: &P::Boundary,
        beta_c: &P::Boundary,
    ) -> Vec<P::State> {
        match blocks.len() {
            0 => Vec::new(),
            1 => {
                let b = &blocks[0];
                self.problem
                    .reconstruct_block(b.start, b.end, beta_a, beta_c)
            }
            _ => {
                // Split range into left/right halves.
                let mid = blocks.len() / 2;
                let (left, right) = blocks.split_at(mid);

                let a = left.first().unwrap().start;
                let m = left.last().unwrap().end;
                let c = right.last().unwrap().end;

                // Merge summaries within left and right.
                #[cfg(feature = "parallel")]
                let (sigma_left, sigma_right) = {
                    // Requires P: Sync at call sites when feature is enabled.
                    let (l, r) = rayon::join(
                        || merge_chain_parallel(&self.problem, left),
                        || merge_chain_parallel(&self.problem, right),
                    );
                    (l, r)
                };
                #[cfg(not(feature = "parallel"))]
                let (sigma_left, sigma_right) = {
                    (
                        merge_chain(&self.problem, left),
                        merge_chain(&self.problem, right),
                    )
                };

                // Select boundary at layer m consistent with optimality.
                let beta_m = self.problem.choose_boundary(
                    a,
                    m,
                    c,
                    &sigma_left,
                    &sigma_right,
                    beta_a,
                    beta_c,
                );

                // Recurse on both halves.
                let path_left = self.reconstruct_path_on_blocks(left, beta_a, &beta_m);
                let path_right = self.reconstruct_path_on_blocks(right, &beta_m, beta_c);

                // Concatenate, omit boundary duplication at m (by contract segments meet at m).
                if path_left.is_empty() {
                    return path_right;
                }
                if path_right.is_empty() {
                    return path_left;
                }

                let mut out = path_left;
                // Drop first element of right only if it duplicates the last of left.
                let offset = if out.last() == path_right.first() {
                    1
                } else {
                    0
                };
                out.extend_from_slice(&path_right[offset..]);
                out
            }
        }
    }
}

/// Helper: merge a chain of block summaries Σ[start,end] via problem's monoid.
fn merge_chain<P: HcpProblem>(problem: &P, blocks: &[BlockSummary<P::Summary>]) -> P::Summary {
    assert!(!blocks.is_empty(), "cannot merge empty block chain");
    let mut acc = blocks[0].summary.clone();
    for blk in &blocks[1..] {
        acc = problem.merge_summary(&acc, &blk.summary);
    }
    acc
}

#[cfg(feature = "parallel")]
fn merge_chain_parallel<P>(problem: &P, blocks: &[BlockSummary<P::Summary>]) -> P::Summary
where
    P: HcpProblem + Sync,
    P::Summary: Send,
{
    use rayon::prelude::*;
    assert!(!blocks.is_empty(), "cannot merge empty block chain");
    // Reduce pairwise across summaries to leverage associativity.
    blocks.par_iter().map(|b| b.summary.clone()).reduce(
        || blocks[0].summary.clone(),
        |a, b| problem.merge_summary(&a, &b),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct DummyProblem {
        t: usize,
    }

    #[derive(Clone)]
    struct F(u32);
    #[derive(Clone)]
    #[allow(dead_code)]
    struct S(usize);
    #[derive(Clone)]
    #[allow(dead_code)]
    struct B(usize);

    impl HcpProblem for DummyProblem {
        type State = usize;
        type Frontier = F;
        type Summary = S;
        type Boundary = B;
        type Cost = i32;

        fn num_layers(&self) -> usize {
            self.t
        }
        fn init_frontier(&self) -> Self::Frontier {
            F(0)
        }
        fn forward_step(&self, _layer: usize, f: &Self::Frontier) -> Self::Frontier {
            F(f.0 + 1)
        }
        fn summarize_block(
            &self,
            a: usize,
            b: usize,
            frontier_a: &Self::Frontier,
        ) -> (Self::Frontier, Self::Summary) {
            let mut f = frontier_a.clone();
            for layer in a..b {
                f = self.forward_step(layer, &f);
            }
            (f.clone(), S(b))
        }
        fn merge_summary(&self, _left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
            right.clone()
        }
        fn initial_boundary(&self) -> Self::Boundary {
            B(0)
        }
        fn terminal_boundary(&self, _frontier_t: &Self::Frontier) -> Self::Boundary {
            B(self.t)
        }
        fn choose_boundary(
            &self,
            a: usize,
            m: usize,
            c: usize,
            _sigma_left: &Self::Summary,
            _sigma_right: &Self::Summary,
            _beta_a: &Self::Boundary,
            _beta_c: &Self::Boundary,
        ) -> Self::Boundary {
            let _ = (a, c);
            B(m)
        }
        fn reconstruct_block(
            &self,
            a: usize,
            b: usize,
            _beta_a: &Self::Boundary,
            _beta_b: &Self::Boundary,
        ) -> Vec<Self::State> {
            (a..=b).collect()
        }
        fn extract_cost(
            &self,
            _frontier_t: &Self::Frontier,
            _beta_t: &Self::Boundary,
        ) -> Self::Cost {
            0
        }
    }

    #[test]
    fn build_block_summaries_shapes() {
        let problem = DummyProblem { t: 4 };
        let engine = HcpEngine::with_block_size(problem, 2);
        let (blocks, _f) = engine.build_block_summaries();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].start, 0);
        assert_eq!(blocks[0].end, 2);
        assert_eq!(blocks[1].start, 2);
        assert_eq!(blocks[1].end, 4);
    }

    #[test]
    fn run_concatenates_without_duplicate_midpoint() {
        let problem = DummyProblem { t: 4 };
        let engine = HcpEngine::with_block_size(problem, 2);
        let (_cost, path) = engine.run();
        // Expect 5 states from 0..=4
        assert_eq!(path, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    #[should_panic]
    fn with_block_size_panics_on_zero() {
        let _ = HcpEngine::with_block_size(DummyProblem { t: 2 }, 0);
    }
}
