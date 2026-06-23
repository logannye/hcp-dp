//! Generic height-compressed DP engine.
//!
//! The engine is intentionally conservative: it builds composable interval
//! summaries, computes the final frontier by applying those summaries, and
//! reconstructs by recursively selecting endpoint-constrained split boundaries.

use crate::blocks::BlockSummary;
use crate::traits::{HcpProblem, SummaryApply};
use crate::utils::default_block_size;
use std::time::Instant;

/// Timing and size statistics from one engine run.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HcpRunStats {
    /// Milliseconds spent building summaries, applying them, and building the summary tree.
    pub summary_build_ms: f64,
    /// Milliseconds spent recursively reconstructing the path.
    pub reconstruction_ms: f64,
}

/// Height-compressed DP engine for one problem instance.
pub struct HcpEngine<P: HcpProblem> {
    problem: P,
    block_size: usize,
}

impl<P: HcpProblem> HcpEngine<P> {
    /// Create an engine with a square-root block-size heuristic.
    pub fn new(problem: P) -> Self {
        let t = problem.num_layers().max(1);
        Self::with_block_size(problem, default_block_size(t))
    }

    /// Create an engine in the memory-minimal linear-space traceback profile.
    ///
    /// This uses one-layer leaves (`block_size = 1`). For lightweight interval
    /// summaries, this retains `O(T + F + L)` state for `T` layers, frontier
    /// width `F`, and output path length `L`, while still reconstructing an
    /// exact path.
    pub fn linear_space(problem: P) -> Self {
        Self::with_block_size(problem, 1)
    }

    /// Create an engine with an explicit block size.
    ///
    /// # Panics
    /// Panics when `block_size == 0`.
    pub fn with_block_size(problem: P, block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be positive");
        Self {
            problem,
            block_size,
        }
    }

    /// The underlying problem.
    pub fn problem(&self) -> &P {
        &self.problem
    }

    /// Mutable access to the underlying problem.
    pub fn problem_mut(&mut self) -> &mut P {
        &mut self.problem
    }

    /// Configured block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Run the DP and return `(optimal_cost, optimal_path)`.
    ///
    /// # Panics
    /// Panics if the problem violates endpoint reconstruction invariants.
    pub fn run(&self) -> (P::Cost, Vec<P::State>) {
        let (cost, path, _stats) = self.run_with_stats();
        (cost, path)
    }

    /// Run the DP and return `(optimal_cost, optimal_path, stats)`.
    ///
    /// # Panics
    /// Panics if the problem violates endpoint reconstruction invariants.
    pub fn run_with_stats(&self) -> (P::Cost, Vec<P::State>, HcpRunStats) {
        #[cfg(feature = "tracing")]
        let span = tracing::info_span!("hcp_run");
        #[cfg(feature = "tracing")]
        let _enter = span.enter();

        let build_start = Instant::now();
        let BuildArtifacts {
            blocks,
            tree,
            frontier_t,
        } = self.build_block_summaries();
        let summary_build_ms = build_start.elapsed().as_secs_f64() * 1000.0;

        let beta_0 = self.problem.initial_boundary();
        let beta_t = self.problem.terminal_boundary(&frontier_t);
        let reconstruct_start = Instant::now();
        let path = if blocks.is_empty() {
            self.problem.reconstruct_leaf(0, 0, &beta_0, &beta_t)
        } else {
            self.reconstruct_path_on_blocks(&blocks, &tree, 0, blocks.len(), &beta_0, &beta_t)
        };
        let reconstruction_ms = reconstruct_start.elapsed().as_secs_f64() * 1000.0;
        let cost = self.problem.extract_cost(&frontier_t, &beta_t);
        (
            cost,
            path,
            HcpRunStats {
                summary_build_ms,
                reconstruction_ms,
            },
        )
    }

    fn build_block_summaries(&self) -> BuildArtifacts<P> {
        #[cfg(feature = "tracing")]
        let span = tracing::trace_span!("build_block_summaries");
        #[cfg(feature = "tracing")]
        let _enter = span.enter();

        let t = self.problem.num_layers();
        let b = self.block_size;
        let num_blocks = if t == 0 { 0 } else { t.div_ceil(b) };

        let mut blocks = Vec::with_capacity(num_blocks);
        let mut frontier = self.problem.init_frontier();

        for block_idx in 0..num_blocks {
            let start = block_idx * b;
            let end = ((block_idx + 1) * b).min(t);
            let summary = self.problem.summarize_interval(start, end);
            frontier = summary.apply(&frontier);
            blocks.push(BlockSummary {
                start,
                end,
                summary,
            });
        }

        let tree = SummaryTree::new(&self.problem, &blocks);
        BuildArtifacts {
            blocks,
            tree,
            frontier_t: frontier,
        }
    }

    fn reconstruct_path_on_blocks(
        &self,
        blocks: &[BlockSummary<P::Summary>],
        tree: &SummaryTree<P>,
        start_idx: usize,
        end_idx: usize,
        beta_a: &P::Boundary,
        beta_c: &P::Boundary,
    ) -> Vec<P::State> {
        match end_idx.saturating_sub(start_idx) {
            0 => Vec::new(),
            1 => {
                let block = &blocks[start_idx];
                self.problem
                    .reconstruct_leaf(block.start, block.end, beta_a, beta_c)
            }
            _ => {
                let mid_idx = start_idx + (end_idx - start_idx) / 2;
                let a = blocks[start_idx].start;
                let m = blocks[mid_idx - 1].end;
                let c = blocks[end_idx - 1].end;

                let sigma_left = tree
                    .query(&self.problem, start_idx, mid_idx)
                    .expect("left interval must be non-empty");
                let sigma_right = tree
                    .query(&self.problem, mid_idx, end_idx)
                    .expect("right interval must be non-empty");

                let beta_m =
                    self.problem
                        .choose_split(a, m, c, beta_a, beta_c, &sigma_left, &sigma_right);

                let left = self
                    .reconstruct_path_on_blocks(blocks, tree, start_idx, mid_idx, beta_a, &beta_m);
                let right = self
                    .reconstruct_path_on_blocks(blocks, tree, mid_idx, end_idx, &beta_m, beta_c);
                concatenate_segments(left, right)
            }
        }
    }
}

fn concatenate_segments<S: Clone + PartialEq>(mut left: Vec<S>, right: Vec<S>) -> Vec<S> {
    if left.is_empty() {
        return right;
    }
    if right.is_empty() {
        return left;
    }
    assert!(
        left.last() == right.first(),
        "reconstructed path segments must join at the split boundary"
    );
    left.extend_from_slice(&right[1..]);
    left
}

#[derive(Debug)]
struct SummaryTreeNode<S> {
    start: usize,
    end: usize,
    left: Option<usize>,
    right: Option<usize>,
    summary: S,
}

struct SummaryTree<P: HcpProblem> {
    nodes: Vec<SummaryTreeNode<P::Summary>>,
    root: Option<usize>,
}

struct BuildArtifacts<P: HcpProblem> {
    blocks: Vec<BlockSummary<P::Summary>>,
    tree: SummaryTree<P>,
    frontier_t: P::Frontier,
}

impl<P: HcpProblem> SummaryTree<P> {
    fn new(problem: &P, blocks: &[BlockSummary<P::Summary>]) -> Self {
        let mut tree = Self {
            nodes: Vec::with_capacity(blocks.len().saturating_mul(2).max(1)),
            root: None,
        };
        if !blocks.is_empty() {
            tree.root = Some(tree.build(problem, blocks, 0, blocks.len()));
        }
        tree
    }

    fn build(
        &mut self,
        problem: &P,
        blocks: &[BlockSummary<P::Summary>],
        start: usize,
        end: usize,
    ) -> usize {
        if start + 1 == end {
            let node_idx = self.nodes.len();
            self.nodes.push(SummaryTreeNode {
                start,
                end,
                left: None,
                right: None,
                summary: blocks[start].summary.clone(),
            });
            return node_idx;
        }

        let mid = start + (end - start) / 2;
        let left_idx = self.build(problem, blocks, start, mid);
        let right_idx = self.build(problem, blocks, mid, end);
        let summary = problem.merge_summary(
            &self.nodes[left_idx].summary,
            &self.nodes[right_idx].summary,
        );

        let node_idx = self.nodes.len();
        self.nodes.push(SummaryTreeNode {
            start,
            end,
            left: Some(left_idx),
            right: Some(right_idx),
            summary,
        });
        node_idx
    }

    fn query(&self, problem: &P, range_start: usize, range_end: usize) -> Option<P::Summary> {
        self.query_node(problem, self.root?, range_start, range_end)
    }

    fn query_node(
        &self,
        problem: &P,
        node_idx: usize,
        range_start: usize,
        range_end: usize,
    ) -> Option<P::Summary> {
        let node = &self.nodes[node_idx];
        if range_end <= node.start || node.end <= range_start {
            return None;
        }
        if range_start <= node.start && node.end <= range_end {
            return Some(node.summary.clone());
        }

        let left = node
            .left
            .and_then(|idx| self.query_node(problem, idx, range_start, range_end));
        let right = node
            .right
            .and_then(|idx| self.query_node(problem, idx, range_start, range_end));

        match (left, right) {
            (Some(l), Some(r)) => Some(problem.merge_summary(&l, &r)),
            (Some(summary), None) | (None, Some(summary)) => Some(summary),
            (None, None) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct DummyProblem {
        layers: usize,
    }

    #[derive(Clone)]
    struct DummyFrontier(usize);

    #[derive(Clone)]
    struct DummySummary(usize);

    impl SummaryApply<DummyFrontier> for DummySummary {
        fn apply(&self, frontier: &DummyFrontier) -> DummyFrontier {
            DummyFrontier(frontier.0 + self.0)
        }
    }

    impl HcpProblem for DummyProblem {
        type State = usize;
        type Frontier = DummyFrontier;
        type Summary = DummySummary;
        type Boundary = usize;
        type Cost = usize;

        fn num_layers(&self) -> usize {
            self.layers
        }

        fn init_frontier(&self) -> Self::Frontier {
            DummyFrontier(0)
        }

        fn forward_step(&self, _layer: usize, frontier: &Self::Frontier) -> Self::Frontier {
            DummyFrontier(frontier.0 + 1)
        }

        fn summarize_interval(&self, a: usize, b: usize) -> Self::Summary {
            DummySummary(b - a)
        }

        fn merge_summary(&self, left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
            DummySummary(left.0 + right.0)
        }

        fn initial_boundary(&self) -> Self::Boundary {
            0
        }

        fn terminal_boundary(&self, frontier_t: &Self::Frontier) -> Self::Boundary {
            frontier_t.0
        }

        fn choose_split(
            &self,
            _a: usize,
            m: usize,
            _c: usize,
            _beta_a: &Self::Boundary,
            _beta_c: &Self::Boundary,
            _sigma_left: &Self::Summary,
            _sigma_right: &Self::Summary,
        ) -> Self::Boundary {
            m
        }

        fn reconstruct_leaf(
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
            frontier_t: &Self::Frontier,
            _beta_t: &Self::Boundary,
        ) -> Self::Cost {
            frontier_t.0
        }
    }

    #[test]
    fn run_concatenates_without_duplicate_midpoint() {
        let engine = HcpEngine::with_block_size(DummyProblem { layers: 4 }, 1);
        let (cost, path) = engine.run();
        assert_eq!(cost, 4);
        assert_eq!(path, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn linear_space_constructor_uses_one_layer_blocks() {
        let engine = HcpEngine::linear_space(DummyProblem { layers: 4 });
        assert_eq!(engine.block_size(), 1);
        let (cost, path) = engine.run();
        assert_eq!(cost, 4);
        assert_eq!(path, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn zero_layers_still_reconstructs_leaf() {
        let engine = HcpEngine::new(DummyProblem { layers: 0 });
        let (cost, path) = engine.run();
        assert_eq!(cost, 0);
        assert_eq!(path, vec![0]);
    }

    #[test]
    #[should_panic(expected = "block_size must be positive")]
    fn with_block_size_panics_on_zero() {
        let _ = HcpEngine::with_block_size(DummyProblem { layers: 1 }, 0);
    }
}
