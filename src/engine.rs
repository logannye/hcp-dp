//! Generic height-compressed DP engine.
//!
//! This module implements the two-phase algorithm:
//! 1. A forward pass that builds block-level summaries.
//! 2. A recursive reconstruction that uses these summaries to recover
//!    an exact optimal path with reduced memory.
//!
//! The engine is completely generic over implementations of [`HcpProblem`].

use crate::blocks::BlockSummary;
use crate::traits::{HcpProblem, SummaryApply};
use crate::utils::default_block_size;
#[cfg(feature = "parallel")]
use rayon::join;

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
        Self::with_block_size(problem, b)
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

    /// Phase I: build summaries for each block along the layer dimension.
    ///
    /// Returns:
    /// - the list of `BlockSummary` items covering [0, T), and
    /// - the final frontier at layer T.
    fn build_block_summaries(&self) -> BuildArtifacts<P> {
        let t = self.problem.num_layers();
        let b = self.block_size;
        let num_blocks = if t == 0 { 0 } else { t.div_ceil(b) };

        let mut blocks = Vec::with_capacity(num_blocks);
        let mut frontier = self.problem.init_frontier();
        let mut frontiers = Vec::with_capacity(num_blocks + 1);
        frontiers.push(frontier.clone());

        for k in 0..num_blocks {
            let start = k * b;
            let end = ((k + 1) * b).min(t);
            #[cfg(feature = "tracing")]
            let span = tracing::trace_span!("summarize_block", block = k, start, end);
            #[cfg(feature = "tracing")]
            let _enter = span.enter();
            let (frontier_end, sigma) = self.problem.summarize_block(start, end, &frontier);
            blocks.push(BlockSummary {
                start,
                end,
                summary: sigma,
            });
            frontier = frontier_end;
            frontiers.push(frontier.clone());
        }

        let tree = SummaryTree::new(&self.problem, &blocks);

        BuildArtifacts {
            blocks,
            tree,
            frontiers,
        }
    }
}

#[cfg(all(feature = "parallel", feature = "tracing"))]
fn query_forward_pair<P>(
    problem: &P,
    tree: &SummaryTree<P>,
    start_idx: usize,
    mid_idx: usize,
    end_idx: usize,
) -> (P::Summary, P::Summary)
where
    P: HcpProblem + Sync,
    P::Summary: Send + Sync,
{
    let left_span = tracing::trace_span!(
        "summary_query_forward",
        range_start = start_idx,
        range_end = mid_idx
    );
    let right_span = tracing::trace_span!(
        "summary_query_forward",
        range_start = mid_idx,
        range_end = end_idx
    );
    join(
        {
            let left_span = left_span.clone();
            move || {
                left_span.in_scope(|| {
                    tree.query_forward(problem, start_idx, mid_idx)
                        .expect("left interval must be non-empty")
                })
            }
        },
        {
            let right_span = right_span.clone();
            move || {
                right_span.in_scope(|| {
                    tree.query_forward(problem, mid_idx, end_idx)
                        .expect("right interval must be non-empty")
                })
            }
        },
    )
}

#[cfg(all(feature = "parallel", not(feature = "tracing")))]
fn query_forward_pair<P>(
    problem: &P,
    tree: &SummaryTree<P>,
    start_idx: usize,
    mid_idx: usize,
    end_idx: usize,
) -> (P::Summary, P::Summary)
where
    P: HcpProblem + Sync,
    P::Summary: Send + Sync,
{
    join(
        || {
            tree.query_forward(problem, start_idx, mid_idx)
                .expect("left interval must be non-empty")
        },
        || {
            tree.query_forward(problem, mid_idx, end_idx)
                .expect("right interval must be non-empty")
        },
    )
}

#[cfg(all(not(feature = "parallel"), feature = "tracing"))]
fn query_forward_pair<P>(
    problem: &P,
    tree: &SummaryTree<P>,
    start_idx: usize,
    mid_idx: usize,
    end_idx: usize,
) -> (P::Summary, P::Summary)
where
    P: HcpProblem,
{
    let left_span = tracing::trace_span!(
        "summary_query_forward",
        range_start = start_idx,
        range_end = mid_idx
    );
    let sigma_left = left_span.in_scope(|| {
        tree.query_forward(problem, start_idx, mid_idx)
            .expect("left interval must be non-empty")
    });
    let right_span = tracing::trace_span!(
        "summary_query_forward",
        range_start = mid_idx,
        range_end = end_idx
    );
    let sigma_right = right_span.in_scope(|| {
        tree.query_forward(problem, mid_idx, end_idx)
            .expect("right interval must be non-empty")
    });
    (sigma_left, sigma_right)
}

#[cfg(all(not(feature = "parallel"), not(feature = "tracing")))]
fn query_forward_pair<P>(
    problem: &P,
    tree: &SummaryTree<P>,
    start_idx: usize,
    mid_idx: usize,
    end_idx: usize,
) -> (P::Summary, P::Summary)
where
    P: HcpProblem,
{
    (
        tree.query_forward(problem, start_idx, mid_idx)
            .expect("left interval must be non-empty"),
        tree.query_forward(problem, mid_idx, end_idx)
            .expect("right interval must be non-empty"),
    )
}

#[cfg(all(feature = "parallel", feature = "tracing"))]
#[allow(clippy::too_many_arguments)]
fn apply_summaries_pair<P>(
    sigma_left: &P::Summary,
    sigma_right: &P::Summary,
    frontier_a: &P::Frontier,
    frontier_c: &P::Frontier,
    a: usize,
    m: usize,
    c: usize,
    start_idx: usize,
    mid_idx: usize,
    end_idx: usize,
) -> (P::Frontier, P::Frontier)
where
    P: HcpProblem + Sync,
    P::Summary: Send + Sync,
    P::Frontier: Send + Sync,
{
    let forward_span = tracing::trace_span!(
        "summary_apply_forward",
        start = a,
        end = m,
        blocks = mid_idx - start_idx
    );
    let backward_span = tracing::trace_span!(
        "summary_apply_reverse",
        start = m,
        end = c,
        blocks = end_idx - mid_idx
    );
    join(
        {
            let forward_span = forward_span.clone();
            move || forward_span.in_scope(|| sigma_left.apply(frontier_a))
        },
        {
            let backward_span = backward_span.clone();
            move || backward_span.in_scope(|| sigma_right.apply_reverse(frontier_c))
        },
    )
}

#[cfg(all(feature = "parallel", not(feature = "tracing")))]
#[allow(clippy::too_many_arguments)]
fn apply_summaries_pair<P>(
    sigma_left: &P::Summary,
    sigma_right: &P::Summary,
    frontier_a: &P::Frontier,
    frontier_c: &P::Frontier,
    _a: usize,
    _m: usize,
    _c: usize,
    _start_idx: usize,
    _mid_idx: usize,
    _end_idx: usize,
) -> (P::Frontier, P::Frontier)
where
    P: HcpProblem + Sync,
    P::Summary: Send + Sync,
    P::Frontier: Send + Sync,
{
    join(
        || sigma_left.apply(frontier_a),
        || sigma_right.apply_reverse(frontier_c),
    )
}

#[cfg(all(not(feature = "parallel"), feature = "tracing"))]
#[allow(clippy::too_many_arguments)]
fn apply_summaries_pair<P>(
    sigma_left: &P::Summary,
    sigma_right: &P::Summary,
    frontier_a: &P::Frontier,
    frontier_c: &P::Frontier,
    a: usize,
    m: usize,
    c: usize,
    start_idx: usize,
    mid_idx: usize,
    end_idx: usize,
) -> (P::Frontier, P::Frontier)
where
    P: HcpProblem,
{
    let forward_span = tracing::trace_span!(
        "summary_apply_forward",
        start = a,
        end = m,
        blocks = mid_idx - start_idx
    );
    let backward_span = tracing::trace_span!(
        "summary_apply_reverse",
        start = m,
        end = c,
        blocks = end_idx - mid_idx
    );
    let forward = forward_span.in_scope(|| sigma_left.apply(frontier_a));
    let backward = backward_span.in_scope(|| sigma_right.apply_reverse(frontier_c));
    (forward, backward)
}

#[cfg(all(not(feature = "parallel"), not(feature = "tracing")))]
#[allow(clippy::too_many_arguments)]
fn apply_summaries_pair<P>(
    sigma_left: &P::Summary,
    sigma_right: &P::Summary,
    frontier_a: &P::Frontier,
    frontier_c: &P::Frontier,
    _a: usize,
    _m: usize,
    _c: usize,
    _start_idx: usize,
    _mid_idx: usize,
    _end_idx: usize,
) -> (P::Frontier, P::Frontier)
where
    P: HcpProblem,
{
    (
        sigma_left.apply(frontier_a),
        sigma_right.apply_reverse(frontier_c),
    )
}

#[cfg(all(feature = "parallel", feature = "tracing"))]
#[allow(clippy::too_many_arguments)]
fn recurse_halves<P>(
    engine: &HcpEngine<P>,
    blocks: &[BlockSummary<P::Summary>],
    tree: &SummaryTree<P>,
    frontiers: &[P::Frontier],
    start_idx: usize,
    mid_idx: usize,
    end_idx: usize,
    beta_a: &P::Boundary,
    beta_m: &P::Boundary,
    beta_c: &P::Boundary,
    a: usize,
    m: usize,
    c: usize,
) -> (Vec<P::State>, Vec<P::State>)
where
    P: HcpProblem + Sync,
    P::Summary: Send + Sync,
    P::Frontier: Send + Sync,
    P::State: Send,
    P::Boundary: Sync,
{
    let left_span = tracing::trace_span!(
        "reconstruct_left",
        start_idx,
        mid_idx,
        a,
        m,
        depth = (mid_idx - start_idx)
    );
    let right_span = tracing::trace_span!(
        "reconstruct_right",
        mid_idx,
        end_idx,
        m,
        c,
        depth = (end_idx - mid_idx)
    );
    join(
        {
            let left_span = left_span.clone();
            move || {
                left_span.in_scope(|| {
                    engine.reconstruct_path_on_blocks(
                        blocks, tree, frontiers, start_idx, mid_idx, beta_a, beta_m,
                    )
                })
            }
        },
        {
            let right_span = right_span.clone();
            move || {
                right_span.in_scope(|| {
                    engine.reconstruct_path_on_blocks(
                        blocks, tree, frontiers, mid_idx, end_idx, beta_m, beta_c,
                    )
                })
            }
        },
    )
}

#[cfg(all(feature = "parallel", not(feature = "tracing")))]
#[allow(clippy::too_many_arguments)]
fn recurse_halves<P>(
    engine: &HcpEngine<P>,
    blocks: &[BlockSummary<P::Summary>],
    tree: &SummaryTree<P>,
    frontiers: &[P::Frontier],
    start_idx: usize,
    mid_idx: usize,
    end_idx: usize,
    beta_a: &P::Boundary,
    beta_m: &P::Boundary,
    beta_c: &P::Boundary,
    _a: usize,
    _m: usize,
    _c: usize,
) -> (Vec<P::State>, Vec<P::State>)
where
    P: HcpProblem + Sync,
    P::Summary: Send + Sync,
    P::Frontier: Send + Sync,
    P::State: Send,
    P::Boundary: Sync,
{
    join(
        || {
            engine.reconstruct_path_on_blocks(
                blocks, tree, frontiers, start_idx, mid_idx, beta_a, beta_m,
            )
        },
        || {
            engine.reconstruct_path_on_blocks(
                blocks, tree, frontiers, mid_idx, end_idx, beta_m, beta_c,
            )
        },
    )
}

#[cfg(all(not(feature = "parallel"), feature = "tracing"))]
#[allow(clippy::too_many_arguments)]
fn recurse_halves<P>(
    engine: &HcpEngine<P>,
    blocks: &[BlockSummary<P::Summary>],
    tree: &SummaryTree<P>,
    frontiers: &[P::Frontier],
    start_idx: usize,
    mid_idx: usize,
    end_idx: usize,
    beta_a: &P::Boundary,
    beta_m: &P::Boundary,
    beta_c: &P::Boundary,
    a: usize,
    m: usize,
    c: usize,
) -> (Vec<P::State>, Vec<P::State>)
where
    P: HcpProblem,
{
    let left_span = tracing::trace_span!(
        "reconstruct_left",
        start_idx,
        mid_idx,
        a,
        m,
        depth = (mid_idx - start_idx)
    );
    let right_span = tracing::trace_span!(
        "reconstruct_right",
        mid_idx,
        end_idx,
        m,
        c,
        depth = (end_idx - mid_idx)
    );
    let left = left_span.in_scope(|| {
        engine
            .reconstruct_path_on_blocks(blocks, tree, frontiers, start_idx, mid_idx, beta_a, beta_m)
    });
    let right = right_span.in_scope(|| {
        engine.reconstruct_path_on_blocks(blocks, tree, frontiers, mid_idx, end_idx, beta_m, beta_c)
    });
    (left, right)
}

#[cfg(all(not(feature = "parallel"), not(feature = "tracing")))]
#[allow(clippy::too_many_arguments)]
fn recurse_halves<P>(
    engine: &HcpEngine<P>,
    blocks: &[BlockSummary<P::Summary>],
    tree: &SummaryTree<P>,
    frontiers: &[P::Frontier],
    start_idx: usize,
    mid_idx: usize,
    end_idx: usize,
    beta_a: &P::Boundary,
    beta_m: &P::Boundary,
    beta_c: &P::Boundary,
    _a: usize,
    _m: usize,
    _c: usize,
) -> (Vec<P::State>, Vec<P::State>)
where
    P: HcpProblem,
{
    (
        engine.reconstruct_path_on_blocks(
            blocks, tree, frontiers, start_idx, mid_idx, beta_a, beta_m,
        ),
        engine
            .reconstruct_path_on_blocks(blocks, tree, frontiers, mid_idx, end_idx, beta_m, beta_c),
    )
}

#[cfg(feature = "parallel")]
impl<P> HcpEngine<P>
where
    P: HcpProblem + Sync,
    P::Summary: Send + Sync,
    P::Frontier: Send + Sync,
    P::State: Send,
    P::Boundary: Sync,
{
    /// Run the full height-compressed DP, enforcing the thread-safety bounds required
    /// by the `parallel` feature.
    ///
    /// Returns `(optimal_cost, optimal_path_states)`. The path is guaranteed to be
    /// monotone with respect to layer indices and compatible with the boundaries
    /// returned by the problem implementation.
    ///
    /// # Panics
    /// Panics if the supplied [`HcpProblem`] violates the trait contract (for
    /// example by returning summaries that are not adjacent or by producing
    /// incompatible boundaries).
    pub fn run(&self) -> (P::Cost, Vec<P::State>) {
        #[cfg(feature = "tracing")]
        let span = tracing::info_span!("hcp_run");
        #[cfg(feature = "tracing")]
        let _enter = span.enter();

        let BuildArtifacts {
            blocks,
            tree,
            frontiers,
        } = {
            #[cfg(feature = "tracing")]
            let span = tracing::info_span!("build_block_summaries");
            #[cfg(feature = "tracing")]
            let _enter = span.enter();
            self.build_block_summaries()
        };

        let frontier_t = frontiers
            .last()
            .cloned()
            .unwrap_or_else(|| self.problem.init_frontier());

        let beta_0 = self.problem.initial_boundary();
        let beta_t = self.problem.terminal_boundary(&frontier_t);

        let path = {
            #[cfg(feature = "tracing")]
            let span = tracing::info_span!("reconstruct");
            #[cfg(feature = "tracing")]
            let _enter = span.enter();
            self.reconstruct_path_on_blocks(
                &blocks,
                &tree,
                &frontiers,
                0,
                blocks.len(),
                &beta_0,
                &beta_t,
            )
        };
        let cost = self.problem.extract_cost(&frontier_t, &beta_t);

        (cost, path)
    }

    /// Recursively reconstruct an optimal path across a contiguous block range.
    ///
    /// The input slices must correspond to the artefacts returned by
    /// [`build_block_summaries`](Self::build_block_summaries); supplying mismatched
    /// blocks or frontiers will panic.
    #[allow(clippy::too_many_arguments)]
    fn reconstruct_path_on_blocks(
        &self,
        blocks: &[BlockSummary<P::Summary>],
        tree: &SummaryTree<P>,
        frontiers: &[P::Frontier],
        start_idx: usize,
        end_idx: usize,
        beta_a: &P::Boundary,
        beta_c: &P::Boundary,
    ) -> Vec<P::State> {
        #[cfg(feature = "tracing")]
        let span = tracing::trace_span!(
            "reconstruct_interval",
            start_idx,
            end_idx,
            blocks = end_idx.saturating_sub(start_idx)
        );
        #[cfg(feature = "tracing")]
        let _enter = span.enter();

        match end_idx.saturating_sub(start_idx) {
            0 => Vec::new(),
            1 => {
                let b = &blocks[start_idx];
                self.problem
                    .reconstruct_block(b.start, b.end, beta_a, beta_c)
            }
            _ => {
                let mid_idx = start_idx + (end_idx - start_idx) / 2;

                let a = blocks[start_idx].start;
                let m = blocks[mid_idx - 1].end;
                let c = blocks[end_idx - 1].end;

                #[cfg(feature = "tracing")]
                let interval_span = tracing::trace_span!(
                    "reconstruct_interval_detail",
                    a,
                    m,
                    c,
                    start_idx,
                    mid_idx,
                    end_idx
                );
                #[cfg(feature = "tracing")]
                let _interval_guard = interval_span.enter();

                let (sigma_left, sigma_right) =
                    query_forward_pair::<P>(&self.problem, tree, start_idx, mid_idx, end_idx);

                let frontier_a = &frontiers[start_idx];
                let frontier_c = &frontiers[end_idx];
                let (frontier_m_forward, frontier_m_backward) = apply_summaries_pair::<P>(
                    &sigma_left,
                    &sigma_right,
                    frontier_a,
                    frontier_c,
                    a,
                    m,
                    c,
                    start_idx,
                    mid_idx,
                    end_idx,
                );

                #[cfg(feature = "tracing")]
                let beta_m = {
                    let span = tracing::trace_span!("choose_boundary", a, m, c);
                    span.in_scope(|| {
                        self.problem.choose_boundary_with_frontiers(
                            a,
                            m,
                            c,
                            frontier_a,
                            &frontier_m_forward,
                            &frontier_m_backward,
                            frontier_c,
                            &sigma_left,
                            &sigma_right,
                            beta_a,
                            beta_c,
                        )
                    })
                };
                #[cfg(not(feature = "tracing"))]
                let beta_m = self.problem.choose_boundary_with_frontiers(
                    a,
                    m,
                    c,
                    frontier_a,
                    &frontier_m_forward,
                    &frontier_m_backward,
                    frontier_c,
                    &sigma_left,
                    &sigma_right,
                    beta_a,
                    beta_c,
                );

                let (path_left, path_right) = recurse_halves::<P>(
                    self, blocks, tree, frontiers, start_idx, mid_idx, end_idx, beta_a, &beta_m,
                    beta_c, a, m, c,
                );

                if path_left.is_empty() {
                    return path_right;
                }
                if path_right.is_empty() {
                    return path_left;
                }

                let mut out = path_left;
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

#[cfg(not(feature = "parallel"))]
impl<P: HcpProblem> HcpEngine<P> {
    /// Run the full height-compressed DP (serial execution).
    ///
    /// Returns `(optimal_cost, optimal_path_states)`. The reconstructed path
    /// begins at the problem's initial boundary and ends at the terminal
    /// boundary selected by [`HcpProblem::terminal_boundary`].
    ///
    /// # Panics
    /// Panics if the `HcpProblem` implementation violates the algebraic laws
    /// (e.g. summaries are not boundary independent or merges are not associative).
    pub fn run(&self) -> (P::Cost, Vec<P::State>) {
        #[cfg(feature = "tracing")]
        let span = tracing::info_span!("hcp_run");
        #[cfg(feature = "tracing")]
        let _enter = span.enter();

        let BuildArtifacts {
            blocks,
            tree,
            frontiers,
        } = {
            #[cfg(feature = "tracing")]
            let span = tracing::info_span!("build_block_summaries");
            #[cfg(feature = "tracing")]
            let _enter = span.enter();
            self.build_block_summaries()
        };

        let frontier_t = frontiers
            .last()
            .cloned()
            .unwrap_or_else(|| self.problem.init_frontier());

        let beta_0 = self.problem.initial_boundary();
        let beta_t = self.problem.terminal_boundary(&frontier_t);

        let path = {
            #[cfg(feature = "tracing")]
            let span = tracing::info_span!("reconstruct");
            #[cfg(feature = "tracing")]
            let _enter = span.enter();
            self.reconstruct_path_on_blocks(
                &blocks,
                &tree,
                &frontiers,
                0,
                blocks.len(),
                &beta_0,
                &beta_t,
            )
        };
        let cost = self.problem.extract_cost(&frontier_t, &beta_t);

        (cost, path)
    }

    /// Recursively reconstruct an optimal path across a contiguous block range.
    ///
    /// The input slices must correspond to the artefacts returned by
    /// [`build_block_summaries`](Self::build_block_summaries); supplying mismatched
    /// blocks or frontiers will panic.
    #[allow(clippy::too_many_arguments)]
    fn reconstruct_path_on_blocks(
        &self,
        blocks: &[BlockSummary<P::Summary>],
        tree: &SummaryTree<P>,
        frontiers: &[P::Frontier],
        start_idx: usize,
        end_idx: usize,
        beta_a: &P::Boundary,
        beta_c: &P::Boundary,
    ) -> Vec<P::State> {
        #[cfg(feature = "tracing")]
        let span = tracing::trace_span!(
            "reconstruct_interval",
            start_idx,
            end_idx,
            blocks = end_idx.saturating_sub(start_idx)
        );
        #[cfg(feature = "tracing")]
        let _enter = span.enter();

        match end_idx.saturating_sub(start_idx) {
            0 => Vec::new(),
            1 => {
                let b = &blocks[start_idx];
                self.problem
                    .reconstruct_block(b.start, b.end, beta_a, beta_c)
            }
            _ => {
                let mid_idx = start_idx + (end_idx - start_idx) / 2;

                let a = blocks[start_idx].start;
                let m = blocks[mid_idx - 1].end;
                let c = blocks[end_idx - 1].end;

                #[cfg(feature = "tracing")]
                let interval_span = tracing::trace_span!(
                    "reconstruct_interval_detail",
                    a,
                    m,
                    c,
                    start_idx,
                    mid_idx,
                    end_idx
                );
                #[cfg(feature = "tracing")]
                let _interval_guard = interval_span.enter();

                let (sigma_left, sigma_right) =
                    query_forward_pair::<P>(&self.problem, tree, start_idx, mid_idx, end_idx);

                let frontier_a = &frontiers[start_idx];
                let frontier_c = &frontiers[end_idx];
                let (frontier_m_forward, frontier_m_backward) = apply_summaries_pair::<P>(
                    &sigma_left,
                    &sigma_right,
                    frontier_a,
                    frontier_c,
                    a,
                    m,
                    c,
                    start_idx,
                    mid_idx,
                    end_idx,
                );

                #[cfg(feature = "tracing")]
                let beta_m = {
                    let span = tracing::trace_span!("choose_boundary", a, m, c);
                    span.in_scope(|| {
                        self.problem.choose_boundary_with_frontiers(
                            a,
                            m,
                            c,
                            frontier_a,
                            &frontier_m_forward,
                            &frontier_m_backward,
                            frontier_c,
                            &sigma_left,
                            &sigma_right,
                            beta_a,
                            beta_c,
                        )
                    })
                };
                #[cfg(not(feature = "tracing"))]
                let beta_m = self.problem.choose_boundary_with_frontiers(
                    a,
                    m,
                    c,
                    frontier_a,
                    &frontier_m_forward,
                    &frontier_m_backward,
                    frontier_c,
                    &sigma_left,
                    &sigma_right,
                    beta_a,
                    beta_c,
                );

                let (path_left, path_right) = recurse_halves::<P>(
                    self, blocks, tree, frontiers, start_idx, mid_idx, end_idx, beta_a, &beta_m,
                    beta_c, a, m, c,
                );

                if path_left.is_empty() {
                    return path_right;
                }
                if path_right.is_empty() {
                    return path_left;
                }

                let mut out = path_left;
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

#[derive(Debug)]
struct SummaryTreeNode<S> {
    start: usize,
    end: usize,
    left: Option<usize>,
    right: Option<usize>,
    forward: Option<S>,
    reverse: Option<S>,
}

impl<S> SummaryTreeNode<S> {
    fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            left: None,
            right: None,
            forward: None,
            reverse: None,
        }
    }
}

struct SummaryTree<P: HcpProblem> {
    nodes: Vec<SummaryTreeNode<P::Summary>>,
    root: Option<usize>,
}

struct BuildArtifacts<P: HcpProblem> {
    blocks: Vec<BlockSummary<P::Summary>>,
    tree: SummaryTree<P>,
    frontiers: Vec<P::Frontier>,
}

impl<P: HcpProblem> SummaryTree<P> {
    fn new(problem: &P, blocks: &[BlockSummary<P::Summary>]) -> Self {
        #[cfg(feature = "tracing")]
        let span = tracing::trace_span!(
            "summary_tree_new",
            num_blocks = blocks.len(),
            empty = blocks.is_empty()
        );
        #[cfg(feature = "tracing")]
        let _enter = span.enter();

        let mut tree = SummaryTree {
            nodes: Vec::with_capacity(blocks.len().saturating_mul(4).max(1)),
            root: None,
        };
        if !blocks.is_empty() {
            let root = tree.build(problem, blocks, 0, blocks.len());
            tree.root = Some(root);
        }
        tree
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, problem, blocks))
    )]
    fn build(
        &mut self,
        problem: &P,
        blocks: &[BlockSummary<P::Summary>],
        start: usize,
        end: usize,
    ) -> usize {
        let node_idx = self.nodes.len();
        self.nodes.push(SummaryTreeNode::new(start, end));

        if start + 1 == end {
            let summary = blocks[start].summary.clone();
            let node = &mut self.nodes[node_idx];
            node.forward = Some(summary.clone());
            node.reverse = Some(summary);
            return node_idx;
        }

        let mid = start + (end - start) / 2;
        let left_idx = self.build(problem, blocks, start, mid);
        let right_idx = self.build(problem, blocks, mid, end);

        let forward = {
            let left = self.nodes[left_idx]
                .forward
                .as_ref()
                .expect("left summary missing");
            let right = self.nodes[right_idx]
                .forward
                .as_ref()
                .expect("right summary missing");
            problem.merge_summary(left, right)
        };

        let reverse = {
            let right = self.nodes[right_idx]
                .reverse
                .as_ref()
                .expect("right reverse summary missing");
            let left = self.nodes[left_idx]
                .reverse
                .as_ref()
                .expect("left reverse summary missing");
            problem.merge_summary(right, left)
        };

        let node = &mut self.nodes[node_idx];
        node.left = Some(left_idx);
        node.right = Some(right_idx);
        node.forward = Some(forward);
        node.reverse = Some(reverse);

        node_idx
    }

    fn query_forward(
        &self,
        problem: &P,
        range_start: usize,
        range_end: usize,
    ) -> Option<P::Summary> {
        #[cfg(feature = "tracing")]
        let span = tracing::trace_span!(
            "summary_query_forward_root",
            range_start,
            range_end,
            has_root = self.root.is_some()
        );
        #[cfg(feature = "tracing")]
        let _enter = span.enter();

        self.query(
            problem,
            self.root,
            range_start,
            range_end,
            Direction::Forward,
        )
    }

    #[allow(dead_code)]
    fn query_reverse(
        &self,
        problem: &P,
        range_start: usize,
        range_end: usize,
    ) -> Option<P::Summary> {
        #[cfg(feature = "tracing")]
        let span = tracing::trace_span!(
            "summary_query_reverse_root",
            range_start,
            range_end,
            has_root = self.root.is_some()
        );
        #[cfg(feature = "tracing")]
        let _enter = span.enter();

        self.query(
            problem,
            self.root,
            range_start,
            range_end,
            Direction::Reverse,
        )
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            level = "trace",
            skip(self, problem),
            fields(direction = tracing::field::Empty)
        )
    )]
    fn query(
        &self,
        problem: &P,
        node_idx: Option<usize>,
        range_start: usize,
        range_end: usize,
        direction: Direction,
    ) -> Option<P::Summary> {
        #[cfg(feature = "tracing")]
        tracing::Span::current().record("direction", direction.as_str());

        let idx = node_idx?;
        let node = &self.nodes[idx];

        if range_end <= node.start || node.end <= range_start {
            return None;
        }
        if range_start <= node.start && node.end <= range_end {
            return match direction {
                Direction::Forward => node.forward.clone(),
                Direction::Reverse => node.reverse.clone(),
            };
        }

        let left = self.query(problem, node.left, range_start, range_end, direction);
        let right = self.query(problem, node.right, range_start, range_end, direction);

        match (left, right) {
            (Some(lv), Some(rv)) => {
                let merged = match direction {
                    Direction::Forward => problem.merge_summary(&lv, &rv),
                    Direction::Reverse => problem.merge_summary(&rv, &lv),
                };
                Some(merged)
            }
            (Some(val), None) | (None, Some(val)) => Some(val),
            (None, None) => None,
        }
    }
}

#[allow(dead_code)]
#[derive(Copy, Clone)]
enum Direction {
    Forward,
    Reverse,
}

impl Direction {
    #[cfg(feature = "tracing")]
    fn as_str(self) -> &'static str {
        match self {
            Direction::Forward => "forward",
            Direction::Reverse => "reverse",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::SummaryApply;

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

    impl SummaryApply<F> for S {
        fn apply(&self, frontier: &F) -> F {
            let _ = self;
            frontier.clone()
        }
    }

    #[test]
    fn build_block_summaries_shapes() {
        let problem = DummyProblem { t: 4 };
        let engine = HcpEngine::with_block_size(problem, 2);
        let artifacts = engine.build_block_summaries();
        let blocks = artifacts.blocks;
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
