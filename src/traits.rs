//! Core contracts for height-compressed dynamic programs.
//!
//! The crate intentionally exposes a small pre-1.0 interface. A problem is
//! admissible only when its interval summaries are true operators and its
//! reconstruction is constrained by explicit interval endpoints.

/// An interval summary that can advance a compatible frontier.
///
/// A summary must depend only on the fixed problem data and the interval it
/// covers. It must not be a cached output for one particular input frontier.
pub trait SummaryApply<Frontier> {
    /// Apply this interval operator to a frontier at the interval start.
    fn apply(&self, frontier: &Frontier) -> Frontier;
}

/// A fixed height-compressible DP instance.
///
/// Implementations must provide:
/// - a layered recurrence over `0..=num_layers()`,
/// - boundary-independent summaries for any interval `[a, b]`,
/// - associative summary merging for adjacent intervals,
/// - endpoint-constrained split selection and leaf reconstruction.
pub trait HcpProblem {
    /// A state on the returned optimal path.
    type State: Clone + PartialEq;

    /// A DP frontier at one layer.
    type Frontier: Clone;

    /// Boundary-independent interval summary.
    type Summary: Clone + SummaryApply<Self::Frontier>;

    /// A reconstruction boundary at a layer.
    type Boundary: Clone + PartialEq;

    /// Objective value.
    type Cost: Copy + Ord;

    /// Number of layer transitions.
    fn num_layers(&self) -> usize;

    /// Frontier at layer 0.
    fn init_frontier(&self) -> Self::Frontier;

    /// Direct one-layer recurrence, used for validation and simple replay.
    fn forward_step(&self, layer: usize, frontier: &Self::Frontier) -> Self::Frontier;

    /// Boundary-independent summary for interval `[a, b)`.
    fn summarize_interval(&self, a: usize, b: usize) -> Self::Summary;

    /// Merge adjacent summaries into their union.
    fn merge_summary(&self, left: &Self::Summary, right: &Self::Summary) -> Self::Summary;

    /// Boundary at layer 0.
    fn initial_boundary(&self) -> Self::Boundary;

    /// Terminal boundary selected from the final frontier.
    fn terminal_boundary(&self, frontier_t: &Self::Frontier) -> Self::Boundary;

    /// Choose a boundary at split layer `m` for interval `[a, c)`.
    ///
    /// The returned boundary must be compatible with at least one optimal path
    /// from `beta_a` to `beta_c`.
    #[allow(clippy::too_many_arguments)]
    fn choose_split(
        &self,
        a: usize,
        m: usize,
        c: usize,
        beta_a: &Self::Boundary,
        beta_c: &Self::Boundary,
        sigma_left: &Self::Summary,
        sigma_right: &Self::Summary,
    ) -> Self::Boundary;

    /// Reconstruct an optimal leaf segment on `[a, b]` under fixed endpoints.
    ///
    /// The returned segment must start at `beta_a` and end at `beta_b`.
    fn reconstruct_leaf(
        &self,
        a: usize,
        b: usize,
        beta_a: &Self::Boundary,
        beta_b: &Self::Boundary,
    ) -> Vec<Self::State>;

    /// Extract the global objective value from the final frontier.
    fn extract_cost(&self, frontier_t: &Self::Frontier, beta_t: &Self::Boundary) -> Self::Cost;
}
