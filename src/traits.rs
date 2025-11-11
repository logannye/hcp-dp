//! Core trait definitions for height-compressible dynamic programs.
//!
//! To plug your DP into the height-compressed engine, implement [`HcpProblem`]
//! for a problem struct that captures your instance (e.g. sequences, costs).
//!
//! The trait encodes the interface described in the paper:
//! - Layered structure: layers 0..=T, with transitions between consecutive layers.
//! - Frontiers: compact representation of all DP values on a layer.
//! - Interval summaries: composable Σ[a,b] for blocks of layers.
//! - Boundary conditions: constraints that pin down feasible endpoints.
//! - Local reconstruction: ability to reconstruct an optimal path inside a block.
//!
//! The engine orchestrates block-level summarization and recursive reconstruction
//! using only these primitives. Implementations are free to recompute within
//! blocks as needed, as long as they adhere to the asymptotic space constraints.

/// Trait implemented by interval summaries Σ[a,b] so they can be applied to a frontier.
///
/// The whitepaper requires *boundary-independent* summaries: Σ[a,b] depends
/// only on the DP structure between layers `a` and `b`. Applying the summary
/// to a frontier representing layer `a` must yield the corresponding frontier
/// at `b` as if the DP had been run locally.
pub trait SummaryApply<Frontier> {
    /// Apply the summary to an input frontier to obtain the output frontier.
    ///
    /// Implementations should be side-effect free and must not mutate the input.
    /// The result should match a direct execution of the DP over the interval.
    fn apply(&self, frontier: &Frontier) -> Frontier;

    /// Apply the summary in reverse, propagating information from the interval's
    /// end back to its start.
    ///
    /// The default implementation clones the input frontier, which is useful for
    /// problems that do not require reverse propagation.
    fn apply_reverse(&self, frontier: &Frontier) -> Frontier
    where
        Frontier: Clone,
    {
        frontier.clone()
    }
}

/// Trait for a height-compressible dynamic program instance.
///
/// An `HcpProblem` corresponds to a *fixed* DP instance:
/// typically a struct containing immutable input data (sequences, matrices,
/// graph layers, emission tables, …).
///
/// # Contract
/// Implementations must guarantee:
///
/// - `num_layers()` equals the number of forward steps needed to reach the
///   terminal frontier. Layers are indexed `0..=T`.
/// - `forward_step(i, frontier)` consumes layer `i` and returns the frontier for
///   layer `i + 1` without mutating the input frontier.
/// - `summarize_block(a, b, frontier_a)` is equivalent to iterating
///   `forward_step` for `layer = a..b-1`, yielding both the frontier at `b` and
///   a boundary-independent summary Σ[a,b].
/// - `merge_summary` is associative: for adjacent blocks `[a,b)`, `[b,c)`,
///   `[c,d)` we expect `Σ[a,b] ⊕ (Σ[b,c] ⊕ Σ[c,d]) == (Σ[a,b] ⊕ Σ[b,c]) ⊕ Σ[c,d]`
///   and applying either side to a frontier produces the same result.
/// - `choose_boundary` (or `choose_boundary_with_frontiers`) always returns a
///   boundary that is consistent with at least one globally optimal solution,
///   ensuring recursive reconstruction terminates.
///
/// Violating these invariants will cause panics or incorrect reconstructions.
pub trait HcpProblem {
    /// A single state along the reconstructed optimal path.
    /// For sequence DPs, often `(i,j)` indices; for others, a vertex id, etc.
    type State: Clone + PartialEq;

    /// Representation of the DP frontier at a layer (e.g., DP row/column).
    type Frontier: Clone;

    /// Interval summary type Σ[a,b].
    ///
    /// The summary must be boundary-independent, support the associative merge
    /// operator defined by [`merge_summary`], and be able to advance any
    /// compatible frontier by calling [`SummaryApply::apply`].
    type Summary: Clone + SummaryApply<Self::Frontier>;

    /// Boundary condition at a layer.
    ///
    /// Encodes constraints on valid states at a layer during reconstruction,
    /// such as a fixed index, a small set of indices, or additional metadata.
    type Boundary: Clone;

    /// Objective / cost type.
    ///
    /// Must support ordering so we can distinguish optimal solutions.
    type Cost: Copy + Ord;

    /// Number of DP layers/steps `T`.
    ///
    /// We will:
    /// - initialize a frontier at layer 0,
    /// - apply `forward_step` for i = 0..T-1,
    /// - obtain final frontier at layer T.
    ///
    /// Return the total number of DP layers (T).
    ///
    /// The engine performs exactly `T` calls to [`forward_step`](Self::forward_step)
    /// during the forward phase.
    fn num_layers(&self) -> usize;

    /// Initialize the frontier at layer 0.
    ///
    /// Initialise the frontier at layer 0. Must be side-effect free.
    fn init_frontier(&self) -> Self::Frontier;

    /// Perform one DP step: from layer `i` to `i+1`.
    ///
    ///
    /// Requirements:
    /// - May only depend on `frontier_i` and immutable problem data.
    /// - Must run in O(W) space, where W is the frontier width.
    fn forward_step(&self, layer: usize, frontier_i: &Self::Frontier) -> Self::Frontier;

    /// Summarize a block of layers [a, b).
    ///
    /// Repeatedly calls `forward_step` from layer `a` to `b`, using only O(W) space,
    /// and returns:
    /// - the resulting frontier at layer b, and
    /// - a summary Σ[a,b] capturing the effect of this interval.
    fn summarize_block(
        &self,
        a: usize,
        b: usize,
        frontier_a: &Self::Frontier,
    ) -> (Self::Frontier, Self::Summary);

    /// Merge two adjacent summaries:
    ///
    /// Given Σ[a,b] and Σ[b,c], return Σ[a,c] = Σ[a,b] ⊕ Σ[b,c].
    ///
    /// This operator must be associative across chains of blocks and the
    /// returned summary must satisfy the same [`SummaryApply::apply`] law.
    fn merge_summary(&self, left: &Self::Summary, right: &Self::Summary) -> Self::Summary;

    /// Boundary condition at layer 0 representing the initial constraint.
    ///
    /// For many problems, this fixes a unique start state.
    fn initial_boundary(&self) -> Self::Boundary;

    /// Choose a terminal boundary at layer T from the final frontier.
    ///
    /// Typically selects one or more states achieving the optimal objective.
    fn terminal_boundary(&self, frontier_t: &Self::Frontier) -> Self::Boundary;

    /// Choose an intermediate boundary at layer `m` for the interval [a,c].
    ///
    /// Inputs:
    /// - `a < m < c`: layer indices split at `m`.
    /// - `sigma_left = Σ[a,m]`, `sigma_right = Σ[m,c]`: summaries for left/right.
    /// - `beta_a`, `beta_c`: boundary conditions at layers a and c that are
    ///   known to be compatible with some globally optimal path.
    ///
    /// Must return a boundary `beta_m` at layer m such that:
    /// - there exists at least one optimal path over [a,c] consistent with
    ///   (beta_a, beta_m, beta_c).
    ///
    /// Implementations are allowed to recompute locally inside [a,c] as needed,
    /// as long as they respect the intended space bounds (O(W) working memory).
    #[allow(clippy::too_many_arguments)]
    fn choose_boundary(
        &self,
        a: usize,
        m: usize,
        c: usize,
        sigma_left: &Self::Summary,
        sigma_right: &Self::Summary,
        beta_a: &Self::Boundary,
        beta_c: &Self::Boundary,
    ) -> Self::Boundary;

    /// Optional hook that receives forward/backward frontiers derived from Σ and
    /// can be used to select the midpoint boundary without local recomputation.
    ///
    /// The default implementation falls back to [`choose_boundary`].
    #[allow(clippy::too_many_arguments)]
    fn choose_boundary_with_frontiers(
        &self,
        a: usize,
        m: usize,
        c: usize,
        _frontier_a: &Self::Frontier,
        _frontier_m_forward: &Self::Frontier,
        _frontier_m_backward: &Self::Frontier,
        _frontier_c: &Self::Frontier,
        sigma_left: &Self::Summary,
        sigma_right: &Self::Summary,
        beta_a: &Self::Boundary,
        beta_c: &Self::Boundary,
    ) -> Self::Boundary {
        self.choose_boundary(a, m, c, sigma_left, sigma_right, beta_a, beta_c)
    }

    /// Reconstruct an optimal subpath on [a,b] under boundary conditions.
    ///
    /// Inputs:
    /// - `a < b` layer indices,
    /// - `beta_a`, `beta_b`: feasible boundary conditions at layers a and b.
    ///
    /// Must:
    /// - recompute DP locally inside [a,b] using O(W) extra space,
    /// - return a sequence of states `(state_a, ..., state_b)` representing
    ///   an optimal path segment consistent with these boundaries.
    ///
    /// The engine will concatenate these segments and drop duplicated
    /// boundary states at joins.
    fn reconstruct_block(
        &self,
        a: usize,
        b: usize,
        beta_a: &Self::Boundary,
        beta_b: &Self::Boundary,
    ) -> Vec<Self::State>;

    /// Extract the objective value from final frontier and terminal boundary.
    fn extract_cost(&self, frontier_t: &Self::Frontier, beta_t: &Self::Boundary) -> Self::Cost;
}
