use crate::engine::HcpEngine;
use crate::utils::default_block_size;
use crate::HcpProblem;

/// Builder for configuring an [`HcpEngine`] before execution.
///
/// The builder lets you override the block size heuristic and execution profile.
/// By default it mirrors [`HcpEngine::new`] with a square-root checkpoint block
/// size.
///
/// # Examples
/// ```
/// use hcp_dp::{HcpEngineBuilder, problems::lcs::LcsProblem};
///
/// let engine = HcpEngineBuilder::new(LcsProblem::new(b"AA", b"A"))
///     .linear_space()
///     .build();
///
/// let (cost, path) = engine.run();
/// assert_eq!(cost, 1);
/// assert_eq!(path.last(), Some(&(2, 1)));
/// ```
pub struct HcpEngineBuilder<P: HcpProblem> {
    problem: P,
    block_size: Option<usize>,
}

impl<P: HcpProblem> HcpEngineBuilder<P> {
    /// Start a builder for the supplied problem, using default configuration.
    pub fn new(problem: P) -> Self {
        Self {
            problem,
            block_size: None,
        }
    }
    /// Override the block size used during forward summarisation.
    ///
    /// # Panics
    /// Panics if `block_size` is zero when [`build`](Self::build) is called.
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = Some(block_size);
        self
    }

    /// Use the memory-minimal exact traceback profile (`block_size = 1`).
    ///
    /// This is the named builder equivalent of [`HcpEngine::linear_space`].
    pub fn linear_space(mut self) -> Self {
        self.block_size = Some(1);
        self
    }

    /// Finalise the builder and return a configured [`HcpEngine`].
    pub fn build(self) -> HcpEngine<P> {
        match self.block_size {
            Some(b) => HcpEngine::with_block_size(self.problem, b),
            None => {
                let t = self.problem.num_layers().max(1);
                let b = default_block_size(t).max(1);
                HcpEngine::with_block_size(self.problem, b)
            }
        }
    }
}
