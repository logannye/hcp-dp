//! Height-Compressed Dynamic Programming (HCP-DP).
//!
//! This crate provides a generic summary-tree engine plus correctness-tested
//! sequence-alignment kernels. The current public surface is intentionally
//! small while each problem is proven against the same contract:
//!
//! - interval summaries apply like direct DP replay,
//! - adjacent summaries merge into the direct combined interval,
//! - recursive split boundaries are endpoint constrained,
//! - reconstructed paths join exactly,
//! - independently scored paths realize the reported objective.
//!
//! The companion `hcp-align` binary is the primary alpha user surface. It
//! exposes global, local, semi-global, affine-gap, and edit-distance modes with
//! structured output and optional baseline verification.
//!
//! See the repository README and `docs/design.md` for the technical overview.

pub mod alignment;
pub mod blocks;
pub mod builder;
pub mod engine;
pub mod problems;
pub mod traits;
pub mod utils;

pub use crate::builder::HcpEngineBuilder;
pub use crate::engine::{HcpEngine, HcpRunStats};
pub use crate::traits::{HcpProblem, SummaryApply};
