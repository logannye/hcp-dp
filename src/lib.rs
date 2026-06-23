//! Exact traceback from height-compressed dynamic-programming summaries.
//!
//! This crate provides a generic summary-tree engine, correctness-tested
//! sequence-alignment kernels, and a small dynamic-time-warping implementation.
//! The public surface is intentionally small while each exported problem is
//! proven against the same contract:
//!
//! - interval summaries apply like direct DP replay,
//! - adjacent summaries merge into the direct combined interval,
//! - recursive split boundaries are endpoint constrained,
//! - reconstructed paths join exactly,
//! - independently scored paths realize the reported objective.
//!
//! The companion `hcp-align` binary is the primary user surface. It
//! exposes global, local, semi-global, affine-gap, and edit-distance modes with
//! structured output, deterministic edit-distance backend selection, and
//! optional baseline verification.
//!
//! The edit-distance module also exposes specialized exact engines: rolling-row
//! linear-space scoring, adaptive-banded traceback for low-edit regimes, and a
//! Myers bit-vector score backend for arbitrary pattern lengths. Those are used
//! in report tooling to make the generic HCP traceback engine's tradeoffs
//! visible against specialized frontier algorithms.
//!
//! The [`contract`] module exposes reusable bounded-test helpers for authors of
//! new [`HcpProblem`] implementations.
//!
//! See the repository README and `docs/design.md` for the technical overview.

pub mod alignment;
pub mod blocks;
pub mod builder;
pub mod contract;
pub mod engine;
pub mod problems;
pub mod scoring;
pub mod seeding;
pub mod traits;
pub mod utils;

pub use crate::builder::HcpEngineBuilder;
pub use crate::engine::{HcpEngine, HcpRunStats};
pub use crate::traits::{HcpProblem, SummaryApply};
