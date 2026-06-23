//! Height-Compressed Dynamic Programming (HCP-DP).
//!
//! This crate is an alpha-stage correctness rebuild. The public surface is
//! intentionally small: a generic summary-tree engine plus correctness-tested
//! reference implementations for LCS, linear and affine-gap Needleman-Wunsch,
//! linear-gap Smith-Waterman, edit distance, and semi-global alignment.

pub mod alignment;
pub mod blocks;
pub mod builder;
pub mod engine;
pub mod problems;
#[doc(hidden)]
pub mod sequence_io;
pub mod traits;
pub mod utils;

pub use crate::builder::HcpEngineBuilder;
pub use crate::engine::HcpEngine;
pub use crate::traits::{HcpProblem, SummaryApply};
