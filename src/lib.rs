//! Height-Compressed Dynamic Programming (HCP-DP).
//!
//! This crate is an alpha-stage correctness rebuild. The public surface is
//! intentionally small: a generic summary-tree engine plus two reference
//! implementations, LCS and linear Needleman-Wunsch, whose returned paths are
//! tested against independent scoring baselines.

pub mod blocks;
pub mod builder;
pub mod engine;
pub mod problems;
pub mod traits;
pub mod utils;

pub use crate::builder::HcpEngineBuilder;
pub use crate::engine::HcpEngine;
pub use crate::traits::{HcpProblem, SummaryApply};
