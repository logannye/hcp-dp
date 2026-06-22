//! Height-Compressed Dynamic Programming (HCP-DP).
//!
//! This crate is an alpha-stage correctness rebuild. The public surface is
//! intentionally small: a generic summary-tree engine plus correctness-tested
//! reference implementations for LCS, linear Needleman-Wunsch, and affine-gap
//! Needleman-Wunsch.

pub mod blocks;
pub mod builder;
pub mod engine;
pub mod problems;
pub mod traits;
pub mod utils;

pub use crate::builder::HcpEngineBuilder;
pub use crate::engine::HcpEngine;
pub use crate::traits::{HcpProblem, SummaryApply};
