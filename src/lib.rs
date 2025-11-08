//! Height-Compressed Dynamic Programming (HCP-DP)
//!
//! This crate provides a generic engine for running large dynamic programs
//! in sublinear space (in the number of layers), following the
//! height-compression framework described in the associated paper.
//!
//! The core idea:
//! - Express your DP as a *layered* computation with bounded frontier width.
//! - Implement the [`HcpProblem`] trait for your problem.
//! - Use [`HcpEngine`] to compute exact optimal values and reconstruct
//!   an optimal path using significantly less memory than a naive DP table.
//!
//! The `problems` module contains reference implementations for:
//! - Longest Common Subsequence (LCS)
//! - Needleman–Wunsch global alignment
//! - Matrix-chain multiplication
//! - Viterbi decoding for small HMMs
//!
//! These serve both as ready-to-use tools and as templates for your own DPs.

pub mod blocks;
pub mod builder;
pub mod engine;
pub mod problems;
pub mod traits;
pub mod utils;

pub use crate::builder::HcpEngineBuilder;
pub use crate::engine::HcpEngine;
pub use crate::traits::HcpProblem;
