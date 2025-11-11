//! Height-Compressed Dynamic Programming (HCP-DP)
//!
//! This crate provides a generic engine for running large dynamic programs
//! in sublinear space (in the number of layers), following the
//! height-compression framework described in the whitepaper.
//!
//! ## Core idea
//! 1. Model your recurrence as a *layered* DP with a bounded-width frontier.
//! 2. Implement the [`HcpProblem`] trait for that recurrence.
//! 3. Let [`HcpEngine`] orchestrate block summarisation and recursive
//!    reconstruction to obtain the optimal cost *and* an explicit path.
//!
//! Compared to a classic table-filling DP, the engine keeps only O(√T) layers
//! live at any time (T = number of layers) while still returning exact answers.
//!
//! ## Quick start
//! ```
//! use hcp_dp::{HcpEngine, problems::lcs::LcsProblem};
//!
//! let problem = LcsProblem::new(b"ACCG", b"ACGC");
//! let (len, path) = HcpEngine::new(problem).run();
//! assert_eq!(len, 3);
//! assert_eq!(path.first(), Some(&(0, 0)));
//! assert_eq!(path.last(), Some(&(4, 4)));
//! ```
//!
//! ## Built-in problems
//! The `problems` module contains reference implementations for:
//! - Longest Common Subsequence (LCS)
//! - Needleman–Wunsch global alignment (linear- and affine-gap variants)
//! - Matrix-chain multiplication
//! - Viterbi decoding for small HMMs
//! - Layered DAG shortest path
//! - Banded variants for near-diagonal alignments
//!
//! These serve both as ready-to-use tools and as templates for constructing your
//! own height-compressible dynamic programs.

pub mod blocks;
pub mod builder;
pub mod engine;
pub mod problems;
pub mod traits;
pub mod utils;

pub use crate::builder::HcpEngineBuilder;
pub use crate::engine::HcpEngine;
pub use crate::traits::HcpProblem;
