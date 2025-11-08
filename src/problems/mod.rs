//! Reference problem implementations for the HCP-DP engine.
//!
//! These modules show how to implement [`HcpProblem`](crate::traits::HcpProblem)
//! for concrete dynamic programs.
//!
//! They are both usable and serve as templates:
//! - [`lcs`]           : Longest Common Subsequence via Hirschberg-style splitting.
//! - [`nw_align`]      : Needleman–Wunsch global alignment (linear gap).
//! - [`matrix_chain`]  : Matrix-chain multiplication.
//! - [`viterbi`]       : Viterbi decoding for HMMs.

pub mod dag_sp;
pub mod lcs;
pub mod lcs_banded;
pub mod matrix_chain;
pub mod nw_affine;
pub mod nw_align;
pub mod viterbi;
