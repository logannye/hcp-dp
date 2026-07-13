//! Public problem implementations.
//!
//! Only problems that satisfy the current cost/path correctness contract are
//! exported from this module.

pub mod edit_distance;
#[cfg(feature = "experimental-hcp")]
pub mod lcs;
pub mod nw_affine;
pub mod nw_align;
pub mod semiglobal;
pub mod smith_waterman;
