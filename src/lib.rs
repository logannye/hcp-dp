//! Exact reference alignments, independent verification, and backend comparison.
//!
//! The supported v0.2 API is centered on [`solve`], [`verify`], and [`compare`].
//! It uses one canonical SAM-oriented CIGAR model across JSON, JSONL, PAF, and
//! SAM and verifies objective equivalence rather than requiring identical
//! traceback tie-breaking.
//!
//! The original height-compressed DP engine is available only through the
//! opt-in `experimental-hcp` feature and carries no v0.2 stability guarantee.

pub mod alignment;
pub mod batch;
#[allow(dead_code)]
mod blocks;
#[cfg(feature = "experimental-hcp")]
mod builder;
#[cfg(feature = "experimental-hcp")]
mod contract;
#[allow(dead_code)]
mod engine;
pub mod oracle;
#[allow(dead_code)]
mod problems;
pub mod record_io;
pub mod scoring;
pub mod sequence_io;
pub mod sequence_store;
mod traits;
mod utils;

pub use crate::batch::{
    BatchOptions, BatchSummary, ErrorContext, ErrorRecord, ErrorStage, OracleRecord,
};
pub use crate::oracle::{
    compare, record_to_paf, record_to_sam, records_from_json, records_from_jsonl, records_from_paf,
    records_from_sam, records_to_json, records_to_jsonl, records_to_sam, solve, verify,
    AlignmentMode, AlignmentRecord, AlignmentSpan, Cigar, CigarOp, ComparisonInputs,
    ComparisonReport, GapModel, ProblemSpec, ScoringScheme, SolveOptions, SolveRequest, SolveScope,
    SubstitutionScheme, VerificationLevel, VerificationReport, VerificationRequest,
    VerificationStatus,
};
pub use crate::scoring::SubstitutionMatrix;

/// Unstable height-compressed DP implementation APIs.
///
/// These APIs are intentionally excluded from the supported v0.2 product
/// contract and may change without migration support.
#[cfg(feature = "experimental-hcp")]
pub mod experimental {
    pub mod hcp {
        pub use crate::builder::HcpEngineBuilder;
        pub use crate::contract::*;
        pub use crate::engine::{HcpEngine, HcpRunStats};
        pub use crate::traits::{HcpProblem, SummaryApply};

        pub mod problems {
            pub use crate::problems::edit_distance;
            pub use crate::problems::lcs;
            pub use crate::problems::nw_affine;
            pub use crate::problems::nw_align;
            pub use crate::problems::semiglobal;
            pub use crate::problems::smith_waterman;
        }
    }
}
