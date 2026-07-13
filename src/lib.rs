//! Exact reference alignments, independent verification, and backend comparison.
//!
//! The supported v0.2 API is centered on [`solve`], [`verify`], and [`compare`].
//! It uses one canonical SAM-oriented CIGAR model across JSON, JSONL, PAF, and
//! SAM and verifies objective equivalence rather than requiring identical
//! traceback tie-breaking.
//!
//! The original height-compressed DP engine is available only through the
//! opt-in `experimental-hcp` feature and carries no v0.2 stability guarantee.

/// Version of the `hcp-align` oracle implementation in this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

mod alignment;
mod batch;
#[allow(dead_code)]
mod blocks;
#[cfg(feature = "experimental-hcp")]
mod builder;
mod cli;
#[cfg(feature = "experimental-hcp")]
mod contract;
#[allow(dead_code)]
mod engine;
mod oracle;
#[allow(dead_code)]
mod problems;
mod record_io;
mod scoring;
pub mod sequence_io;
mod sequence_store;
mod traits;
mod utils;

pub use crate::batch::{
    BatchOptions, BatchSummary, ErrorContext, ErrorDetail, ErrorRecord, ErrorStage, OracleRecord,
};
pub use crate::oracle::{
    compare, record_to_paf, record_to_sam, records_from_json, records_from_jsonl, records_from_paf,
    records_from_sam, records_to_json, records_to_jsonl, records_to_sam, solve, verify,
    verify_integrity, AlignmentMode, AlignmentRecord, AlignmentResult, AlignmentSpan, CheckResult,
    Cigar, CigarOp, CigarRun, ComparisonInputs, ComparisonReport, GapModel, IntegrityFingerprint,
    Metrics, Objective, ObjectiveKind, PairIdentity, ProblemSpec, Producer, ScoringScheme,
    SequenceIdentity, SolveOptions, SolveRequest, SolveScope, SubstitutionScheme, TraceRecord,
    VerificationLevel, VerificationReport, VerificationRequest, VerificationStatus,
    VerificationSummary, DEFAULT_MAX_CELLS, SCHEMA_VERSION,
};
pub use crate::scoring::SubstitutionMatrix;

/// Internal entry point used by the packaged `hcp-align` binary.
///
/// This is not part of the supported Rust API. Library users should call
/// [`solve`], [`verify`], or [`compare`] directly.
#[doc(hidden)]
pub use crate::cli::main as __hcp_align_main;

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
