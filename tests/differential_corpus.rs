use hcp_dp::{
    solve, AlignmentMode, ProblemSpec, ScoringScheme, SolveOptions, SolveRequest,
    VerificationStatus,
};
use serde::Deserialize;
use std::str::FromStr;

#[derive(Deserialize)]
struct Corpus {
    schema_version: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    mode: String,
    query: String,
    target: String,
    objective: i64,
}

#[test]
fn versioned_differential_corpus_matches_reference_objectives() {
    let corpus: Corpus =
        serde_json::from_str(include_str!("fixtures/differential-corpus-v1.json")).unwrap();
    assert_eq!(corpus.schema_version, "hcp-align.corpus.v1");
    for case in corpus.cases {
        let mode = AlignmentMode::from_str(&case.mode).unwrap();
        let record = solve(SolveRequest {
            pair_index: 0,
            query_id: case.name.clone(),
            target_id: format!("{}-target", case.name),
            query: case.query.into_bytes(),
            target: case.target.into_bytes(),
            problem: ProblemSpec {
                mode,
                scoring: ScoringScheme::default_for(mode),
            },
            options: SolveOptions {
                cross_check: true,
                ..SolveOptions::default()
            },
        })
        .unwrap_or_else(|error| panic!("corpus case {} failed: {error}", case.name));
        assert_eq!(
            record.alignment.objective.value, case.objective,
            "wrong objective for {}",
            case.name
        );
        assert_eq!(
            record.verification.status,
            VerificationStatus::Verified,
            "cross-check failed for {}",
            case.name
        );
    }
}
