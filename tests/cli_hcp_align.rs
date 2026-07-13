use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_hcp-align")
}

fn run(args: &[&str]) -> Output {
    Command::new(bin()).args(args).output().expect("CLI runs")
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("UTF-8 stdout")
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("UTF-8 stderr")
}

fn temp_file(name: &str, content: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("hcp-dp-v2-{nonce}-{name}"));
    fs::write(&path, content).expect("write temp file");
    path
}

#[test]
fn version_reports_v02() {
    let output = run(&["--version"]);
    assert!(output.status.success());
    assert!(stdout(&output).contains("0.2.0-alpha.2"));
}

#[test]
fn solve_edit_distance_emits_v2_oracle_record() {
    let output = run(&[
        "solve",
        "edit-distance",
        "--query",
        "kitten",
        "--target",
        "sitting",
        "--cross-check",
        "--integrity",
        "--format",
        "json",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let record: Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(record["schema_version"], "hcp-align.v2");
    assert_eq!(record["record_type"], "alignment");
    assert_eq!(record["alignment"]["objective"]["value"], 3);
    assert_eq!(record["alignment"]["trace"]["cigar"], "1X3=1X1=1D");
    assert_eq!(record["verification"]["status"], "verified");
    assert_eq!(record["integrity"]["hash_algorithm"], "sha256");
}

#[test]
fn canonical_cigar_uses_sam_insertion_deletion_orientation() {
    let insertion = run(&[
        "solve",
        "edit-distance",
        "--query",
        "ACG",
        "--target",
        "AG",
        "--format",
        "json",
    ]);
    let deletion = run(&[
        "solve",
        "edit-distance",
        "--query",
        "AG",
        "--target",
        "ACG",
        "--format",
        "json",
    ]);
    let insertion: Value = serde_json::from_str(&stdout(&insertion)).unwrap();
    let deletion: Value = serde_json::from_str(&stdout(&deletion)).unwrap();
    assert!(insertion["alignment"]["trace"]["cigar"]
        .as_str()
        .unwrap()
        .contains('I'));
    assert!(deletion["alignment"]["trace"]["cigar"]
        .as_str()
        .unwrap()
        .contains('D'));
}

#[test]
fn every_supported_mode_cross_checks() {
    let cases: &[(&[&str], i64)] = &[
        (&["global-linear", "GATTACA", "GCATGCU"], 2),
        (&["global-affine", "ACB", "A"], -3),
        (&["local-linear", "ACACACTA", "AGCACACA"], 10),
        (&["semiglobal-linear", "ACGT", "TTACGTTT"], 8),
    ];
    for (case, expected) in cases {
        let output = run(&[
            "solve",
            case[0],
            "--query",
            case[1],
            "--target",
            case[2],
            "--cross-check",
            "--format",
            "json",
        ]);
        assert!(output.status.success(), "{}", stderr(&output));
        let value: Value = serde_json::from_str(&stdout(&output)).unwrap();
        assert_eq!(value["alignment"]["objective"]["value"], *expected);
        assert_eq!(value["verification"]["status"], "verified");
    }
}

#[test]
fn score_scope_omits_traceback() {
    let output = run(&[
        "solve",
        "edit-distance",
        "--query",
        "kitten",
        "--target",
        "sitting",
        "--scope",
        "score",
        "--format",
        "json",
    ]);
    let value: Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(value["producer"]["backend"], "myers");
    assert!(value["alignment"].get("trace").is_none());

    let output = run(&[
        "solve",
        "global-linear",
        "--query",
        "GATTACA",
        "--target",
        "GCATGCU",
        "--scope",
        "score",
        "--cross-check",
        "--format",
        "json",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let value: Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(value["producer"]["backend"], "hcp");
    assert_eq!(value["verification"]["status"], "verified");
    assert!(value["alignment"].get("trace").is_none());
}

#[test]
fn json_verify_checks_exact_optimality() {
    let record = temp_file("record.json", "");
    let solve = run(&[
        "solve",
        "edit-distance",
        "--query",
        "kitten",
        "--target",
        "sitting",
        "--format",
        "json",
        "--output",
        record.to_str().unwrap(),
    ]);
    assert!(solve.status.success(), "{}", stderr(&solve));
    let output = run(&[
        "verify",
        "--alignment",
        record.to_str().unwrap(),
        "--format",
        "json",
        "--query",
        "kitten",
        "--target",
        "sitting",
        "--output-format",
        "json",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let value: Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(value["record_type"], "verification");
    assert_eq!(value["verification"]["status"], "verified");
}

#[test]
fn work_limit_is_exit_three_and_inconclusive() {
    let record = temp_file("limited.json", "");
    assert!(run(&[
        "solve",
        "edit-distance",
        "--query",
        "kitten",
        "--target",
        "sitting",
        "--format",
        "json",
        "--output",
        record.to_str().unwrap(),
    ])
    .status
    .success());
    let output = run(&[
        "verify",
        "--alignment",
        record.to_str().unwrap(),
        "--format",
        "json",
        "--query",
        "kitten",
        "--target",
        "sitting",
        "--max-cells",
        "1",
        "--output-format",
        "json",
    ]);
    assert_eq!(output.status.code(), Some(3));
    let value: Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(value["verification"]["status"], "inconclusive");
}

#[test]
fn mutated_objective_is_exit_one_with_specific_check() {
    let original = run(&[
        "solve",
        "edit-distance",
        "--query",
        "kitten",
        "--target",
        "sitting",
        "--format",
        "json",
    ]);
    let mut value: Value = serde_json::from_str(&stdout(&original)).unwrap();
    value["alignment"]["objective"]["value"] = Value::from(4);
    let record = temp_file(
        "mutated.json",
        &serde_json::to_string_pretty(&value).unwrap(),
    );
    let output = run(&[
        "verify",
        "--alignment",
        record.to_str().unwrap(),
        "--format",
        "json",
        "--query",
        "kitten",
        "--target",
        "sitting",
        "--output-format",
        "json",
    ]);
    assert_eq!(output.status.code(), Some(1));
    let report: Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(report["verification"]["status"], "invalid");
    assert!(report["verification"]["checks"]
        .as_array()
        .unwrap()
        .iter()
        .any(|check| check["name"] == "realized_objective" && check["passed"] == false));
}

#[test]
fn paf_round_trip_verifies() {
    let paf = temp_file("record.paf", "");
    assert!(run(&[
        "solve",
        "edit-distance",
        "--query",
        "kitten",
        "--target",
        "sitting",
        "--format",
        "paf",
        "--output",
        paf.to_str().unwrap(),
    ])
    .status
    .success());
    let output = run(&[
        "verify",
        "--alignment",
        paf.to_str().unwrap(),
        "--format",
        "paf",
        "--query",
        "kitten",
        "--target",
        "sitting",
        "--mode",
        "edit-distance",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(stdout(&output).contains("Verified"));
}

#[test]
fn sam_round_trip_verifies() {
    let sam = temp_file("record.sam", "");
    assert!(run(&[
        "solve",
        "edit-distance",
        "--query",
        "kitten",
        "--target",
        "sitting",
        "--format",
        "sam",
        "--output",
        sam.to_str().unwrap(),
    ])
    .status
    .success());
    let output = run(&[
        "verify",
        "--alignment",
        sam.to_str().unwrap(),
        "--format",
        "sam",
        "--query",
        "kitten",
        "--target",
        "sitting",
        "--mode",
        "edit-distance",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
}

#[test]
fn external_sam_m_is_expanded_against_sources() {
    let sam = temp_file(
        "external.sam",
        "@HD\tVN:1.6\n@SQ\tSN:target\tLN:4\nquery\t0\ttarget\t1\t255\t4M\t*\t0\t0\tACGT\t*\tNM:i:0\n",
    );
    let output = run(&[
        "verify",
        "--alignment",
        sam.to_str().unwrap(),
        "--format",
        "sam",
        "--query",
        "ACGT",
        "--target",
        "ACGT",
        "--mode",
        "edit-distance",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
}

#[test]
fn paf_requires_explicit_problem() {
    let paf = temp_file(
        "missing-mode.paf",
        "query\t1\t0\t1\t+\ttarget\t1\t0\t1\t1\t1\t255\tcg:Z:1=\tNM:i:0\n",
    );
    let output = run(&[
        "verify",
        "--alignment",
        paf.to_str().unwrap(),
        "--format",
        "paf",
        "--query",
        "A",
        "--target",
        "A",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("required for PAF"));

    let output = run(&[
        "verify",
        "--alignment",
        paf.to_str().unwrap(),
        "--format",
        "paf",
        "--query",
        "A",
        "--target",
        "A",
        "--mode",
        "global-linear",
        "--gap",
        "-2",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("--match is required"));
}

#[test]
fn compare_accepts_equivalent_records() {
    let record = temp_file("compare.json", "");
    assert!(run(&[
        "solve",
        "edit-distance",
        "--query",
        "AA",
        "--target",
        "A",
        "--format",
        "json",
        "--output",
        record.to_str().unwrap(),
    ])
    .status
    .success());
    let output = run(&[
        "compare",
        "--left",
        record.to_str().unwrap(),
        "--left-format",
        "json",
        "--right",
        record.to_str().unwrap(),
        "--right-format",
        "json",
        "--query",
        "AA",
        "--target",
        "A",
        "--output-format",
        "json",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let value: Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(value["equivalent"], true);
    assert_eq!(value["inputs"]["query"], "AA");
    assert_eq!(value["inputs"]["target"], "A");
    assert!(value["left"].get("metrics").is_none());
}

#[test]
fn integrity_mutation_is_a_named_invalid_check() {
    let solved = run(&[
        "solve",
        "edit-distance",
        "--query",
        "ACGT",
        "--target",
        "ACGT",
        "--integrity",
        "--format",
        "json",
    ]);
    let mut value: Value = serde_json::from_str(&stdout(&solved)).unwrap();
    value["producer"]["backend"] = Value::from("tampered");
    let record = temp_file(
        "tampered-integrity.json",
        &serde_json::to_string(&value).unwrap(),
    );
    let output = run(&[
        "verify",
        "--alignment",
        record.to_str().unwrap(),
        "--format",
        "json",
        "--query",
        "ACGT",
        "--target",
        "ACGT",
        "--output-format",
        "json",
    ]);
    assert_eq!(output.status.code(), Some(1));
    let report: Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert!(report["verification"]["checks"]
        .as_array()
        .unwrap()
        .iter()
        .any(|check| check["name"] == "integrity_fingerprint" && check["passed"] == false));
}

#[test]
fn paf_rejects_corrupt_match_count_and_nm_tag() {
    let solved = run(&[
        "solve",
        "edit-distance",
        "--query",
        "ACGT",
        "--target",
        "ACGA",
        "--format",
        "paf",
    ]);
    assert!(solved.status.success());
    let line = stdout(&solved);
    let mut fields: Vec<String> = line.trim().split('\t').map(str::to_string).collect();
    fields[9] = "4".to_string();
    let bad_matches = temp_file("bad-matches.paf", &(fields.join("\t") + "\n"));
    let output = run(&[
        "verify",
        "--alignment",
        bad_matches.to_str().unwrap(),
        "--format",
        "paf",
        "--query",
        "ACGT",
        "--target",
        "ACGA",
        "--mode",
        "edit-distance",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("matching bases"));

    let mut fields: Vec<String> = line.trim().split('\t').map(str::to_string).collect();
    let nm = fields
        .iter()
        .position(|field| field.starts_with("NM:i:"))
        .unwrap();
    fields[nm] = "NM:i:2".to_string();
    let bad_nm = temp_file("bad-nm.paf", &(fields.join("\t") + "\n"));
    let output = run(&[
        "verify",
        "--alignment",
        bad_nm.to_str().unwrap(),
        "--format",
        "paf",
        "--query",
        "ACGT",
        "--target",
        "ACGA",
        "--mode",
        "edit-distance",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("NM:i:2"));
}

#[test]
fn sam_validates_sequence_and_soft_clipping() {
    let mismatched = temp_file(
        "bad-seq.sam",
        "@SQ\tSN:target\tLN:4\nquery\t0\ttarget\t1\t255\t4=\t*\t0\t0\tTGCA\t*\tNM:i:0\n",
    );
    let output = run(&[
        "verify",
        "--alignment",
        mismatched.to_str().unwrap(),
        "--format",
        "sam",
        "--query",
        "ACGT",
        "--target",
        "ACGT",
        "--mode",
        "edit-distance",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("SEQ does not match"));

    let clipped = temp_file(
        "soft-clipped.sam",
        "@SQ\tSN:target\tLN:4\nquery\t0\ttarget\t1\t255\t1S4=1S\t*\t0\t0\tTACGTG\t*\tAS:i:8\tNM:i:0\n",
    );
    let output = run(&[
        "verify",
        "--alignment",
        clipped.to_str().unwrap(),
        "--format",
        "sam",
        "--query",
        "TACGTG",
        "--target",
        "ACGT",
        "--mode",
        "local-linear",
        "--match",
        "2",
        "--mismatch-penalty",
        "1",
        "--gap",
        "-2",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
}

#[test]
fn duplicate_and_missing_source_ids_are_errors() {
    let queries = temp_file("duplicate.fa", ">q\nA\n>q\nA\n");
    let targets = temp_file("targets.fa", ">t1\nA\n>t2\nA\n");
    let record = temp_file("empty.json", "[]\n");
    let output = run(&[
        "verify",
        "--alignment",
        record.to_str().unwrap(),
        "--format",
        "json",
        "--query-file",
        queries.to_str().unwrap(),
        "--target-file",
        targets.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("duplicate query id 'q'"));

    let sam = temp_file(
        "missing-id.sam",
        "other\t0\ttarget\t1\t255\t1=\t*\t0\t0\tA\t*\tNM:i:0\n",
    );
    let output = run(&[
        "verify",
        "--alignment",
        sam.to_str().unwrap(),
        "--format",
        "sam",
        "--query",
        "A",
        "--target",
        "A",
        "--mode",
        "edit-distance",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("query id 'other' is missing"));
}

#[test]
fn verification_source_collections_do_not_require_positional_pairing() {
    let solve_queries = temp_file("one-query.fa", ">q\nACGT\n");
    let solve_targets = temp_file("one-target.fa", ">t\nACGT\n");
    let record = temp_file("id-record.json", "");
    let solved = run(&[
        "solve",
        "edit-distance",
        "--query-file",
        solve_queries.to_str().unwrap(),
        "--target-file",
        solve_targets.to_str().unwrap(),
        "--format",
        "json",
        "--output",
        record.to_str().unwrap(),
    ]);
    assert!(solved.status.success(), "{}", stderr(&solved));

    let verification_targets = temp_file("reference-set.fa", ">unused\nTTTT\n>t\nACGT\n");
    let output = run(&[
        "verify",
        "--alignment",
        record.to_str().unwrap(),
        "--format",
        "json",
        "--query-file",
        solve_queries.to_str().unwrap(),
        "--target-file",
        verification_targets.to_str().unwrap(),
        "--output-format",
        "json",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
}

#[test]
fn jsonl_continues_with_an_ordered_error_record() {
    let solved = run(&[
        "solve",
        "edit-distance",
        "--query",
        "AC",
        "--target",
        "AC",
        "--format",
        "json",
    ]);
    assert!(solved.status.success());
    let valid = stdout(&solved);
    let records = temp_file(
        "recoverable.jsonl",
        &format!("{}\n{{not-json}}\n{}\n", valid.trim(), valid.trim()),
    );
    let output = run(&[
        "verify",
        "--alignment",
        records.to_str().unwrap(),
        "--format",
        "jsonl",
        "--query",
        "AC",
        "--target",
        "AC",
        "--output-format",
        "jsonl",
    ]);
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    let lines: Vec<Value> = stdout(&output)
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0]["record_type"], "verification");
    assert_eq!(lines[1]["record_type"], "error");
    assert_eq!(lines[1]["error"]["stage"], "record_parse");
    assert_eq!(lines[1]["context"]["pair_index"], 1);
    assert_eq!(lines[1]["context"]["source_line"], 2);
    assert_eq!(lines[2]["record_type"], "verification");
}

#[test]
fn non_jsonl_sidecar_enables_record_continuation() {
    let paf = temp_file(
        "recoverable.paf",
        concat!(
            "broken\n",
            "query\t2\t0\t2\t+\ttarget\t2\t0\t2\t2\t2\t255\tcg:Z:2=\tNM:i:0\n"
        ),
    );
    let errors = temp_file("recoverable-errors.jsonl", "");
    let output = run(&[
        "verify",
        "--alignment",
        paf.to_str().unwrap(),
        "--format",
        "paf",
        "--query",
        "AC",
        "--target",
        "AC",
        "--mode",
        "edit-distance",
        "--errors",
        errors.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    assert!(stdout(&output).contains("Verified"));
    let error: Value = serde_json::from_str(&fs::read_to_string(errors).unwrap()).unwrap();
    assert_eq!(error["error"]["code"], "too_few_fields");
}

#[test]
fn unequal_sequence_counts_emit_one_terminal_jsonl_error() {
    let queries = temp_file("unequal-query.fa", ">q1\nA\n>q2\nC\n");
    let targets = temp_file("unequal-target.fa", ">t1\nA\n");
    let output = run(&[
        "solve",
        "edit-distance",
        "--query-file",
        queries.to_str().unwrap(),
        "--target-file",
        targets.to_str().unwrap(),
        "--format",
        "jsonl",
    ]);
    assert_eq!(output.status.code(), Some(2));
    let lines: Vec<Value> = stdout(&output)
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[1]["error"]["code"], "pair_count_mismatch");
    assert_eq!(lines[1]["error"]["recoverable"], false);
}

#[test]
fn solve_recovers_after_a_record_scoped_fastq_error() {
    let queries = temp_file(
        "recoverable-query.fastq",
        "@q0\nAC\n+\n!!!\n@q1\nTG\n+\n##\n",
    );
    let targets = temp_file("recoverable-target.fa", ">t0\nAC\n>t1\nTG\n");
    let output = run(&[
        "solve",
        "edit-distance",
        "--query-file",
        queries.to_str().unwrap(),
        "--target-file",
        targets.to_str().unwrap(),
        "--format",
        "jsonl",
    ]);
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    let lines: Vec<Value> = stdout(&output)
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["record_type"], "error");
    assert_eq!(lines[0]["error"]["recoverable"], true);
    assert_eq!(lines[1]["record_type"], "alignment");
    assert_eq!(lines[1]["pair"]["index"], 1);
    assert_eq!(lines[1]["pair"]["query"]["id"], "q1");
    assert_eq!(lines[1]["pair"]["target"]["id"], "t1");
}

#[test]
fn fasta_batch_emits_jsonl() {
    let queries = temp_file("queries.fa", ">q1\nACGT\n>q2\nAAAA\n");
    let targets = temp_file("targets.fa", ">t1\nACGT\n>t2\nTTTT\n");
    let output = run(&[
        "solve",
        "edit-distance",
        "--query-file",
        queries.to_str().unwrap(),
        "--target-file",
        targets.to_str().unwrap(),
        "--format",
        "jsonl",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let rows: Vec<Value> = stdout(&output)
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["pair"]["query"]["id"], "q1");
    assert_eq!(rows[1]["alignment"]["objective"]["value"], 4);
}

#[test]
fn legacy_top_level_mode_is_rejected() {
    let output = run(&["edit-distance", "--query", "A", "--target", "A"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("unrecognized subcommand"));
}
