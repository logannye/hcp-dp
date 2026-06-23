use serde_json::Value;
use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_hcp-align")
}

fn run(args: &[&str]) -> Output {
    Command::new(bin())
        .args(args)
        .output()
        .expect("hcp-align must run")
}

fn temp_file(name: &str, content: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock must be after epoch")
        .as_nanos();
    path.push(format!("hcp-dp-{name}-{}-{nanos}.txt", std::process::id()));
    fs::write(&path, content).expect("temp file must be writable");
    path
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("stdout must be utf8")
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr must be utf8")
}

#[test]
fn raw_json_full_verification_still_works() {
    let output = run(&[
        "global-linear",
        "--query",
        "GATTACA",
        "--target",
        "GCATGCU",
        "--match",
        "1",
        "--mismatch-penalty",
        "1",
        "--gap",
        "-1",
        "--verify",
        "--format",
        "json",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let json: Value = serde_json::from_str(&stdout(&output)).expect("valid json");
    assert_eq!(json["score"], 0);
    assert_eq!(json["path_score"], 0);
    assert_eq!(json["verification_status"], "full");
    assert_eq!(json["verified"], true);
    assert_eq!(json["pair_index"], 0);
    assert_eq!(json["query_id"], "query");
    assert_eq!(json["target_id"], "target");
}

#[test]
fn single_record_fasta_json_inherits_record_ids() {
    let query = temp_file("single-query", ">q1\nACGT\n");
    let target = temp_file("single-target", ">t1\nACGT\n");
    let output = run(&[
        "semiglobal-linear",
        "--query-file",
        query.to_str().unwrap(),
        "--target-file",
        target.to_str().unwrap(),
        "--verify",
        "--format",
        "json",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let json: Value = serde_json::from_str(&stdout(&output)).expect("valid json");
    assert_eq!(json["query_id"], "q1");
    assert_eq!(json["target_id"], "t1");
    assert_eq!(json["verification_status"], "full");
}

#[test]
fn wrapped_multi_record_fasta_pairwise_jsonl_works() {
    let query = temp_file("wrapped-query", ">q1\nAC\nGT\n>q2\nAA\nAA\n");
    let target = temp_file("wrapped-target", ">t1\nACGT\n>t2\nTTTT\n");
    let output = run(&[
        "edit-distance",
        "--query-file",
        query.to_str().unwrap(),
        "--target-file",
        target.to_str().unwrap(),
        "--verify",
        "--format",
        "jsonl",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let rows: Vec<Value> = stdout(&output)
        .lines()
        .map(|line| serde_json::from_str(line).expect("valid jsonl"))
        .collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["pair_index"], 0);
    assert_eq!(rows[0]["query_id"], "q1");
    assert_eq!(rows[0]["target_id"], "t1");
    assert_eq!(rows[0]["distance"], 0);
    assert_eq!(rows[1]["pair_index"], 1);
    assert_eq!(rows[1]["distance"], 4);
}

#[test]
fn multi_record_fastq_pairwise_cigar_works() {
    let query = temp_file("fastq-query", "@q1\nAC\n+\n!!\n@q2\nAA\n+\n!!\n");
    let target = temp_file("fastq-target", "@t1\nAC\n+\n!!\n@t2\nAT\n+\n!!\n");
    let output = run(&[
        "global-linear",
        "--query-file",
        query.to_str().unwrap(),
        "--target-file",
        target.to_str().unwrap(),
        "--match",
        "1",
        "--mismatch-penalty",
        "1",
        "--gap",
        "-1",
        "--format",
        "cigar",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 3);
    assert!(lines[0].contains("pair_index\tquery_id\ttarget_id"));
    assert!(lines[1].contains("\tq1\tt1\t"));
    assert!(lines[2].contains("\tq2\tt2\t"));
}

#[test]
fn tsv_output_has_stable_header_and_alignment_strings_can_be_requested() {
    let output = run(&[
        "local-linear",
        "--query",
        "ACACACTA",
        "--target",
        "AGCACACA",
        "--match",
        "2",
        "--mismatch-penalty",
        "1",
        "--gap",
        "-2",
        "--show-alignment",
        "--format",
        "tsv",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2);
    assert!(lines[0].starts_with("pair_index\tquery_id\ttarget_id"));

    let json = run(&[
        "local-linear",
        "--query",
        "ACACACTA",
        "--target",
        "AGCACACA",
        "--show-alignment",
        "--format",
        "json",
    ]);
    assert!(json.status.success(), "{}", stderr(&json));
    let parsed: Value = serde_json::from_str(&stdout(&json)).expect("valid json");
    assert!(parsed["aligned_query"]
        .as_str()
        .is_some_and(|s| !s.is_empty()));
    assert!(parsed["aligned_target"]
        .as_str()
        .is_some_and(|s| !s.is_empty()));
}

#[test]
fn mismatched_record_counts_fail() {
    let query = temp_file("mismatch-query", ">q1\nAC\n>q2\nGT\n");
    let target = temp_file("mismatch-target", ">t1\nAC\n");
    let output = run(&[
        "edit-distance",
        "--query-file",
        query.to_str().unwrap(),
        "--target-file",
        target.to_str().unwrap(),
    ]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("equal record counts"));
}

#[test]
fn verify_limit_can_report_path_only() {
    let output = run(&[
        "global-linear",
        "--query",
        "ACGT",
        "--target",
        "ACGT",
        "--verify",
        "--verify-limit",
        "1",
        "--format",
        "json",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let json: Value = serde_json::from_str(&stdout(&output)).expect("valid json");
    assert_eq!(json["verification_status"], "path_only");
    assert!(json["verified"].is_null());
}
