use serde_json::{json, Value};
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

fn with_normalized_elapsed(mut value: Value) -> Value {
    value["elapsed_ms"] = json!("<elapsed_ms>");
    value
}

fn normalize_tsv_elapsed(text: &str) -> String {
    text.lines()
        .map(|line| {
            let mut fields: Vec<&str> = line.split('\t').collect();
            if fields
                .last()
                .is_some_and(|last| last.parse::<f64>().is_ok())
            {
                let last_idx = fields.len() - 1;
                fields[last_idx] = "<elapsed_ms>";
            }
            fields.join("\t")
        })
        .collect::<Vec<_>>()
        .join("\n")
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

#[test]
fn json_output_has_stable_golden_contract() {
    let output = run(&[
        "edit-distance",
        "--query",
        "ACGT",
        "--target",
        "ACGT",
        "--verify",
        "--block-size",
        "1",
        "--format",
        "json",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let parsed: Value = serde_json::from_str(&stdout(&output)).expect("valid json");
    assert_eq!(
        with_normalized_elapsed(parsed),
        json!({
            "pair_index": 0,
            "query_id": "query",
            "target_id": "target",
            "mode": "edit-distance",
            "score": null,
            "distance": 0,
            "path_score": 0,
            "verification_status": "full",
            "verified": true,
            "query_start": 0,
            "query_end": 4,
            "target_start": 0,
            "target_end": 4,
            "cigar": "4=",
            "operations": [
                {"op": "match", "query_pos": 0, "target_pos": 0, "query_base": "A", "target_base": "A"},
                {"op": "match", "query_pos": 1, "target_pos": 1, "query_base": "C", "target_base": "C"},
                {"op": "match", "query_pos": 2, "target_pos": 2, "query_base": "G", "target_base": "G"},
                {"op": "match", "query_pos": 3, "target_pos": 3, "query_base": "T", "target_base": "T"}
            ],
            "block_size": 1,
            "elapsed_ms": "<elapsed_ms>"
        })
    );
}

#[test]
fn jsonl_output_has_stable_batch_golden_contract() {
    let query = temp_file("golden-jsonl-query", ">q1\nAC\n>q2\nAA\n");
    let target = temp_file("golden-jsonl-target", ">t1\nAC\n>t2\nAT\n");
    let output = run(&[
        "edit-distance",
        "--query-file",
        query.to_str().unwrap(),
        "--target-file",
        target.to_str().unwrap(),
        "--verify",
        "--block-size",
        "1",
        "--format",
        "jsonl",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let rows: Vec<Value> = stdout(&output)
        .lines()
        .map(|line| with_normalized_elapsed(serde_json::from_str(line).expect("valid jsonl")))
        .collect();
    assert_eq!(
        rows,
        json!([
            {
                "pair_index": 0,
                "query_id": "q1",
                "target_id": "t1",
                "mode": "edit-distance",
                "score": null,
                "distance": 0,
                "path_score": 0,
                "verification_status": "full",
                "verified": true,
                "query_start": 0,
                "query_end": 2,
                "target_start": 0,
                "target_end": 2,
                "cigar": "2=",
                "operations": [
                    {"op": "match", "query_pos": 0, "target_pos": 0, "query_base": "A", "target_base": "A"},
                    {"op": "match", "query_pos": 1, "target_pos": 1, "query_base": "C", "target_base": "C"}
                ],
                "block_size": 1,
                "elapsed_ms": "<elapsed_ms>"
            },
            {
                "pair_index": 1,
                "query_id": "q2",
                "target_id": "t2",
                "mode": "edit-distance",
                "score": null,
                "distance": 1,
                "path_score": 1,
                "verification_status": "full",
                "verified": true,
                "query_start": 0,
                "query_end": 2,
                "target_start": 0,
                "target_end": 2,
                "cigar": "1=1X",
                "operations": [
                    {"op": "match", "query_pos": 0, "target_pos": 0, "query_base": "A", "target_base": "A"},
                    {"op": "mismatch", "query_pos": 1, "target_pos": 1, "query_base": "A", "target_base": "T"}
                ],
                "block_size": 1,
                "elapsed_ms": "<elapsed_ms>"
            }
        ])
        .as_array()
        .expect("expected golden array")
        .clone()
    );
}

#[test]
fn tsv_and_cigar_outputs_have_stable_golden_contracts() {
    let tsv = run(&[
        "edit-distance",
        "--query",
        "ACGT",
        "--target",
        "ACGT",
        "--verify",
        "--block-size",
        "1",
        "--format",
        "tsv",
    ]);
    assert!(tsv.status.success(), "{}", stderr(&tsv));
    assert_eq!(
        normalize_tsv_elapsed(&stdout(&tsv)),
        "pair_index\tquery_id\ttarget_id\tmode\tscore\tdistance\tpath_score\tverification_status\tquery_start\tquery_end\ttarget_start\ttarget_end\tcigar\tblock_size\telapsed_ms\n0\tquery\ttarget\tedit-distance\t\t0\t0\tfull\t0\t4\t0\t4\t4=\t1\t<elapsed_ms>"
    );

    let cigar = run(&[
        "edit-distance",
        "--query",
        "ACGT",
        "--target",
        "ACGT",
        "--verify",
        "--block-size",
        "1",
        "--format",
        "cigar",
    ]);
    assert!(cigar.status.success(), "{}", stderr(&cigar));
    assert_eq!(
        stdout(&cigar).trim_end(),
        "pair_index\tquery_id\ttarget_id\tmode\tscore\tdistance\tpath_score\tverification_status\tquery_start\tquery_end\ttarget_start\ttarget_end\tcigar\n0\tquery\ttarget\tedit-distance\t\t0\t0\tfull\t0\t4\t0\t4\t4="
    );
}
