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

fn with_normalized_timings(mut value: Value) -> Value {
    value["elapsed_ms"] = json!("<elapsed_ms>");
    value["summary_build_ms"] = json!("<summary_build_ms>");
    value["reconstruction_ms"] = json!("<reconstruction_ms>");
    value["verification_ms"] = json!("<verification_ms>");
    value
}

fn normalize_tsv_timings(text: &str) -> String {
    text.lines()
        .map(|line| {
            let mut fields: Vec<&str> = line.split('\t').collect();
            let replacements = [
                "<summary_build_ms>",
                "<reconstruction_ms>",
                "<verification_ms>",
                "<elapsed_ms>",
            ];
            if fields.len() >= replacements.len() {
                let start = fields.len() - replacements.len();
                for (idx, replacement) in replacements.iter().enumerate() {
                    if fields[start + idx].parse::<f64>().is_ok() {
                        fields[start + idx] = replacement;
                    }
                }
            }
            fields.join("\t")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn version_flag_reports_package_version() {
    let output = run(&["--version"]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert_eq!(
        stdout(&output).trim(),
        format!("hcp-align {}", env!("CARGO_PKG_VERSION"))
    );
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
fn operation_detail_controls_json_trace_size() {
    let summary = run(&[
        "edit-distance",
        "--query",
        "ACGT",
        "--target",
        "ACGA",
        "--format",
        "json",
    ]);
    assert!(summary.status.success(), "{}", stderr(&summary));
    let summary_json: Value = serde_json::from_str(&stdout(&summary)).expect("valid json");
    assert!(summary_json.get("operations").is_none());
    assert_eq!(summary_json["operation_counts"][0]["op"], "match");

    let none = run(&[
        "edit-distance",
        "--query",
        "ACGT",
        "--target",
        "ACGA",
        "--operation-detail",
        "none",
        "--format",
        "json",
    ]);
    assert!(none.status.success(), "{}", stderr(&none));
    let none_json: Value = serde_json::from_str(&stdout(&none)).expect("valid json");
    assert!(none_json.get("operations").is_none());
    assert!(none_json.get("operation_counts").is_none());

    let full = run(&[
        "edit-distance",
        "--query",
        "ACGT",
        "--target",
        "ACGA",
        "--operation-detail",
        "full",
        "--format",
        "json",
    ]);
    assert!(full.status.success(), "{}", stderr(&full));
    let full_json: Value = serde_json::from_str(&stdout(&full)).expect("valid json");
    assert_eq!(full_json["operations"].as_array().unwrap().len(), 4);
    assert_eq!(full_json["operation_counts"].as_array().unwrap().len(), 2);
}

#[test]
fn output_flag_writes_to_file() {
    let output_path = temp_file("output-json", "");
    let output = run(&[
        "edit-distance",
        "--query",
        "kitten",
        "--target",
        "sitting",
        "--verify",
        "--format",
        "json",
        "--output",
        output_path.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert_eq!(stdout(&output), "");
    let parsed: Value =
        serde_json::from_str(&fs::read_to_string(output_path).expect("output readable"))
            .expect("valid json");
    assert_eq!(parsed["distance"], 3);
}

#[test]
fn edit_distance_alternate_engines_return_verified_paths() {
    let auto = run(&[
        "edit-distance",
        "--query",
        "ACGTACGT",
        "--target",
        "ACGTTCGT",
        "--verify",
        "--format",
        "json",
    ]);
    assert!(auto.status.success(), "{}", stderr(&auto));
    let auto_json: Value = serde_json::from_str(&stdout(&auto)).expect("valid json");
    assert_eq!(auto_json["distance"], 1);
    assert_eq!(auto_json["path_score"], 1);
    assert_eq!(auto_json["verification_status"], "full");
    assert_eq!(auto_json["backend"], "adaptive-banded");
    assert_eq!(auto_json["block_size"], 0);

    let adaptive = run(&[
        "edit-distance",
        "--engine",
        "adaptive-banded",
        "--query",
        "kitten",
        "--target",
        "sitting",
        "--verify",
        "--format",
        "json",
    ]);
    assert!(adaptive.status.success(), "{}", stderr(&adaptive));
    let adaptive_json: Value = serde_json::from_str(&stdout(&adaptive)).expect("valid json");
    assert_eq!(adaptive_json["distance"], 3);
    assert_eq!(adaptive_json["path_score"], 3);
    assert_eq!(adaptive_json["verification_status"], "full");
    assert_eq!(adaptive_json["backend"], "adaptive-banded");
    assert_eq!(adaptive_json["block_size"], 0);

    let hcp_linear = run(&[
        "edit-distance",
        "--engine",
        "hcp-linear",
        "--query",
        "kitten",
        "--target",
        "sitting",
        "--verify",
        "--format",
        "json",
    ]);
    assert!(hcp_linear.status.success(), "{}", stderr(&hcp_linear));
    let hcp_linear_json: Value = serde_json::from_str(&stdout(&hcp_linear)).expect("valid json");
    assert_eq!(hcp_linear_json["distance"], 3);
    assert_eq!(hcp_linear_json["path_score"], 3);
    assert_eq!(hcp_linear_json["verification_status"], "full");
    assert_eq!(hcp_linear_json["backend"], "hcp-linear");
    assert_eq!(hcp_linear_json["block_size"], 1);
}

#[test]
fn adaptive_banded_engine_rejects_block_size_override() {
    let output = run(&[
        "edit-distance",
        "--engine",
        "adaptive-banded",
        "--query",
        "kitten",
        "--target",
        "sitting",
        "--block-size",
        "1",
    ]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("--block-size applies only to HCP"));
}

#[test]
fn auto_engine_rejects_block_size_override() {
    let output = run(&[
        "edit-distance",
        "--query",
        "kitten",
        "--target",
        "sitting",
        "--block-size",
        "1",
    ]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("--block-size requires an explicit HCP"));
}

#[test]
fn continue_on_error_emits_failed_batch_records() {
    let query = temp_file("continue-query", ">q1\nAC\n>q2\nAA\n");
    let target = temp_file("continue-target", ">t1\nAC\n>t2\nAT\n");
    let output = run(&[
        "edit-distance",
        "--query-file",
        query.to_str().unwrap(),
        "--target-file",
        target.to_str().unwrap(),
        "--engine",
        "hcp",
        "--block-size",
        "0",
        "--continue-on-error",
        "--format",
        "jsonl",
    ]);
    assert_eq!(output.status.code(), Some(1));
    let rows: Vec<Value> = stdout(&output)
        .lines()
        .map(|line| serde_json::from_str(line).expect("valid jsonl"))
        .collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["verification_status"], "failed");
    assert!(rows[0]["error"]
        .as_str()
        .is_some_and(|err| err.contains("block size")));
}

#[test]
fn progress_always_writes_to_stderr() {
    let query = temp_file("progress-query", ">q1\nAC\n>q2\nAA\n");
    let target = temp_file("progress-target", ">t1\nAC\n>t2\nAT\n");
    let output = run(&[
        "edit-distance",
        "--query-file",
        query.to_str().unwrap(),
        "--target-file",
        target.to_str().unwrap(),
        "--progress",
        "always",
        "--format",
        "jsonl",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(stderr(&output).contains("hcp-align: 2/2"));
}

#[test]
fn threads_one_works_without_parallel_feature() {
    let output = run(&[
        "edit-distance",
        "--engine",
        "hcp",
        "--query",
        "ACGT",
        "--target",
        "ACGT",
        "--threads",
        "1",
        "--format",
        "json",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
}

#[cfg(not(feature = "parallel"))]
#[test]
fn threads_above_one_requires_parallel_feature() {
    let output = run(&[
        "edit-distance",
        "--engine",
        "hcp",
        "--query",
        "ACGT",
        "--target",
        "ACGT",
        "--threads",
        "2",
        "--format",
        "json",
    ]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("parallel"));
}

#[cfg(feature = "parallel")]
#[test]
fn threads_above_one_works_with_parallel_feature() {
    let query = temp_file("threads-query", ">q1\nAC\n>q2\nAA\n");
    let target = temp_file("threads-target", ">t1\nAC\n>t2\nAT\n");
    let output = run(&[
        "edit-distance",
        "--query-file",
        query.to_str().unwrap(),
        "--target-file",
        target.to_str().unwrap(),
        "--threads",
        "2",
        "--format",
        "jsonl",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert_eq!(stdout(&output).lines().count(), 2);
}

#[test]
fn json_output_has_stable_golden_contract() {
    let output = run(&[
        "edit-distance",
        "--engine",
        "hcp",
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
        with_normalized_timings(parsed),
        json!({
            "schema_version": "hcp-align.v1",
            "engine": "hcp-dp",
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
            "operation_counts": [
                {"op": "match", "count": 4}
            ],
            "block_size": 1,
            "path_length": 5,
            "summary_build_ms": "<summary_build_ms>",
            "reconstruction_ms": "<reconstruction_ms>",
            "verification_ms": "<verification_ms>",
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
        "--engine",
        "hcp",
        "--verify",
        "--block-size",
        "1",
        "--format",
        "jsonl",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let rows: Vec<Value> = stdout(&output)
        .lines()
        .map(|line| with_normalized_timings(serde_json::from_str(line).expect("valid jsonl")))
        .collect();
    assert_eq!(
        rows,
        json!([
            {
                "schema_version": "hcp-align.v1",
                "engine": "hcp-dp",
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
                "operation_counts": [
                    {"op": "match", "count": 2}
                ],
                "block_size": 1,
                "path_length": 3,
                "summary_build_ms": "<summary_build_ms>",
                "reconstruction_ms": "<reconstruction_ms>",
                "verification_ms": "<verification_ms>",
                "elapsed_ms": "<elapsed_ms>"
            },
            {
                "schema_version": "hcp-align.v1",
                "engine": "hcp-dp",
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
                "operation_counts": [
                    {"op": "match", "count": 1},
                    {"op": "mismatch", "count": 1}
                ],
                "block_size": 1,
                "path_length": 3,
                "summary_build_ms": "<summary_build_ms>",
                "reconstruction_ms": "<reconstruction_ms>",
                "verification_ms": "<verification_ms>",
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
        "--engine",
        "hcp",
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
        normalize_tsv_timings(&stdout(&tsv)),
        "pair_index\tquery_id\ttarget_id\tmode\tscore\tdistance\tpath_score\tverification_status\tquery_start\tquery_end\ttarget_start\ttarget_end\tcigar\tblock_size\tpath_length\tsummary_build_ms\treconstruction_ms\tverification_ms\telapsed_ms\n0\tquery\ttarget\tedit-distance\t\t0\t0\tfull\t0\t4\t0\t4\t4=\t1\t5\t<summary_build_ms>\t<reconstruction_ms>\t<verification_ms>\t<elapsed_ms>"
    );

    let cigar = run(&[
        "edit-distance",
        "--engine",
        "hcp",
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
