//! Example: Needleman–Wunsch global alignment via HCP-DP.
//!
//! Run with:
//! `cargo run --example align`
//!
//! This uses the `nw_align` problem implementation as a reference.

use hcp_dp::{problems::nw_align::NwProblem, HcpEngine};

fn main() {
    let s = b"GATTACA";
    let t = b"GCATGCU";

    // Typical scoring: +1 match, -1 mismatch, -1 gap.
    // Here we pass `match_score = 1`, `mismatch_penalty = 1` (used as -1),
    // and `gap_penalty = -1` directly for flexibility.
    let problem = NwProblem::new(s, t, 1, 1, -1);
    let engine = HcpEngine::new(problem);

    let (score, path) = engine.run();

    println!("Global alignment score: {score}");
    println!("Path length (DP steps): {}", path.len());

    let (aln_s, aln_t) = materialize_alignment(s, t, &path);
    println!("S': {aln_s}");
    println!("T': {aln_t}");
}

/// Reconstruct aligned strings from a DP path of (i,j) coordinates.
///
/// This is a visualization helper for the example only.
fn materialize_alignment(s: &[u8], t: &[u8], path: &[(usize, usize)]) -> (String, String) {
    if path.is_empty() {
        return ("".into(), "".into());
    }
    let mut out_s = String::new();
    let mut out_t = String::new();

    let mut prev = path[0];
    for &cur in path.iter().skip(1) {
        match (
            cur.0 as isize - prev.0 as isize,
            cur.1 as isize - prev.1 as isize,
        ) {
            (1, 1) => {
                // diagonal: match or mismatch
                out_s.push(s[prev.0] as char);
                out_t.push(t[prev.1] as char);
            }
            (1, 0) => {
                // gap in t
                out_s.push(s[prev.0] as char);
                out_t.push('-');
            }
            (0, 1) => {
                // gap in s
                out_s.push('-');
                out_t.push(t[prev.1] as char);
            }
            _ => {
                // Shouldn't happen in valid NW, but be defensive.
                if cur.0 > prev.0 && cur.1 >= prev.1 {
                    out_s.push(s[prev.0] as char);
                    out_t.push(if cur.1 > prev.1 {
                        t[prev.1] as char
                    } else {
                        '-'
                    });
                }
            }
        };
        prev = cur;
    }

    (out_s, out_t)
}
