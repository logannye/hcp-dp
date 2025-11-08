//! Example: Longest Common Subsequence via the HCP-DP engine.
//!
//! Run with:
//! `cargo run --example lcs`

use hcp_dp::{problems::lcs::LcsProblem, HcpEngine};

fn main() {
    let s = b"ACCGGTCGAGTGCGCGGAAGCCGGCCGAA";
    let t = b"GTCGTTCGGAATGCCGTTGCTCTGTAAA";

    let problem = LcsProblem::new(s, t);
    let engine = HcpEngine::new(problem);

    let (cost, path) = engine.run();

    println!("LCS length: {cost}");
    println!("Reconstructed path length: {}", path.len());

    // Recover the actual LCS string from the path.
    let lcs = extract_lcs_from_path(s, t, &path);
    println!("LCS: {}", String::from_utf8_lossy(&lcs));
}

/// Given an LCS path (sequence of (i,j) DP states), extract one LCS.
fn extract_lcs_from_path(s: &[u8], t: &[u8], path: &[(usize, usize)]) -> Vec<u8> {
    let mut lcs = Vec::new();
    if path.is_empty() {
        return lcs;
    }

    let mut prev = path[0];
    for &cur in path.iter().skip(1) {
        // If both i and j advanced by 1 and chars match, it's part of the LCS.
        if cur.0 == prev.0 + 1 && cur.1 == prev.1 + 1 && s[prev.0] == t[prev.1] {
            lcs.push(s[prev.0]);
        }
        prev = cur;
    }
    lcs
}
