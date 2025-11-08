//! Longest Common Subsequence (LCS) as a height-compressible DP.
//!
//! This implementation demonstrates an exact, linear-space-style reconstruction
//! using the generic HCP-DP engine. It closely mirrors Hirschberg's algorithm,
//! but is structured through the [`HcpProblem`] trait.
//!
//! We treat layers as positions in `s` (0..=n), and the frontier at layer `i`
//! is the DP row for LCS(s[0..i], t[0..=m]).
//!
//! Boundaries are DP coordinates (row, col). The reconstruction returns a path
//! of (i,j) states along some optimal LCS path.

use crate::traits::HcpProblem;

#[derive(Clone)]
pub struct LcsProblem<'a> {
    pub s: &'a [u8],
    pub t: &'a [u8],
}

/// Frontier row: LCS length for all prefixes of `t` at a fixed `i`.
#[derive(Clone, Debug)]
pub struct LcsFrontier {
    pub scores: Vec<u32>, // length = t.len() + 1
}

/// Summary Σ[a,b]: for this simple implementation, we store:
/// - the ending frontier at layer b.
///
/// More sophisticated summaries can cache additional info, but this suffices
/// for correctness since `choose_boundary` is allowed to recompute locally.
#[derive(Clone, Debug)]
pub struct LcsSummary {
    pub end_frontier: LcsFrontier,
}

/// Boundary condition: fixed DP cell (row, col).
///
/// Semantics:
/// - `row` in [0..=s.len()]
/// - `col` in [0..=t.len()]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LcsBoundary {
    pub row: usize,
    pub col: usize,
}

/// State along the reconstructed path: also a DP cell (row, col).
pub type LcsState = (usize, usize);

impl<'a> LcsProblem<'a> {
    pub fn new(s: &'a [u8], t: &'a [u8]) -> Self {
        Self { s, t }
    }

    fn n(&self) -> usize {
        self.s.len()
    }

    fn m(&self) -> usize {
        self.t.len()
    }
}

impl<'a> HcpProblem for LcsProblem<'a> {
    type State = LcsState;
    type Frontier = LcsFrontier;
    type Summary = LcsSummary;
    type Boundary = LcsBoundary;
    type Cost = u32;

    fn num_layers(&self) -> usize {
        // T = number of "steps": each consumes one char from s
        self.n()
    }

    fn init_frontier(&self) -> Self::Frontier {
        // i = 0: LCS(ε, t[0..j]) = 0
        LcsFrontier {
            scores: vec![0; self.m() + 1],
        }
    }

    fn forward_step(&self, layer: usize, frontier_i: &Self::Frontier) -> Self::Frontier {
        // layer corresponds to i in [0..n-1], building row i+1 from row i
        let i = layer;
        let ch = self.s[i];
        let m = self.m();
        let mut next = vec![0u32; m + 1];

        for j in 1..=m {
            let up = frontier_i.scores[j];
            let left = next[j - 1];
            let diag = frontier_i.scores[j - 1] + if self.t[j - 1] == ch { 1 } else { 0 };
            next[j] = up.max(left).max(diag);
        }

        LcsFrontier { scores: next }
    }

    fn summarize_block(
        &self,
        a: usize,
        b: usize,
        frontier_a: &Self::Frontier,
    ) -> (Self::Frontier, Self::Summary) {
        let mut f = frontier_a.clone();
        for layer in a..b {
            f = self.forward_step(layer, &f);
        }
        (f.clone(), LcsSummary { end_frontier: f })
    }

    fn merge_summary(&self, _left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
        // For LCS, we only need the end frontier; composition is effectively:
        // Σ[a,c].end_frontier = Σ[b,c].end_frontier when used hierarchically.
        right.clone()
    }

    fn initial_boundary(&self) -> Self::Boundary {
        LcsBoundary { row: 0, col: 0 }
    }

    fn terminal_boundary(&self, _frontier_t: &Self::Frontier) -> Self::Boundary {
        // Global LCS: end at (n, m)
        LcsBoundary {
            row: self.n(),
            col: self.m(),
        }
    }

    fn choose_boundary(
        &self,
        a: usize,
        m: usize,
        c: usize,
        _sigma_left: &Self::Summary,
        _sigma_right: &Self::Summary,
        beta_a: &Self::Boundary,
        beta_c: &Self::Boundary,
    ) -> Self::Boundary {
        // Hirschberg-style mid-column choice for subproblem:
        // LCS of s[a..c] vs t[p..q], where:
        let p = beta_a.col;
        let q = beta_c.col;

        // Sanity: we only support well-formed boundaries produced by this algorithm.
        debug_assert!(a <= beta_a.row && beta_a.row <= m);
        debug_assert!(m <= beta_c.row && beta_c.row <= c);
        debug_assert!(p <= q && q <= self.m());

        // Forward: LCS(s[a..m], t[p..q])
        let fwd = lcs_last_row(&self.s[a..m], &self.t[p..q]);

        // Backward: LCS(reverse(s[m..c]), reverse(t[p..q]))
        let bwd = {
            let s_rev: Vec<u8> = self.s[m..c].iter().rev().copied().collect();
            let t_rev: Vec<u8> = self.t[p..q].iter().rev().copied().collect();
            lcs_last_row(&s_rev, &t_rev)
        };
        let len_t_sub = q - p;

        // Choose j maximizing forward[j] + backward[len_t_sub - j].
        let mut best_j = 0usize;
        let mut best_val = 0u32;
        for j in 0..=len_t_sub {
            let v = fwd[j] + bwd[len_t_sub - j];
            if v > best_val {
                best_val = v;
                best_j = j;
            }
        }

        LcsBoundary {
            row: m,
            col: p + best_j,
        }
    }

    fn reconstruct_block(
        &self,
        a: usize,
        b: usize,
        beta_a: &Self::Boundary,
        beta_b: &Self::Boundary,
    ) -> Vec<Self::State> {
        // Local Hirschberg-style reconstruction on subproblem:
        // s[a..b], t[p..q], from (a,p) to (b,q).
        let p = beta_a.col;
        let q = beta_b.col;
        debug_assert!(a <= b && b <= self.n());
        debug_assert!(p <= q && q <= self.m());

        let s_sub = &self.s[a..b];
        let t_sub = &self.t[p..q];
        let n = s_sub.len();
        let m = t_sub.len();

        // Standard DP with full table inside this block (allowed: block is small).
        let mut dp = vec![vec![0u32; m + 1]; n + 1];
        for i in 1..=n {
            for j in 1..=m {
                let up = dp[i - 1][j];
                let left = dp[i][j - 1];
                let diag = dp[i - 1][j - 1] + if s_sub[i - 1] == t_sub[j - 1] { 1 } else { 0 };
                dp[i][j] = up.max(left).max(diag);
            }
        }

        // Backtrack from (n,m) to (0,0).
        let mut i = n;
        let mut j = m;
        let mut rev_path: Vec<LcsState> = Vec::with_capacity(n + m + 1);
        rev_path.push((a + i, p + j));

        while i > 0 || j > 0 {
            if i > 0 && j > 0 && s_sub[i - 1] == t_sub[j - 1] && dp[i][j] == dp[i - 1][j - 1] + 1 {
                i -= 1;
                j -= 1;
            } else if i > 0 && dp[i][j] == dp[i - 1][j] {
                i -= 1;
            } else if j > 0 && dp[i][j] == dp[i][j - 1] {
                j -= 1;
            } else {
                // Should not happen for LCS; defensive fallback:
                if i > 0 {
                    i -= 1;
                } else {
                    j -= 1;
                }
            }
            rev_path.push((a + i, p + j));
        }

        rev_path.reverse();
        rev_path
    }

    fn extract_cost(&self, frontier_t: &Self::Frontier, _beta_t: &Self::Boundary) -> Self::Cost {
        // Global LCS length at (n,m).
        *frontier_t.scores.last().unwrap_or(&0)
    }
}

/// Compute the last DP row of LCS(x, y).
///
/// Returns a vector `row` of length |y|+1, where row[j] = LCS(x, y[0..j]).
fn lcs_last_row(x: &[u8], y: &[u8]) -> Vec<u32> {
    let m = y.len();
    let mut prev = vec![0u32; m + 1];
    let mut curr = vec![0u32; m + 1];

    for &cx in x {
        for j in 1..=m {
            let up = prev[j];
            let left = curr[j - 1];
            let diag = prev[j - 1] + if cx == y[j - 1] { 1 } else { 0 };
            curr[j] = up.max(left).max(diag);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HcpEngine;

    fn valid_lcs_path(path: &[(usize, usize)], n: usize, m: usize) -> bool {
        if path.is_empty() {
            // For degenerate cases with zero layers, an empty path is acceptable.
            return n == 0 || m == 0;
        }
        if *path.first().unwrap() != (0, 0) {
            return false;
        }
        if *path.last().unwrap() != (n, m) {
            return false;
        }
        for w in path.windows(2) {
            let (a, b) = (w[0], w[1]);
            let di = b.0 as isize - a.0 as isize;
            let dj = b.1 as isize - a.1 as isize;
            match (di, dj) {
                (1, 0) | (0, 1) | (1, 1) => {}
                _ => return false,
            }
        }
        true
    }

    #[test]
    fn last_row_basic_cases() {
        assert_eq!(lcs_last_row(b"", b""), vec![0]);
        assert_eq!(lcs_last_row(b"A", b""), vec![0]);
        assert_eq!(lcs_last_row(b"", b"A"), vec![0, 0]);
        assert_eq!(lcs_last_row(b"A", b"A"), vec![0, 1]);
        assert_eq!(lcs_last_row(b"A", b"B"), vec![0, 0]);
    }

    #[test]
    fn e2e_example_pairs() {
        let s = b"ACCGGTCGAGTGCGCGGAAGCCGGCCGAA";
        let t = b"GTCGTTCGGAATGCCGTTGCTCTGTAAA";
        let problem = LcsProblem::new(s, t);
        let engine = HcpEngine::new(problem);
        let (len_lcs, path) = engine.run();
        assert_eq!(len_lcs, 20);
        assert!(valid_lcs_path(&path, s.len(), t.len()));
    }

    #[test]
    fn edge_cases_empty_and_identical() {
        // empty
        let problem = LcsProblem::new(b"", b"ABC");
        let engine = HcpEngine::new(problem);
        let (len_lcs, path) = engine.run();
        assert_eq!(len_lcs, 0);
        assert!(valid_lcs_path(&path, 0, 3));

        // identical
        let s = b"HELLO";
        let problem = LcsProblem::new(s, s);
        let engine = HcpEngine::new(problem);
        let (len_lcs, path) = engine.run();
        assert_eq!(len_lcs, s.len() as u32);
        assert!(valid_lcs_path(&path, s.len(), s.len()));
    }
}
