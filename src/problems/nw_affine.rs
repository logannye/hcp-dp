//! Needleman–Wunsch global alignment with affine gaps (Gotoh).
//! States: M (match/mismatch), Ix (gap in t), Iy (gap in s).
//! Scoring parameters:
//! - match_score (added on match)
//! - mismatch_penalty (subtracted on mismatch)
//! - gap_open_penalty (added when opening a gap; typically negative)
//! - gap_extend_penalty (added when extending a gap; typically negative)

use crate::traits::{HcpProblem, SummaryApply};

#[derive(Clone)]
pub struct NwAffineProblem<'a> {
    pub s: &'a [u8],
    pub t: &'a [u8],
    pub match_score: i32,
    pub mismatch_penalty: i32,
    pub gap_open_penalty: i32,
    pub gap_extend_penalty: i32,
}

#[derive(Clone, Debug)]
pub struct NwAffFrontier {
    // Last row scores for columns 0..=m
    pub m_row: Vec<i32>,
    pub ix_row: Vec<i32>,
    pub iy_row: Vec<i32>,
}

#[derive(Clone, Debug)]
pub struct NwAffSummary {
    pub end_frontier: NwAffFrontier,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NwAffBoundary {
    pub row: usize,
    pub col: usize,
}

pub type NwAffState = (usize, usize);

impl<'a> NwAffineProblem<'a> {
    pub fn new(
        s: &'a [u8],
        t: &'a [u8],
        match_score: i32,
        mismatch_penalty: i32,
        gap_open_penalty: i32,
        gap_extend_penalty: i32,
    ) -> Self {
        Self {
            s,
            t,
            match_score,
            mismatch_penalty,
            gap_open_penalty,
            gap_extend_penalty,
        }
    }

    fn n(&self) -> usize {
        self.s.len()
    }
    fn m(&self) -> usize {
        self.t.len()
    }

    fn score_pair(&self, a: u8, b: u8) -> i32 {
        if a == b {
            self.match_score
        } else {
            -self.mismatch_penalty
        }
    }
}

impl<'a> HcpProblem for NwAffineProblem<'a> {
    type State = NwAffState;
    type Frontier = NwAffFrontier;
    type Summary = NwAffSummary;
    type Boundary = NwAffBoundary;
    type Cost = i32;

    fn num_layers(&self) -> usize {
        self.n()
    }

    fn init_frontier(&self) -> Self::Frontier {
        // i = 0. Only Iy (gap in s) matters horizontally, with affine costs.
        let m = self.m();
        let mut m_row = vec![i32::MIN / 4; m + 1];
        let mut ix_row = vec![i32::MIN / 4; m + 1];
        let mut iy_row = vec![i32::MIN / 4; m + 1];
        m_row[0] = 0;
        iy_row[0] = i32::MIN / 4;
        ix_row[0] = i32::MIN / 4;
        for j in 1..=m {
            // Opening or extending a gap in s across row 0.
            if j == 1 {
                iy_row[j] = self.gap_open_penalty + self.gap_extend_penalty;
            } else {
                iy_row[j] = iy_row[j - 1] + self.gap_extend_penalty;
            }
            m_row[j] = i32::MIN / 4;
            ix_row[j] = i32::MIN / 4;
        }
        NwAffFrontier {
            m_row,
            ix_row,
            iy_row,
        }
    }

    fn forward_step(&self, layer: usize, f: &Self::Frontier) -> Self::Frontier {
        // Build row i+1 from row i.
        let i = layer;
        let m = self.m();
        let a = self.s[i];

        let mut m_next = vec![i32::MIN / 4; m + 1];
        let mut ix_next = vec![i32::MIN / 4; m + 1];
        let mut iy_next = vec![i32::MIN / 4; m + 1];

        // j = 0 column: vertical gap in t (Ix)
        // Opening or extending gap downwards.
        ix_next[0] = if i == 0 {
            self.gap_open_penalty + self.gap_extend_penalty
        } else {
            f.ix_row[0] + self.gap_extend_penalty
        };

        for j in 1..=m {
            let b = self.t[j - 1];

            // M state: from M/IX/IY diagonal + match/mismatch
            let best_prev = f.m_row[j - 1].max(f.ix_row[j - 1]).max(f.iy_row[j - 1]);
            m_next[j] = best_prev + self.score_pair(a, b);

            // Ix: gap in t (vertical move): from M open or Ix extend
            ix_next[j] = (f.m_row[j] + self.gap_open_penalty + self.gap_extend_penalty)
                .max(f.ix_row[j] + self.gap_extend_penalty);

            // Iy: gap in s (horizontal move): from M open or Iy extend
            iy_next[j] = (m_next[j - 1] + self.gap_open_penalty + self.gap_extend_penalty)
                .max(iy_next[j - 1] + self.gap_extend_penalty);
        }

        NwAffFrontier {
            m_row: m_next,
            ix_row: ix_next,
            iy_row: iy_next,
        }
    }

    fn summarize_block(
        &self,
        a: usize,
        b: usize,
        frontier_a: &Self::Frontier,
    ) -> (Self::Frontier, Self::Summary) {
        let mut f = frontier_a.clone();
        for i in a..b {
            f = self.forward_step(i, &f);
        }
        let end = f.clone();
        (f, NwAffSummary { end_frontier: end })
    }

    fn merge_summary(&self, _left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
        right.clone()
    }

    fn initial_boundary(&self) -> Self::Boundary {
        NwAffBoundary { row: 0, col: 0 }
    }

    fn terminal_boundary(&self, _frontier_t: &Self::Frontier) -> Self::Boundary {
        NwAffBoundary {
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
        // Hirschberg-style split using affine last-row forward/backward sums.
        let p = beta_a.col;
        let q = beta_c.col;
        let fwd = affine_last_row(
            &self.s[a..m],
            &self.t[p..q],
            self.match_score,
            self.mismatch_penalty,
            self.gap_open_penalty,
            self.gap_extend_penalty,
        );

        // Backward: reverse both and compute last row; map j accordingly.
        let s_rev: Vec<u8> = self.s[m..c].iter().rev().copied().collect();
        let t_rev: Vec<u8> = self.t[p..q].iter().rev().copied().collect();
        let bwd = affine_last_row(
            &s_rev,
            &t_rev,
            self.match_score,
            self.mismatch_penalty,
            self.gap_open_penalty,
            self.gap_extend_penalty,
        );

        let len_t = q - p;
        let mut best_j = 0usize;
        let mut best_val = i32::MIN;
        for j in 0..=len_t {
            let v = fwd[j] + bwd[len_t - j];
            if v > best_val {
                best_val = v;
                best_j = j;
            }
        }

        NwAffBoundary {
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
        // Full-table Gotoh inside [a,b] × [p,q]
        let p = beta_a.col;
        let q = beta_b.col;
        let s_sub = &self.s[a..b];
        let t_sub = &self.t[p..q];
        let n = s_sub.len();
        let m = t_sub.len();

        let neg_inf = i32::MIN / 4;
        let mut m_dp = vec![vec![neg_inf; m + 1]; n + 1];
        let mut ix_dp = vec![vec![neg_inf; m + 1]; n + 1];
        let mut iy_dp = vec![vec![neg_inf; m + 1]; n + 1];

        m_dp[0][0] = 0;
        // Initialize first row (horizontal gaps in s)
        for j in 1..=m {
            if j == 1 {
                iy_dp[0][j] = self.gap_open_penalty + self.gap_extend_penalty;
            } else {
                iy_dp[0][j] = iy_dp[0][j - 1] + self.gap_extend_penalty;
            }
        }
        // Initialize first column (vertical gaps in t)
        for i in 1..=n {
            if i == 1 {
                ix_dp[i][0] = self.gap_open_penalty + self.gap_extend_penalty;
            } else {
                ix_dp[i][0] = ix_dp[i - 1][0] + self.gap_extend_penalty;
            }
        }

        for i in 1..=n {
            for j in 1..=m {
                let score = self.score_pair(s_sub[i - 1], t_sub[j - 1]);
                m_dp[i][j] = (m_dp[i - 1][j - 1]
                    .max(ix_dp[i - 1][j - 1])
                    .max(iy_dp[i - 1][j - 1]))
                    + score;
                ix_dp[i][j] = (m_dp[i - 1][j] + self.gap_open_penalty + self.gap_extend_penalty)
                    .max(ix_dp[i - 1][j] + self.gap_extend_penalty);
                iy_dp[i][j] = (m_dp[i][j - 1] + self.gap_open_penalty + self.gap_extend_penalty)
                    .max(iy_dp[i][j - 1] + self.gap_extend_penalty);
            }
        }

        // Backtrack from best of (M, Ix, Iy) at (n,m)
        let mut i = n;
        let mut j = m;
        let mut rev_path: Vec<NwAffState> = Vec::with_capacity(n + m + 1);
        rev_path.push((a + i, p + j));

        // Choose starting state
        let mut state = if m_dp[i][j] >= ix_dp[i][j] && m_dp[i][j] >= iy_dp[i][j] {
            0 // M
        } else if ix_dp[i][j] >= iy_dp[i][j] {
            1 // Ix
        } else {
            2 // Iy
        };

        while i > 0 || j > 0 {
            match state {
                0 => {
                    // M: came from diagonal of max(M, Ix, Iy)
                    let score = self.score_pair(s_sub[i - 1], t_sub[j - 1]);
                    let prev = m_dp[i - 1][j - 1]
                        .max(ix_dp[i - 1][j - 1])
                        .max(iy_dp[i - 1][j - 1]);
                    if m_dp[i][j] == prev + score {
                        i -= 1;
                        j -= 1;
                        // choose which state gave prev
                        if prev == m_dp[i][j] {
                            state = 0;
                        } else if prev == ix_dp[i][j] {
                            state = 1;
                        } else {
                            state = 2;
                        }
                    } else {
                        // fallback
                        if i > 0 && j > 0 {
                            i -= 1;
                            j -= 1;
                        } else if i > 0 {
                            i = i.saturating_sub(1);
                        } else {
                            j = j.saturating_sub(1);
                        }
                    }
                }
                1 => {
                    // Ix: from M open or Ix extend (vertical)
                    if i > 0 && ix_dp[i][j] == ix_dp[i - 1][j] + self.gap_extend_penalty {
                        i -= 1;
                        state = 1;
                    } else if i > 0
                        && ix_dp[i][j]
                            == m_dp[i - 1][j] + self.gap_open_penalty + self.gap_extend_penalty
                    {
                        i -= 1;
                        state = 0;
                    } else {
                        // fallback
                        if i > 0 {
                            i = i.saturating_sub(1);
                        } else if j > 0 {
                            j = j.saturating_sub(1);
                        }
                    }
                }
                _ => {
                    // Iy: from M open or Iy extend (horizontal)
                    if j > 0 && iy_dp[i][j] == iy_dp[i][j - 1] + self.gap_extend_penalty {
                        j -= 1;
                        state = 2;
                    } else if j > 0
                        && iy_dp[i][j]
                            == m_dp[i][j - 1] + self.gap_open_penalty + self.gap_extend_penalty
                    {
                        j -= 1;
                        state = 0;
                    } else {
                        // fallback
                        if j > 0 {
                            j = j.saturating_sub(1);
                        } else if i > 0 {
                            i = i.saturating_sub(1);
                        }
                    }
                }
            }
            rev_path.push((a + i, p + j));
        }

        rev_path.reverse();
        rev_path
    }

    fn extract_cost(&self, frontier_t: &Self::Frontier, _beta_t: &Self::Boundary) -> Self::Cost {
        let j = frontier_t.m_row.len() - 1;
        *[
            frontier_t.m_row[j],
            frontier_t.ix_row[j],
            frontier_t.iy_row[j],
        ]
        .iter()
        .max()
        .unwrap()
    }
}

impl SummaryApply<NwAffFrontier> for NwAffSummary {
    fn apply(&self, _frontier: &NwAffFrontier) -> NwAffFrontier {
        self.end_frontier.clone()
    }
}

/// Compute last row "best score" for affine-gap alignment of x vs y.
/// Returns a vector best[j] for j in 0..=|y| of the best among M/Ix/Iy at row end.
fn affine_last_row(
    x: &[u8],
    y: &[u8],
    match_score: i32,
    mismatch_penalty: i32,
    gap_open_penalty: i32,
    gap_extend_penalty: i32,
) -> Vec<i32> {
    let m = y.len();
    let neg_inf = i32::MIN / 4;
    let mut m_row = vec![neg_inf; m + 1];
    let mut ix_row = vec![neg_inf; m + 1];
    let mut iy_row = vec![neg_inf; m + 1];

    m_row[0] = 0;
    for j in 1..=m {
        if j == 1 {
            iy_row[j] = gap_open_penalty + gap_extend_penalty;
        } else {
            iy_row[j] = iy_row[j - 1] + gap_extend_penalty;
        }
    }

    for &cx in x {
        let mut m_next = vec![neg_inf; m + 1];
        let mut ix_next = vec![neg_inf; m + 1];
        let mut iy_next = vec![neg_inf; m + 1];

        // j = 0 column
        ix_next[0] = if ix_row[0] == neg_inf {
            gap_open_penalty + gap_extend_penalty
        } else {
            ix_row[0] + gap_extend_penalty
        };

        for j in 1..=m {
            let score = if cx == y[j - 1] {
                match_score
            } else {
                -mismatch_penalty
            };
            let best_prev = m_row[j - 1].max(ix_row[j - 1]).max(iy_row[j - 1]);
            m_next[j] = best_prev + score;

            ix_next[j] = (m_row[j] + gap_open_penalty + gap_extend_penalty)
                .max(ix_row[j] + gap_extend_penalty);
            iy_next[j] = (m_next[j - 1] + gap_open_penalty + gap_extend_penalty)
                .max(iy_next[j - 1] + gap_extend_penalty);
        }

        m_row = m_next;
        ix_row = ix_next;
        iy_row = iy_next;
    }

    (0..=m)
        .map(|j| m_row[j].max(ix_row[j]).max(iy_row[j]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HcpEngine;

    fn full_affine_score(s: &[u8], t: &[u8], ms: i32, mm: i32, go: i32, ge: i32) -> i32 {
        let n = s.len();
        let m = t.len();
        let neg_inf = i32::MIN / 4;
        let mut m_dp = vec![vec![neg_inf; m + 1]; n + 1];
        let mut ix_dp = vec![vec![neg_inf; m + 1]; n + 1];
        let mut iy_dp = vec![vec![neg_inf; m + 1]; n + 1];
        m_dp[0][0] = 0;
        for j in 1..=m {
            if j == 1 {
                iy_dp[0][j] = go + ge;
            } else {
                iy_dp[0][j] = iy_dp[0][j - 1] + ge;
            }
        }
        for i in 1..=n {
            if i == 1 {
                ix_dp[i][0] = go + ge;
            } else {
                ix_dp[i][0] = ix_dp[i - 1][0] + ge;
            }
        }
        for i in 1..=n {
            for j in 1..=m {
                let score = if s[i - 1] == t[j - 1] { ms } else { -mm };
                m_dp[i][j] = (m_dp[i - 1][j - 1]
                    .max(ix_dp[i - 1][j - 1])
                    .max(iy_dp[i - 1][j - 1]))
                    + score;
                ix_dp[i][j] = (m_dp[i - 1][j] + go + ge).max(ix_dp[i - 1][j] + ge);
                iy_dp[i][j] = (m_dp[i][j - 1] + go + ge).max(iy_dp[i][j - 1] + ge);
            }
        }
        *[m_dp[n][m], ix_dp[n][m], iy_dp[n][m]].iter().max().unwrap()
    }

    #[test]
    fn affine_matches_full_dp_small() {
        let s = b"GATTACA";
        let t = b"GCATGCU";
        let ms = 1;
        let mm = 1;
        let go = -2;
        let ge = -1;
        let problem = NwAffineProblem::new(s, t, ms, mm, go, ge);
        let engine = HcpEngine::new(problem);
        let (score, _path) = engine.run();
        let ref_score = full_affine_score(s, t, ms, mm, go, ge);
        assert_eq!(score, ref_score);
    }
}
