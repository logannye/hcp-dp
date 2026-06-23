//! Needleman-Wunsch global alignment with linear gap penalties.

use crate::{
    scoring::SubstitutionScoring,
    traits::{HcpProblem, SummaryApply},
};

#[derive(Clone)]
pub struct NwProblem<'a> {
    pub s: &'a [u8],
    pub t: &'a [u8],
    pub match_score: i32,
    pub mismatch_penalty: i32,
    pub gap_penalty: i32,
    pub scoring: SubstitutionScoring,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NwFrontier {
    pub scores: Vec<i32>,
}

#[derive(Clone, Debug)]
pub struct NwSummary<'a> {
    s: &'a [u8],
    t: &'a [u8],
    start: usize,
    end: usize,
    gap_penalty: i32,
    scoring: SubstitutionScoring,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NwBoundary {
    pub row: usize,
    pub col: usize,
}

pub type NwState = (usize, usize);

impl<'a> NwProblem<'a> {
    pub fn new(
        s: &'a [u8],
        t: &'a [u8],
        match_score: i32,
        mismatch_penalty: i32,
        gap_penalty: i32,
    ) -> Self {
        Self {
            s,
            t,
            match_score,
            mismatch_penalty,
            gap_penalty,
            scoring: SubstitutionScoring::match_mismatch(match_score, mismatch_penalty),
        }
    }

    pub fn with_scoring(
        s: &'a [u8],
        t: &'a [u8],
        scoring: SubstitutionScoring,
        gap_penalty: i32,
    ) -> Self {
        Self {
            s,
            t,
            match_score: 0,
            mismatch_penalty: 0,
            gap_penalty,
            scoring,
        }
    }

    pub fn n(&self) -> usize {
        self.s.len()
    }

    pub fn m(&self) -> usize {
        self.t.len()
    }

    pub fn score_path(&self, path: &[NwState]) -> Option<i32> {
        let mut score = 0;
        for window in path.windows(2) {
            let (a, b) = (window[0], window[1]);
            let di = b.0.checked_sub(a.0)?;
            let dj = b.1.checked_sub(a.1)?;
            match (di, dj) {
                (1, 1) => score += self.score_pair(*self.s.get(a.0)?, *self.t.get(a.1)?),
                (1, 0) | (0, 1) => score += self.gap_penalty,
                _ => return None,
            }
        }
        Some(score)
    }

    pub fn full_table_score(&self) -> i32 {
        nw_last_row(self.s, self.t, &self.scoring, self.gap_penalty)
            .last()
            .copied()
            .unwrap_or(0)
    }

    fn score_pair(&self, a: u8, b: u8) -> i32 {
        self.scoring.score(a, b)
    }
}

impl<'a> HcpProblem for NwProblem<'a> {
    type State = NwState;
    type Frontier = NwFrontier;
    type Summary = NwSummary<'a>;
    type Boundary = NwBoundary;
    type Cost = i32;

    fn num_layers(&self) -> usize {
        self.n()
    }

    fn init_frontier(&self) -> Self::Frontier {
        let mut scores = Vec::with_capacity(self.m() + 1);
        scores.push(0);
        for j in 1..=self.m() {
            scores.push(scores[j - 1] + self.gap_penalty);
        }
        NwFrontier { scores }
    }

    fn forward_step(&self, layer: usize, frontier: &Self::Frontier) -> Self::Frontier {
        let ch = self.s[layer];
        let mut next = Vec::with_capacity(self.m() + 1);
        next.push(frontier.scores[0] + self.gap_penalty);
        for j in 1..=self.m() {
            let diag = frontier.scores[j - 1] + self.score_pair(ch, self.t[j - 1]);
            let up = frontier.scores[j] + self.gap_penalty;
            let left = next[j - 1] + self.gap_penalty;
            next.push(diag.max(up).max(left));
        }
        NwFrontier { scores: next }
    }

    fn summarize_interval(&self, a: usize, b: usize) -> Self::Summary {
        assert!(a <= b && b <= self.n(), "invalid NW interval");
        NwSummary {
            s: self.s,
            t: self.t,
            start: a,
            end: b,
            gap_penalty: self.gap_penalty,
            scoring: self.scoring.clone(),
        }
    }

    fn merge_summary(&self, left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
        assert_eq!(left.end, right.start, "NW summaries must be adjacent");
        assert!(
            std::ptr::eq(left.s, right.s) && std::ptr::eq(left.t, right.t),
            "NW summaries must belong to the same problem"
        );
        assert_eq!(left.gap_penalty, right.gap_penalty);
        assert_eq!(left.scoring, right.scoring);
        NwSummary {
            s: self.s,
            t: self.t,
            start: left.start,
            end: right.end,
            gap_penalty: self.gap_penalty,
            scoring: self.scoring.clone(),
        }
    }

    fn initial_boundary(&self) -> Self::Boundary {
        NwBoundary { row: 0, col: 0 }
    }

    fn terminal_boundary(&self, _frontier_t: &Self::Frontier) -> Self::Boundary {
        NwBoundary {
            row: self.n(),
            col: self.m(),
        }
    }

    fn choose_split(
        &self,
        a: usize,
        m: usize,
        c: usize,
        beta_a: &Self::Boundary,
        beta_c: &Self::Boundary,
        _sigma_left: &Self::Summary,
        _sigma_right: &Self::Summary,
    ) -> Self::Boundary {
        assert_eq!(beta_a.row, a, "left boundary row must match interval start");
        assert_eq!(beta_c.row, c, "right boundary row must match interval end");
        assert!(a <= m && m <= c, "split must lie inside interval");

        let p = beta_a.col;
        let q = beta_c.col;
        assert!(p <= q && q <= self.m(), "invalid NW boundary columns");

        let fwd = nw_last_row(
            &self.s[a..m],
            &self.t[p..q],
            &self.scoring,
            self.gap_penalty,
        );
        let s_rev: Vec<u8> = self.s[m..c].iter().rev().copied().collect();
        let t_rev: Vec<u8> = self.t[p..q].iter().rev().copied().collect();
        let bwd = nw_last_row(&s_rev, &t_rev, &self.scoring, self.gap_penalty);

        let width = q - p;
        let mut best_col = p;
        let mut best_score = i32::MIN;
        for j in 0..=width {
            let score = fwd[j] + bwd[width - j];
            if score > best_score {
                best_score = score;
                best_col = p + j;
            }
        }

        NwBoundary {
            row: m,
            col: best_col,
        }
    }

    fn reconstruct_leaf(
        &self,
        a: usize,
        b: usize,
        beta_a: &Self::Boundary,
        beta_b: &Self::Boundary,
    ) -> Vec<Self::State> {
        assert_eq!(beta_a.row, a, "leaf start row must match beta_a");
        assert_eq!(beta_b.row, b, "leaf end row must match beta_b");
        let p = beta_a.col;
        let q = beta_b.col;
        assert!(p <= q && q <= self.m(), "invalid NW leaf columns");

        let s_sub = &self.s[a..b];
        let t_sub = &self.t[p..q];
        let n = s_sub.len();
        let m = t_sub.len();

        let mut dp = vec![vec![0; m + 1]; n + 1];
        for i in 1..=n {
            dp[i][0] = dp[i - 1][0] + self.gap_penalty;
        }
        for j in 1..=m {
            dp[0][j] = dp[0][j - 1] + self.gap_penalty;
        }
        for i in 1..=n {
            for j in 1..=m {
                let diag = dp[i - 1][j - 1] + self.score_pair(s_sub[i - 1], t_sub[j - 1]);
                let up = dp[i - 1][j] + self.gap_penalty;
                let left = dp[i][j - 1] + self.gap_penalty;
                dp[i][j] = diag.max(up).max(left);
            }
        }

        let mut i = n;
        let mut j = m;
        let mut rev_path = Vec::with_capacity(n + m + 1);
        rev_path.push((a + i, p + j));

        while i > 0 || j > 0 {
            if i > 0
                && j > 0
                && dp[i][j] == dp[i - 1][j - 1] + self.score_pair(s_sub[i - 1], t_sub[j - 1])
            {
                i -= 1;
                j -= 1;
            } else if i > 0 && dp[i][j] == dp[i - 1][j] + self.gap_penalty {
                i -= 1;
            } else {
                j -= 1;
            }
            rev_path.push((a + i, p + j));
        }

        rev_path.reverse();
        rev_path
    }

    fn extract_cost(&self, frontier_t: &Self::Frontier, _beta_t: &Self::Boundary) -> Self::Cost {
        frontier_t.scores.last().copied().unwrap_or(0)
    }
}

impl<'a> SummaryApply<NwFrontier> for NwSummary<'a> {
    fn apply(&self, frontier: &NwFrontier) -> NwFrontier {
        assert_eq!(
            frontier.scores.len(),
            self.t.len() + 1,
            "NW frontier width mismatch"
        );
        let mut current = frontier.clone();
        for layer in self.start..self.end {
            let ch = self.s[layer];
            let mut next = Vec::with_capacity(self.t.len() + 1);
            next.push(current.scores[0] + self.gap_penalty);
            for j in 1..=self.t.len() {
                let pair_score = self.scoring.score(ch, self.t[j - 1]);
                let diag = current.scores[j - 1] + pair_score;
                let up = current.scores[j] + self.gap_penalty;
                let left = next[j - 1] + self.gap_penalty;
                next.push(diag.max(up).max(left));
            }
            current = NwFrontier { scores: next };
        }
        current
    }
}

fn nw_last_row(x: &[u8], y: &[u8], scoring: &SubstitutionScoring, gap_penalty: i32) -> Vec<i32> {
    let mut prev = Vec::with_capacity(y.len() + 1);
    let mut curr = vec![0; y.len() + 1];
    prev.push(0);
    for j in 1..=y.len() {
        prev.push(prev[j - 1] + gap_penalty);
    }

    for &cx in x {
        curr[0] = prev[0] + gap_penalty;
        for j in 1..=y.len() {
            let score = scoring.score(cx, y[j - 1]);
            let diag = prev[j - 1] + score;
            let up = prev[j] + gap_penalty;
            let left = curr[j - 1] + gap_penalty;
            curr[j] = diag.max(up).max(left);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{scoring::SubstitutionScoring, HcpEngine};

    #[test]
    fn audit_regression_path_realizes_score() {
        let problem = NwProblem::new(b"CBA", b"ACC", 2, 1, -2);
        let (score, path) = HcpEngine::new(problem.clone()).run();
        assert_eq!(score, -3);
        assert_eq!(problem.score_path(&path), Some(score));
    }

    #[test]
    fn empty_source_reconstructs_gap_path() {
        let problem = NwProblem::new(b"", b"ABC", 1, 1, -2);
        let (score, path) = HcpEngine::new(problem.clone()).run();
        assert_eq!(score, -6);
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(0, 3)));
        assert_eq!(problem.score_path(&path), Some(score));
    }

    #[test]
    fn matrix_scoring_realizes_protein_score() {
        let problem = NwProblem::with_scoring(b"W", b"W", SubstitutionScoring::blosum62(), -4);
        let (score, path) = HcpEngine::new(problem.clone()).run();
        assert_eq!(score, 11);
        assert_eq!(problem.full_table_score(), 11);
        assert_eq!(problem.score_path(&path), Some(score));
    }
}
