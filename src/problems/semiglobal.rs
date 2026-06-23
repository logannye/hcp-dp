//! Semi-global alignment with linear gap penalties.
//!
//! This implementation consumes the full query (`s`) against any target (`t`)
//! interval. Target prefix and suffix are free; query gaps remain penalized.

use crate::{
    scoring::SubstitutionScoring,
    traits::{HcpProblem, SummaryApply},
};

#[derive(Clone)]
pub struct SemiGlobalProblem<'a> {
    pub s: &'a [u8],
    pub t: &'a [u8],
    pub match_score: i32,
    pub mismatch_penalty: i32,
    pub gap_penalty: i32,
    pub scoring: SubstitutionScoring,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemiGlobalCell {
    pub row: usize,
    pub col: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemiGlobalFrontier {
    pub scores: Vec<i32>,
    starts: Vec<SemiGlobalCell>,
}

#[derive(Clone, Debug)]
pub struct SemiGlobalSummary<'a> {
    s: &'a [u8],
    t: &'a [u8],
    start: usize,
    end: usize,
    gap_penalty: i32,
    scoring: SubstitutionScoring,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemiGlobalBoundary {
    pub row: usize,
    pub col: usize,
    selected: Option<SemiGlobalSelection>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SemiGlobalSelection {
    score: i32,
    start: SemiGlobalCell,
    end: SemiGlobalCell,
}

impl<'a> SemiGlobalProblem<'a> {
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

    pub fn score_path(&self, path: &[SemiGlobalCell]) -> Option<i32> {
        let mut score = 0;
        for window in path.windows(2) {
            let (a, b) = (window[0], window[1]);
            let di = b.row.checked_sub(a.row)?;
            let dj = b.col.checked_sub(a.col)?;
            match (di, dj) {
                (1, 1) => score += self.score_pair(*self.s.get(a.row)?, *self.t.get(a.col)?),
                (1, 0) | (0, 1) => score += self.gap_penalty,
                _ => return None,
            }
        }
        Some(score)
    }

    pub fn full_table_score(&self) -> i32 {
        self.full_table_selection().score
    }

    pub fn full_table_target_interval(&self) -> (usize, usize) {
        let selection = self.full_table_selection();
        (selection.start.col, selection.end.col)
    }

    fn full_table_selection(&self) -> SemiGlobalSelection {
        let mut frontier = self.init_frontier();
        for layer in 0..self.n() {
            frontier = self.forward_step(layer, &frontier);
        }
        best_terminal_selection(self.n(), &frontier)
    }

    fn score_pair(&self, a: u8, b: u8) -> i32 {
        self.scoring.score(a, b)
    }

    fn selected_from_boundaries(
        &self,
        beta_a: &SemiGlobalBoundary,
        beta_b: &SemiGlobalBoundary,
    ) -> SemiGlobalSelection {
        beta_b
            .selected
            .or(beta_a.selected)
            .expect("semi-global reconstruction requires a selected interval")
    }

    fn boundary_for_split(
        &self,
        row: usize,
        beta_a: &SemiGlobalBoundary,
        beta_c: &SemiGlobalBoundary,
        selection: SemiGlobalSelection,
    ) -> SemiGlobalBoundary {
        let start = self.cell_for_boundary(beta_a, selection);
        let end = self.cell_for_boundary(beta_c, selection);
        let col = self.split_col_for_segment(row, start, end);
        SemiGlobalBoundary {
            row,
            col,
            selected: Some(selection),
        }
    }

    fn split_col_for_segment(
        &self,
        row: usize,
        start: SemiGlobalCell,
        end: SemiGlobalCell,
    ) -> usize {
        assert!(
            start.row <= row && row <= end.row,
            "split row outside semi-global segment"
        );
        let p = start.col;
        let q = end.col;
        assert!(
            p <= q && q <= self.m(),
            "invalid semi-global selected target interval"
        );

        let fwd = linear_last_row(
            &self.s[start.row..row],
            &self.t[p..q],
            &self.scoring,
            self.gap_penalty,
        );
        let s_rev: Vec<u8> = self.s[row..end.row].iter().rev().copied().collect();
        let t_rev: Vec<u8> = self.t[p..q].iter().rev().copied().collect();
        let bwd = linear_last_row(&s_rev, &t_rev, &self.scoring, self.gap_penalty);

        let width = q - p;
        let mut best_col = p;
        let mut best_score = i32::MIN;
        for local_col in 0..=width {
            let score = fwd[local_col] + bwd[width - local_col];
            if score > best_score {
                best_score = score;
                best_col = p + local_col;
            }
        }
        best_col
    }

    fn cell_for_boundary(
        &self,
        boundary: &SemiGlobalBoundary,
        selection: SemiGlobalSelection,
    ) -> SemiGlobalCell {
        if boundary.selected.is_none() && boundary.row == selection.start.row {
            selection.start
        } else {
            SemiGlobalCell {
                row: boundary.row,
                col: boundary.col,
            }
        }
    }

    fn reconstruct_selected_subsegment(
        &self,
        beta_a: &SemiGlobalBoundary,
        beta_b: &SemiGlobalBoundary,
        selection: SemiGlobalSelection,
    ) -> Vec<SemiGlobalCell> {
        let start = self.cell_for_boundary(beta_a, selection);
        let end = self.cell_for_boundary(beta_b, selection);
        assert!(
            start.row <= end.row && start.col <= end.col,
            "semi-global leaf boundaries must be ordered"
        );
        self.reconstruct_global_segment(start, end)
    }

    fn reconstruct_global_segment(
        &self,
        start: SemiGlobalCell,
        end: SemiGlobalCell,
    ) -> Vec<SemiGlobalCell> {
        assert!(
            start.row <= end.row,
            "semi-global start row must precede end row"
        );
        assert!(
            start.col <= end.col,
            "semi-global start col must precede end col"
        );
        assert!(
            end.row <= self.n() && end.col <= self.m(),
            "semi-global endpoint out of bounds"
        );

        let height = end.row - start.row;
        let width = end.col - start.col;
        let cols = width + 1;
        let mut dp = vec![0; (height + 1) * cols];

        for row in 1..=height {
            dp[cell_idx(row, 0, cols)] = dp[cell_idx(row - 1, 0, cols)] + self.gap_penalty;
        }
        for col in 1..=width {
            dp[cell_idx(0, col, cols)] = dp[cell_idx(0, col - 1, cols)] + self.gap_penalty;
        }
        for row in 1..=height {
            for col in 1..=width {
                let pair =
                    self.score_pair(self.s[start.row + row - 1], self.t[start.col + col - 1]);
                let diag = dp[cell_idx(row - 1, col - 1, cols)] + pair;
                let up = dp[cell_idx(row - 1, col, cols)] + self.gap_penalty;
                let left = dp[cell_idx(row, col - 1, cols)] + self.gap_penalty;
                dp[cell_idx(row, col, cols)] = diag.max(up).max(left);
            }
        }

        let mut row = height;
        let mut col = width;
        let mut rev_path = Vec::with_capacity(height + width + 1);
        rev_path.push(end);

        while row > 0 || col > 0 {
            if row > 0 && col > 0 {
                let pair =
                    self.score_pair(self.s[start.row + row - 1], self.t[start.col + col - 1]);
                if dp[cell_idx(row, col, cols)] == dp[cell_idx(row - 1, col - 1, cols)] + pair {
                    row -= 1;
                    col -= 1;
                    rev_path.push(SemiGlobalCell {
                        row: start.row + row,
                        col: start.col + col,
                    });
                    continue;
                }
            }

            if row > 0
                && dp[cell_idx(row, col, cols)]
                    == dp[cell_idx(row - 1, col, cols)] + self.gap_penalty
            {
                row -= 1;
            } else {
                assert!(col > 0, "semi-global traceback must have predecessor");
                col -= 1;
            }
            rev_path.push(SemiGlobalCell {
                row: start.row + row,
                col: start.col + col,
            });
        }

        rev_path.reverse();
        rev_path
    }
}

impl<'a> HcpProblem for SemiGlobalProblem<'a> {
    type State = SemiGlobalCell;
    type Frontier = SemiGlobalFrontier;
    type Summary = SemiGlobalSummary<'a>;
    type Boundary = SemiGlobalBoundary;
    type Cost = i32;

    fn num_layers(&self) -> usize {
        self.n()
    }

    fn init_frontier(&self) -> Self::Frontier {
        SemiGlobalFrontier {
            scores: vec![0; self.m() + 1],
            starts: (0..=self.m())
                .map(|col| SemiGlobalCell { row: 0, col })
                .collect(),
        }
    }

    fn forward_step(&self, layer: usize, frontier: &Self::Frontier) -> Self::Frontier {
        advance_semiglobal_row(
            self.s[layer],
            self.t,
            &self.scoring,
            self.gap_penalty,
            frontier,
        )
    }

    fn summarize_interval(&self, a: usize, b: usize) -> Self::Summary {
        assert!(a <= b && b <= self.n(), "invalid semi-global interval");
        SemiGlobalSummary {
            s: self.s,
            t: self.t,
            start: a,
            end: b,
            gap_penalty: self.gap_penalty,
            scoring: self.scoring.clone(),
        }
    }

    fn merge_summary(&self, left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
        assert_eq!(
            left.end, right.start,
            "semi-global summaries must be adjacent"
        );
        assert!(
            std::ptr::eq(left.s, right.s) && std::ptr::eq(left.t, right.t),
            "semi-global summaries must belong to the same problem"
        );
        assert_eq!(left.gap_penalty, right.gap_penalty);
        assert_eq!(left.scoring, right.scoring);
        SemiGlobalSummary {
            s: self.s,
            t: self.t,
            start: left.start,
            end: right.end,
            gap_penalty: self.gap_penalty,
            scoring: self.scoring.clone(),
        }
    }

    fn initial_boundary(&self) -> Self::Boundary {
        SemiGlobalBoundary {
            row: 0,
            col: 0,
            selected: None,
        }
    }

    fn terminal_boundary(&self, frontier_t: &Self::Frontier) -> Self::Boundary {
        let selected = best_terminal_selection(self.n(), frontier_t);
        SemiGlobalBoundary {
            row: self.n(),
            col: selected.end.col,
            selected: Some(selected),
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
        let selection = self.selected_from_boundaries(beta_a, beta_c);
        self.boundary_for_split(m, beta_a, beta_c, selection)
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
        let selection = self.selected_from_boundaries(beta_a, beta_b);
        self.reconstruct_selected_subsegment(beta_a, beta_b, selection)
    }

    fn extract_cost(&self, _frontier_t: &Self::Frontier, beta_t: &Self::Boundary) -> Self::Cost {
        beta_t
            .selected
            .expect("semi-global terminal boundary must carry selection")
            .score
    }
}

impl<'a> SummaryApply<SemiGlobalFrontier> for SemiGlobalSummary<'a> {
    fn apply(&self, frontier: &SemiGlobalFrontier) -> SemiGlobalFrontier {
        assert_eq!(
            frontier.scores.len(),
            self.t.len() + 1,
            "semi-global frontier width mismatch"
        );
        let mut current = frontier.clone();
        for layer in self.start..self.end {
            current = advance_semiglobal_row(
                self.s[layer],
                self.t,
                &self.scoring,
                self.gap_penalty,
                &current,
            );
        }
        current
    }
}

fn advance_semiglobal_row(
    s_ch: u8,
    t: &[u8],
    scoring: &SubstitutionScoring,
    gap_penalty: i32,
    frontier: &SemiGlobalFrontier,
) -> SemiGlobalFrontier {
    assert_eq!(
        frontier.scores.len(),
        t.len() + 1,
        "semi-global frontier width mismatch"
    );
    assert_eq!(
        frontier.starts.len(),
        t.len() + 1,
        "semi-global start width mismatch"
    );

    let mut scores = Vec::with_capacity(t.len() + 1);
    let mut starts = Vec::with_capacity(t.len() + 1);
    scores.push(frontier.scores[0] + gap_penalty);
    starts.push(frontier.starts[0]);

    for col in 1..=t.len() {
        let pair = scoring.score(s_ch, t[col - 1]);
        let diag = SemiGlobalCandidate {
            score: frontier.scores[col - 1] + pair,
            start: frontier.starts[col - 1],
        };
        let up = SemiGlobalCandidate {
            score: frontier.scores[col] + gap_penalty,
            start: frontier.starts[col],
        };
        let left = SemiGlobalCandidate {
            score: scores[col - 1] + gap_penalty,
            start: starts[col - 1],
        };
        let chosen = choose_semiglobal_cell(diag, up, left);
        scores.push(chosen.score);
        starts.push(chosen.start);
    }

    SemiGlobalFrontier { scores, starts }
}

#[derive(Clone, Copy)]
struct SemiGlobalCandidate {
    score: i32,
    start: SemiGlobalCell,
}

fn choose_semiglobal_cell(
    diag: SemiGlobalCandidate,
    up: SemiGlobalCandidate,
    left: SemiGlobalCandidate,
) -> SemiGlobalCandidate {
    [diag, up, left]
        .into_iter()
        .max_by_key(|candidate| candidate.score)
        .expect("semi-global cell has candidates")
}

fn linear_last_row(
    x: &[u8],
    y: &[u8],
    scoring: &SubstitutionScoring,
    gap_penalty: i32,
) -> Vec<i32> {
    let mut prev = Vec::with_capacity(y.len() + 1);
    let mut curr = vec![0; y.len() + 1];
    prev.push(0);
    for col in 1..=y.len() {
        prev.push(prev[col - 1] + gap_penalty);
    }

    for &cx in x {
        curr[0] = prev[0] + gap_penalty;
        for col in 1..=y.len() {
            let pair = scoring.score(cx, y[col - 1]);
            let diag = prev[col - 1] + pair;
            let up = prev[col] + gap_penalty;
            let left = curr[col - 1] + gap_penalty;
            curr[col] = diag.max(up).max(left);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev
}

fn best_terminal_selection(row: usize, frontier: &SemiGlobalFrontier) -> SemiGlobalSelection {
    let mut best_col = 0;
    let mut best_score = i32::MIN;
    for (col, score) in frontier.scores.iter().copied().enumerate() {
        if score > best_score {
            best_col = col;
            best_score = score;
        }
    }
    SemiGlobalSelection {
        score: best_score,
        start: frontier.starts[best_col],
        end: SemiGlobalCell { row, col: best_col },
    }
}

fn cell_idx(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HcpEngine;

    #[test]
    fn target_prefix_and_suffix_are_free() {
        let problem = SemiGlobalProblem::new(b"ACGT", b"TTACGTTT", 2, 1, -2);
        let (score, path) = HcpEngine::new(problem.clone()).run();
        assert_eq!(score, 8);
        assert_eq!(problem.full_table_target_interval(), (2, 6));
        assert_eq!(path.first(), Some(&SemiGlobalCell { row: 0, col: 2 }));
        assert_eq!(path.last(), Some(&SemiGlobalCell { row: 4, col: 6 }));
        assert_eq!(problem.score_path(&path), Some(score));
    }

    #[test]
    fn empty_query_has_zero_score() {
        let problem = SemiGlobalProblem::new(b"", b"ABC", 1, 1, -2);
        let (score, path) = HcpEngine::new(problem.clone()).run();
        assert_eq!(score, 0);
        assert_eq!(path.first(), Some(&SemiGlobalCell { row: 0, col: 0 }));
        assert_eq!(path.last(), Some(&SemiGlobalCell { row: 0, col: 0 }));
        assert_eq!(problem.score_path(&path), Some(score));
    }
}
