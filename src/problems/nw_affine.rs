//! Needleman-Wunsch global alignment with Gotoh affine gap penalties.
//!
//! Scoring convention: the first position in a gap costs
//! `gap_open + gap_extend`; each continued gap position costs `gap_extend`.

use crate::{
    scoring::SubstitutionScoring,
    traits::{HcpProblem, SummaryApply},
};

const NEG_INF: i32 = i32::MIN / 4;
const STATES: [AffineState; 3] = [AffineState::Match, AffineState::GapInT, AffineState::GapInS];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AffineState {
    Match,
    /// A vertical move: `s` is consumed and aligned to a gap in `t`.
    GapInT,
    /// A horizontal move: `t` is consumed and aligned to a gap in `s`.
    GapInS,
}

impl AffineState {
    const fn idx(self) -> usize {
        match self {
            Self::Match => 0,
            Self::GapInT => 1,
            Self::GapInS => 2,
        }
    }

    const fn bit(self) -> u8 {
        1 << self.idx()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineStateSet(u8);

impl AffineStateSet {
    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn single(state: AffineState) -> Self {
        Self(state.bit())
    }

    pub const fn all() -> Self {
        Self(0b111)
    }

    pub const fn contains(self, state: AffineState) -> bool {
        self.0 & state.bit() != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    fn insert(&mut self, state: AffineState) {
        self.0 |= state.bit();
    }

    fn iter(self) -> impl Iterator<Item = AffineState> {
        STATES
            .into_iter()
            .filter(move |state| self.contains(*state))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NwAffineState {
    pub row: usize,
    pub col: usize,
    pub state: AffineState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NwAffineBoundary {
    pub row: usize,
    pub col: usize,
    pub states: AffineStateSet,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NwAffineTrace {
    pub score: i32,
    pub path: Vec<NwAffineState>,
}

#[derive(Clone)]
pub struct NwAffineProblem<'a> {
    pub s: &'a [u8],
    pub t: &'a [u8],
    pub match_score: i32,
    pub mismatch_penalty: i32,
    pub gap_open: i32,
    pub gap_extend: i32,
    pub scoring: SubstitutionScoring,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NwAffineFrontier {
    pub match_scores: Vec<i32>,
    pub gap_in_t: Vec<i32>,
    pub gap_in_s: Vec<i32>,
}

#[derive(Clone, Debug)]
pub struct NwAffineSummary<'a> {
    s: &'a [u8],
    t: &'a [u8],
    start: usize,
    end: usize,
    gap_open: i32,
    gap_extend: i32,
    scoring: SubstitutionScoring,
}

impl<'a> NwAffineProblem<'a> {
    pub fn new(
        s: &'a [u8],
        t: &'a [u8],
        match_score: i32,
        mismatch_penalty: i32,
        gap_open: i32,
        gap_extend: i32,
    ) -> Self {
        Self {
            s,
            t,
            match_score,
            mismatch_penalty,
            gap_open,
            gap_extend,
            scoring: SubstitutionScoring::match_mismatch(match_score, mismatch_penalty),
        }
    }

    pub fn with_scoring(
        s: &'a [u8],
        t: &'a [u8],
        scoring: SubstitutionScoring,
        gap_open: i32,
        gap_extend: i32,
    ) -> Self {
        Self {
            s,
            t,
            match_score: 0,
            mismatch_penalty: 0,
            gap_open,
            gap_extend,
            scoring,
        }
    }

    pub fn n(&self) -> usize {
        self.s.len()
    }

    pub fn m(&self) -> usize {
        self.t.len()
    }

    pub fn score_path(&self, path: &[NwAffineState]) -> Option<i32> {
        let mut score = 0;
        for window in path.windows(2) {
            let (a, b) = (&window[0], &window[1]);
            let di = b.row.checked_sub(a.row)?;
            let dj = b.col.checked_sub(a.col)?;
            match (di, dj, b.state) {
                (1, 1, AffineState::Match) => {
                    score += self.score_pair(*self.s.get(a.row)?, *self.t.get(a.col)?);
                }
                (1, 0, AffineState::GapInT) => {
                    let gap_score = self.gap_transition(a.state, AffineState::GapInT);
                    if gap_score <= NEG_INF / 2 {
                        return None;
                    }
                    score += gap_score;
                }
                (0, 1, AffineState::GapInS) => {
                    let gap_score = self.gap_transition(a.state, AffineState::GapInS);
                    if gap_score <= NEG_INF / 2 {
                        return None;
                    }
                    score += gap_score;
                }
                _ => return None,
            }
        }
        Some(score)
    }

    pub fn full_table_score(&self) -> i32 {
        let mut frontier = self.init_frontier();
        for layer in 0..self.n() {
            frontier = self.forward_step(layer, &frontier);
        }
        let terminal = self.terminal_boundary(&frontier);
        self.extract_cost(&frontier, &terminal)
    }

    fn score_pair(&self, a: u8, b: u8) -> i32 {
        self.scoring.score(a, b)
    }

    fn gap_transition(&self, from: AffineState, to: AffineState) -> i32 {
        gap_transition(self.gap_open, self.gap_extend, from, to)
    }

    fn constrained_forward_row(
        &self,
        a: usize,
        m: usize,
        p: usize,
        q: usize,
        beta_a: &NwAffineBoundary,
    ) -> NwAffineFrontier {
        let width = q - p;
        let mut frontier =
            constrained_start_frontier(width, beta_a.states, self.gap_open, self.gap_extend);
        for layer in a..m {
            frontier = advance_affine_row(
                self.s[layer],
                &self.t[p..q],
                &self.scoring,
                self.gap_open,
                self.gap_extend,
                &frontier,
            );
        }
        frontier
    }

    fn constrained_backward_row(
        &self,
        m: usize,
        c: usize,
        p: usize,
        q: usize,
        beta_c: &NwAffineBoundary,
    ) -> NwAffineFrontier {
        let width = q - p;
        let mut next = NwAffineFrontier::filled(width + 1, NEG_INF);
        for state in beta_c.states.iter() {
            next.set(width, state, 0);
        }
        fill_backward_horizontal_suffix(&mut next, self.gap_open, self.gap_extend);

        for layer in (m..c).rev() {
            let mut curr = NwAffineFrontier::filled(width + 1, NEG_INF);

            for state in STATES {
                let vertical = add_score(
                    next.score(width, AffineState::GapInT),
                    self.gap_transition(state, AffineState::GapInT),
                );
                curr.set(width, state, vertical);
            }

            for local_col in (0..width).rev() {
                let pair = self.score_pair(self.s[layer], self.t[p + local_col]);
                for state in STATES {
                    let diag = add_score(next.score(local_col + 1, AffineState::Match), pair);
                    let vertical = add_score(
                        next.score(local_col, AffineState::GapInT),
                        self.gap_transition(state, AffineState::GapInT),
                    );
                    let horizontal = add_score(
                        curr.score(local_col + 1, AffineState::GapInS),
                        self.gap_transition(state, AffineState::GapInS),
                    );
                    curr.set(local_col, state, diag.max(vertical).max(horizontal));
                }
            }

            next = curr;
        }

        next
    }
}

impl<'a> HcpProblem for NwAffineProblem<'a> {
    type State = NwAffineState;
    type Frontier = NwAffineFrontier;
    type Summary = NwAffineSummary<'a>;
    type Boundary = NwAffineBoundary;
    type Cost = i32;

    fn num_layers(&self) -> usize {
        self.n()
    }

    fn init_frontier(&self) -> Self::Frontier {
        constrained_start_frontier(
            self.m(),
            AffineStateSet::single(AffineState::Match),
            self.gap_open,
            self.gap_extend,
        )
    }

    fn forward_step(&self, layer: usize, frontier: &Self::Frontier) -> Self::Frontier {
        advance_affine_row(
            self.s[layer],
            self.t,
            &self.scoring,
            self.gap_open,
            self.gap_extend,
            frontier,
        )
    }

    fn summarize_interval(&self, a: usize, b: usize) -> Self::Summary {
        assert!(a <= b && b <= self.n(), "invalid affine NW interval");
        NwAffineSummary {
            s: self.s,
            t: self.t,
            start: a,
            end: b,
            gap_open: self.gap_open,
            gap_extend: self.gap_extend,
            scoring: self.scoring.clone(),
        }
    }

    fn merge_summary(&self, left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
        assert_eq!(
            left.end, right.start,
            "affine NW summaries must be adjacent"
        );
        assert!(
            std::ptr::eq(left.s, right.s) && std::ptr::eq(left.t, right.t),
            "affine NW summaries must belong to the same problem"
        );
        assert_eq!(
            (left.gap_open, left.gap_extend),
            (right.gap_open, right.gap_extend)
        );
        assert_eq!(left.scoring, right.scoring);
        NwAffineSummary {
            s: self.s,
            t: self.t,
            start: left.start,
            end: right.end,
            gap_open: self.gap_open,
            gap_extend: self.gap_extend,
            scoring: self.scoring.clone(),
        }
    }

    fn initial_boundary(&self) -> Self::Boundary {
        NwAffineBoundary {
            row: 0,
            col: 0,
            states: AffineStateSet::single(AffineState::Match),
        }
    }

    fn terminal_boundary(&self, frontier_t: &Self::Frontier) -> Self::Boundary {
        NwAffineBoundary {
            row: self.n(),
            col: self.m(),
            states: best_states_at(frontier_t, self.m()),
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
        assert!(
            !beta_a.states.is_empty() && !beta_c.states.is_empty(),
            "affine boundaries must allow at least one state"
        );

        let p = beta_a.col;
        let q = beta_c.col;
        assert!(
            p <= q && q <= self.m(),
            "invalid affine NW boundary columns"
        );

        let fwd = self.constrained_forward_row(a, m, p, q, beta_a);
        let bwd = self.constrained_backward_row(m, c, p, q, beta_c);
        let width = q - p;

        let mut best_col = 0;
        let mut best_state = None;
        let mut best_score = NEG_INF;
        for local_col in 0..=width {
            for state in STATES {
                let score = add_score(fwd.score(local_col, state), bwd.score(local_col, state));
                if score > best_score {
                    best_col = local_col;
                    best_state = Some(state);
                    best_score = score;
                }
            }
        }

        let state = best_state.expect("affine split must find a feasible midpoint state");
        NwAffineBoundary {
            row: m,
            col: p + best_col,
            states: AffineStateSet::single(state),
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
        assert!(
            !beta_a.states.is_empty() && !beta_b.states.is_empty(),
            "affine leaf boundaries must allow at least one state"
        );
        let p = beta_a.col;
        let q = beta_b.col;
        assert!(p <= q && q <= self.m(), "invalid affine NW leaf columns");

        let height = b - a;
        let width = q - p;
        let cols = width + 1;
        let mut dp = vec![[NEG_INF; 3]; (height + 1) * cols];

        for state in beta_a.states.iter() {
            dp[cell_idx(0, 0, cols)][state.idx()] = 0;
        }

        for local_col in 1..=width {
            let prev = dp[cell_idx(0, local_col - 1, cols)];
            dp[cell_idx(0, local_col, cols)][AffineState::GapInS.idx()] =
                best_to_gap(prev, AffineState::GapInS, self.gap_open, self.gap_extend);
        }

        for local_row in 1..=height {
            let up = dp[cell_idx(local_row - 1, 0, cols)];
            dp[cell_idx(local_row, 0, cols)][AffineState::GapInT.idx()] =
                best_to_gap(up, AffineState::GapInT, self.gap_open, self.gap_extend);

            for local_col in 1..=width {
                let pair = self.score_pair(self.s[a + local_row - 1], self.t[p + local_col - 1]);
                let diag = dp[cell_idx(local_row - 1, local_col - 1, cols)];
                let up = dp[cell_idx(local_row - 1, local_col, cols)];
                let left = dp[cell_idx(local_row, local_col - 1, cols)];
                let current = cell_idx(local_row, local_col, cols);

                dp[current][AffineState::Match.idx()] = best_to_match(diag, pair);
                dp[current][AffineState::GapInT.idx()] =
                    best_to_gap(up, AffineState::GapInT, self.gap_open, self.gap_extend);
                dp[current][AffineState::GapInS.idx()] =
                    best_to_gap(left, AffineState::GapInS, self.gap_open, self.gap_extend);
            }
        }

        let (mut state, mut current_score) =
            best_allowed_cell(dp[cell_idx(height, width, cols)], beta_b.states)
                .expect("affine leaf must have a feasible endpoint");
        let mut local_row = height;
        let mut local_col = width;
        let mut rev_path = Vec::with_capacity(height + width + 1);
        rev_path.push(NwAffineState {
            row: b,
            col: q,
            state,
        });

        while local_row > 0 || local_col > 0 {
            match state {
                AffineState::Match => {
                    assert!(
                        local_row > 0 && local_col > 0,
                        "match state requires a diagonal predecessor"
                    );
                    let pair =
                        self.score_pair(self.s[a + local_row - 1], self.t[p + local_col - 1]);
                    let prev = dp[cell_idx(local_row - 1, local_col - 1, cols)];
                    state = find_predecessor(prev, pair, current_score, |_, delta| delta);
                    local_row -= 1;
                    local_col -= 1;
                }
                AffineState::GapInT => {
                    assert!(
                        local_row > 0,
                        "gap-in-t state requires a vertical predecessor"
                    );
                    let prev = dp[cell_idx(local_row - 1, local_col, cols)];
                    state = find_predecessor(prev, self.gap_extend, current_score, |from, _| {
                        self.gap_transition(from, AffineState::GapInT)
                    });
                    local_row -= 1;
                }
                AffineState::GapInS => {
                    assert!(
                        local_col > 0,
                        "gap-in-s state requires a horizontal predecessor"
                    );
                    let prev = dp[cell_idx(local_row, local_col - 1, cols)];
                    state = find_predecessor(prev, self.gap_extend, current_score, |from, _| {
                        self.gap_transition(from, AffineState::GapInS)
                    });
                    local_col -= 1;
                }
            }

            current_score = dp[cell_idx(local_row, local_col, cols)][state.idx()];
            rev_path.push(NwAffineState {
                row: a + local_row,
                col: p + local_col,
                state,
            });
        }

        assert!(
            beta_a.states.contains(state),
            "affine traceback ended in a disallowed start state"
        );
        rev_path.reverse();
        rev_path
    }

    fn extract_cost(&self, frontier_t: &Self::Frontier, beta_t: &Self::Boundary) -> Self::Cost {
        beta_t
            .states
            .iter()
            .map(|state| frontier_t.score(self.m(), state))
            .max()
            .unwrap_or(NEG_INF)
    }
}

impl<'a> SummaryApply<NwAffineFrontier> for NwAffineSummary<'a> {
    fn apply(&self, frontier: &NwAffineFrontier) -> NwAffineFrontier {
        assert_eq!(
            frontier.len(),
            self.t.len() + 1,
            "affine NW frontier width mismatch"
        );
        let mut current = frontier.clone();
        for layer in self.start..self.end {
            current = advance_affine_row(
                self.s[layer],
                self.t,
                &self.scoring,
                self.gap_open,
                self.gap_extend,
                &current,
            );
        }
        current
    }
}

impl NwAffineFrontier {
    fn filled(len: usize, value: i32) -> Self {
        Self {
            match_scores: vec![value; len],
            gap_in_t: vec![value; len],
            gap_in_s: vec![value; len],
        }
    }

    fn len(&self) -> usize {
        self.match_scores.len()
    }

    fn score(&self, col: usize, state: AffineState) -> i32 {
        match state {
            AffineState::Match => self.match_scores[col],
            AffineState::GapInT => self.gap_in_t[col],
            AffineState::GapInS => self.gap_in_s[col],
        }
    }

    fn set(&mut self, col: usize, state: AffineState, value: i32) {
        match state {
            AffineState::Match => self.match_scores[col] = value,
            AffineState::GapInT => self.gap_in_t[col] = value,
            AffineState::GapInS => self.gap_in_s[col] = value,
        }
    }
}

#[derive(Clone, Copy)]
struct BandedAffineCell {
    scores: [i32; 3],
    predecessors: [Option<AffineState>; 3],
}

impl BandedAffineCell {
    fn empty() -> Self {
        Self {
            scores: [NEG_INF; 3],
            predecessors: [None; 3],
        }
    }
}

struct BandedAffineRow {
    start_col: usize,
    cells: Vec<BandedAffineCell>,
}

impl BandedAffineRow {
    fn new(row: usize, target_len: usize, diagonal_band: usize) -> Self {
        let start_col = row.saturating_sub(diagonal_band);
        let end_col = row.saturating_add(diagonal_band).min(target_len);
        Self {
            start_col,
            cells: vec![BandedAffineCell::empty(); end_col - start_col + 1],
        }
    }

    fn contains(&self, col: usize) -> bool {
        col >= self.start_col && col < self.start_col + self.cells.len()
    }

    fn get(&self, col: usize) -> Option<&BandedAffineCell> {
        self.contains(col)
            .then(|| &self.cells[col - self.start_col])
    }

    fn get_mut(&mut self, col: usize) -> Option<&mut BandedAffineCell> {
        self.contains(col)
            .then(move || &mut self.cells[col - self.start_col])
    }
}

/// Exact global affine traceback constrained to a diagonal band.
///
/// Returns `None` when the terminal cell is outside the band or when no
/// feasible path inside the band reaches the terminal cell. The returned path
/// uses the same state semantics as [`NwAffineProblem`] and can be checked with
/// [`NwAffineProblem::score_path`].
pub fn trace_banded_affine(
    problem: &NwAffineProblem<'_>,
    diagonal_band: usize,
) -> Option<NwAffineTrace> {
    let n = problem.n();
    let m = problem.m();
    if n.abs_diff(m) > diagonal_band {
        return None;
    }

    let mut rows: Vec<BandedAffineRow> = (0..=n)
        .map(|row| BandedAffineRow::new(row, m, diagonal_band))
        .collect();
    rows[0].get_mut(0)?.scores[AffineState::Match.idx()] = 0;

    for row in 0..=n {
        let start = rows[row].start_col;
        let end = rows[row].start_col + rows[row].cells.len() - 1;
        for col in start..=end {
            if row == 0 && col == 0 {
                continue;
            }

            let mut scores = [NEG_INF; 3];
            let mut predecessors = [None; 3];

            if row > 0 && col > 0 {
                if let Some(diag) = rows[row - 1].get(col - 1) {
                    let pair = problem.score_pair(problem.s[row - 1], problem.t[col - 1]);
                    let (score, predecessor) = best_banded_transition(diag.scores, |state| {
                        add_score(diag.scores[state.idx()], pair)
                    });
                    scores[AffineState::Match.idx()] = score;
                    predecessors[AffineState::Match.idx()] = predecessor;
                }
            }

            if row > 0 {
                if let Some(up) = rows[row - 1].get(col) {
                    let (score, predecessor) = best_banded_transition(up.scores, |state| {
                        add_score(
                            up.scores[state.idx()],
                            problem.gap_transition(state, AffineState::GapInT),
                        )
                    });
                    scores[AffineState::GapInT.idx()] = score;
                    predecessors[AffineState::GapInT.idx()] = predecessor;
                }
            }

            if col > 0 && rows[row].contains(col - 1) {
                let left_scores = rows[row]
                    .get(col - 1)
                    .expect("checked left cell must exist")
                    .scores;
                let (score, predecessor) = best_banded_transition(left_scores, |state| {
                    add_score(
                        left_scores[state.idx()],
                        problem.gap_transition(state, AffineState::GapInS),
                    )
                });
                scores[AffineState::GapInS.idx()] = score;
                predecessors[AffineState::GapInS.idx()] = predecessor;
            }

            let cell = rows[row]
                .get_mut(col)
                .expect("current band cell must exist while filling");
            cell.scores = scores;
            cell.predecessors = predecessors;
        }
    }

    let end_cell = rows[n].get(m)?;
    let (mut state, score) = best_allowed_cell(end_cell.scores, AffineStateSet::all())?;
    let mut row = n;
    let mut col = m;
    let mut rev_path = Vec::with_capacity(n + m + 1);
    rev_path.push(NwAffineState { row, col, state });

    while row > 0 || col > 0 {
        let predecessor = rows[row].get(col)?.predecessors[state.idx()]?;
        match state {
            AffineState::Match => {
                row = row.checked_sub(1)?;
                col = col.checked_sub(1)?;
            }
            AffineState::GapInT => {
                row = row.checked_sub(1)?;
            }
            AffineState::GapInS => {
                col = col.checked_sub(1)?;
            }
        }
        state = predecessor;
        rev_path.push(NwAffineState { row, col, state });
    }

    if state != AffineState::Match {
        return None;
    }
    rev_path.reverse();
    Some(NwAffineTrace {
        score,
        path: rev_path,
    })
}

fn best_banded_transition<F>(scores: [i32; 3], transition_score: F) -> (i32, Option<AffineState>)
where
    F: Fn(AffineState) -> i32,
{
    let mut best_score = NEG_INF;
    let mut best_state = None;
    for state in STATES {
        if scores[state.idx()] <= NEG_INF / 2 {
            continue;
        }
        let score = transition_score(state);
        if score > best_score {
            best_score = score;
            best_state = Some(state);
        }
    }
    (best_score, best_state)
}

fn constrained_start_frontier(
    width: usize,
    states: AffineStateSet,
    gap_open: i32,
    gap_extend: i32,
) -> NwAffineFrontier {
    assert!(
        !states.is_empty(),
        "affine start frontier needs at least one state"
    );
    let mut frontier = NwAffineFrontier::filled(width + 1, NEG_INF);
    for state in states.iter() {
        frontier.set(0, state, 0);
    }
    for col in 1..=width {
        let prev = [
            frontier.match_scores[col - 1],
            frontier.gap_in_t[col - 1],
            frontier.gap_in_s[col - 1],
        ];
        frontier.gap_in_s[col] = best_to_gap(prev, AffineState::GapInS, gap_open, gap_extend);
    }
    frontier
}

fn advance_affine_row(
    s_ch: u8,
    t: &[u8],
    scoring: &SubstitutionScoring,
    gap_open: i32,
    gap_extend: i32,
    frontier: &NwAffineFrontier,
) -> NwAffineFrontier {
    assert_eq!(
        frontier.len(),
        t.len() + 1,
        "affine frontier width mismatch"
    );
    let mut next = NwAffineFrontier::filled(t.len() + 1, NEG_INF);

    let up_col_zero = [
        frontier.match_scores[0],
        frontier.gap_in_t[0],
        frontier.gap_in_s[0],
    ];
    next.gap_in_t[0] = best_to_gap(up_col_zero, AffineState::GapInT, gap_open, gap_extend);

    for col in 1..=t.len() {
        let pair = scoring.score(s_ch, t[col - 1]);
        let diag = [
            frontier.match_scores[col - 1],
            frontier.gap_in_t[col - 1],
            frontier.gap_in_s[col - 1],
        ];
        let up = [
            frontier.match_scores[col],
            frontier.gap_in_t[col],
            frontier.gap_in_s[col],
        ];
        let left = [
            next.match_scores[col - 1],
            next.gap_in_t[col - 1],
            next.gap_in_s[col - 1],
        ];
        next.match_scores[col] = best_to_match(diag, pair);
        next.gap_in_t[col] = best_to_gap(up, AffineState::GapInT, gap_open, gap_extend);
        next.gap_in_s[col] = best_to_gap(left, AffineState::GapInS, gap_open, gap_extend);
    }

    next
}

fn fill_backward_horizontal_suffix(
    frontier: &mut NwAffineFrontier,
    gap_open: i32,
    gap_extend: i32,
) {
    let width = frontier.len() - 1;
    for col in (0..width).rev() {
        for state in STATES {
            let score = add_score(
                frontier.score(col + 1, AffineState::GapInS),
                gap_transition(gap_open, gap_extend, state, AffineState::GapInS),
            );
            frontier.set(col, state, score);
        }
    }
}

fn best_states_at(frontier: &NwAffineFrontier, col: usize) -> AffineStateSet {
    let best = STATES
        .into_iter()
        .map(|state| frontier.score(col, state))
        .max()
        .unwrap_or(NEG_INF);
    let mut states = AffineStateSet::empty();
    for state in STATES {
        if frontier.score(col, state) == best {
            states.insert(state);
        }
    }
    states
}

fn best_allowed_cell(cell: [i32; 3], allowed: AffineStateSet) -> Option<(AffineState, i32)> {
    let mut best = None;
    for state in STATES {
        if !allowed.contains(state) {
            continue;
        }
        let score = cell[state.idx()];
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((state, score));
        }
    }
    best.filter(|(_, score)| *score > NEG_INF / 2)
}

fn best_to_match(cell: [i32; 3], pair: i32) -> i32 {
    STATES
        .into_iter()
        .map(|state| add_score(cell[state.idx()], pair))
        .max()
        .unwrap_or(NEG_INF)
}

fn best_to_gap(cell: [i32; 3], target: AffineState, gap_open: i32, gap_extend: i32) -> i32 {
    STATES
        .into_iter()
        .map(|state| {
            add_score(
                cell[state.idx()],
                gap_transition(gap_open, gap_extend, state, target),
            )
        })
        .max()
        .unwrap_or(NEG_INF)
}

fn find_predecessor<F>(
    cell: [i32; 3],
    delta: i32,
    current_score: i32,
    score_delta: F,
) -> AffineState
where
    F: Fn(AffineState, i32) -> i32,
{
    STATES
        .into_iter()
        .find(|state| {
            let transition = score_delta(*state, delta);
            add_score(cell[state.idx()], transition) == current_score
        })
        .expect("affine traceback must have a predecessor")
}

fn gap_transition(gap_open: i32, gap_extend: i32, from: AffineState, to: AffineState) -> i32 {
    debug_assert!(matches!(to, AffineState::GapInT | AffineState::GapInS));
    if from == to {
        gap_extend
    } else if from == AffineState::Match {
        gap_open + gap_extend
    } else {
        NEG_INF
    }
}

fn add_score(base: i32, delta: i32) -> i32 {
    if base <= NEG_INF / 2 || delta <= NEG_INF / 2 {
        NEG_INF
    } else {
        base.saturating_add(delta).max(NEG_INF)
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
    fn affine_regression_path_realizes_score() {
        let problem = NwAffineProblem::new(b"ACB", b"A", 2, 1, -3, -1);
        let (score, path) = HcpEngine::new(problem.clone()).run();
        assert_eq!(score, -3);
        assert_eq!(problem.score_path(&path), Some(score));
    }

    #[test]
    fn empty_source_reconstructs_single_horizontal_gap() {
        let problem = NwAffineProblem::new(b"", b"ABC", 2, 1, -3, -1);
        let (score, path) = HcpEngine::new(problem.clone()).run();
        assert_eq!(score, -6);
        assert_eq!(
            path.first()
                .map(|state| (state.row, state.col, state.state)),
            Some((0, 0, AffineState::Match))
        );
        assert_eq!(
            path.last().map(|state| (state.row, state.col)),
            Some((0, 3))
        );
        assert_eq!(problem.score_path(&path), Some(score));
    }

    #[test]
    fn banded_affine_trace_matches_hcp_score() {
        let problem = NwAffineProblem::new(b"ACGTACGT", b"ACGTTACGT", 2, 1, -3, -1);
        let (hcp_score, _) = HcpEngine::new(problem.clone()).run();
        let trace = trace_banded_affine(&problem, 2).expect("band should contain optimum");
        assert_eq!(trace.score, hcp_score);
        assert_eq!(problem.score_path(&trace.path), Some(trace.score));
    }

    #[test]
    fn banded_affine_trace_rejects_too_narrow_band() {
        let problem = NwAffineProblem::new(b"AAAA", b"AAAATTTT", 2, 1, -3, -1);
        assert!(trace_banded_affine(&problem, 2).is_none());
    }
}
