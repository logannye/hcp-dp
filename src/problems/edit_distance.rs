//! Levenshtein edit distance.

use crate::traits::{HcpProblem, SummaryApply};

#[derive(Clone)]
pub struct EditDistanceProblem<'a> {
    pub s: &'a [u8],
    pub t: &'a [u8],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditDistanceFrontier {
    pub costs: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct EditDistanceSummary<'a> {
    s: &'a [u8],
    t: &'a [u8],
    start: usize,
    end: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditDistanceBoundary {
    pub row: usize,
    pub col: usize,
}

pub type EditDistanceState = (usize, usize);

/// Exact edit-distance traceback produced by a specialized backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditDistanceTrace {
    pub distance: u32,
    pub path: Vec<EditDistanceState>,
    pub band: usize,
}

impl<'a> EditDistanceProblem<'a> {
    pub fn new(s: &'a [u8], t: &'a [u8]) -> Self {
        Self { s, t }
    }

    pub fn n(&self) -> usize {
        self.s.len()
    }

    pub fn m(&self) -> usize {
        self.t.len()
    }

    pub fn score_path(&self, path: &[EditDistanceState]) -> Option<u32> {
        let mut distance = 0;
        for window in path.windows(2) {
            let (a, b) = (window[0], window[1]);
            let di = b.0.checked_sub(a.0)?;
            let dj = b.1.checked_sub(a.1)?;
            match (di, dj) {
                (1, 1) => distance += u32::from(self.s.get(a.0)? != self.t.get(a.1)?),
                (1, 0) | (0, 1) => distance += 1,
                _ => return None,
            }
        }
        Some(distance)
    }

    pub fn full_table_distance(&self) -> u32 {
        distance_linear_space(self.s, self.t)
    }
}

impl<'a> HcpProblem for EditDistanceProblem<'a> {
    type State = EditDistanceState;
    type Frontier = EditDistanceFrontier;
    type Summary = EditDistanceSummary<'a>;
    type Boundary = EditDistanceBoundary;
    type Cost = u32;

    fn num_layers(&self) -> usize {
        self.n()
    }

    fn init_frontier(&self) -> Self::Frontier {
        EditDistanceFrontier {
            costs: (0..=self.m() as u32).collect(),
        }
    }

    fn forward_step(&self, layer: usize, frontier: &Self::Frontier) -> Self::Frontier {
        let ch = self.s[layer];
        let mut next = vec![0; self.m() + 1];
        next[0] = frontier.costs[0] + 1;
        for col in 1..=self.m() {
            let subst = u32::from(ch != self.t[col - 1]);
            let diag = frontier.costs[col - 1] + subst;
            let delete = frontier.costs[col] + 1;
            let insert = next[col - 1] + 1;
            next[col] = diag.min(delete).min(insert);
        }
        EditDistanceFrontier { costs: next }
    }

    fn summarize_interval(&self, a: usize, b: usize) -> Self::Summary {
        assert!(a <= b && b <= self.n(), "invalid edit-distance interval");
        EditDistanceSummary {
            s: self.s,
            t: self.t,
            start: a,
            end: b,
        }
    }

    fn merge_summary(&self, left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
        assert_eq!(
            left.end, right.start,
            "edit-distance summaries must be adjacent"
        );
        assert!(
            std::ptr::eq(left.s, right.s) && std::ptr::eq(left.t, right.t),
            "edit-distance summaries must belong to the same problem"
        );
        EditDistanceSummary {
            s: self.s,
            t: self.t,
            start: left.start,
            end: right.end,
        }
    }

    fn initial_boundary(&self) -> Self::Boundary {
        EditDistanceBoundary { row: 0, col: 0 }
    }

    fn terminal_boundary(&self, _frontier_t: &Self::Frontier) -> Self::Boundary {
        EditDistanceBoundary {
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
        assert!(
            p <= q && q <= self.m(),
            "invalid edit-distance boundary columns"
        );

        let fwd = edit_last_row(&self.s[a..m], &self.t[p..q]);
        let s_rev: Vec<u8> = self.s[m..c].iter().rev().copied().collect();
        let t_rev: Vec<u8> = self.t[p..q].iter().rev().copied().collect();
        let bwd = edit_last_row(&s_rev, &t_rev);

        let width = q - p;
        let mut best_col = p;
        let mut best_cost = u32::MAX;
        for col in 0..=width {
            let cost = fwd[col] + bwd[width - col];
            if cost < best_cost {
                best_cost = cost;
                best_col = p + col;
            }
        }

        EditDistanceBoundary {
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
        assert!(
            p <= q && q <= self.m(),
            "invalid edit-distance leaf columns"
        );

        let s_sub = &self.s[a..b];
        let t_sub = &self.t[p..q];
        let height = s_sub.len();
        let width = t_sub.len();
        let mut dp = vec![vec![0; width + 1]; height + 1];
        for row in 1..=height {
            dp[row][0] = dp[row - 1][0] + 1;
        }
        for col in 1..=width {
            dp[0][col] = dp[0][col - 1] + 1;
        }

        for row in 1..=height {
            for col in 1..=width {
                let subst = u32::from(s_sub[row - 1] != t_sub[col - 1]);
                let diag = dp[row - 1][col - 1] + subst;
                let delete = dp[row - 1][col] + 1;
                let insert = dp[row][col - 1] + 1;
                dp[row][col] = diag.min(delete).min(insert);
            }
        }

        let mut row = height;
        let mut col = width;
        let mut rev_path = Vec::with_capacity(height + width + 1);
        rev_path.push((a + row, p + col));

        while row > 0 || col > 0 {
            if row > 0 && col > 0 {
                let subst = u32::from(s_sub[row - 1] != t_sub[col - 1]);
                if dp[row][col] == dp[row - 1][col - 1] + subst {
                    row -= 1;
                    col -= 1;
                    rev_path.push((a + row, p + col));
                    continue;
                }
            }

            if row > 0 && dp[row][col] == dp[row - 1][col] + 1 {
                row -= 1;
            } else {
                assert!(col > 0, "edit-distance traceback must have predecessor");
                col -= 1;
            }
            rev_path.push((a + row, p + col));
        }

        rev_path.reverse();
        rev_path
    }

    fn extract_cost(&self, frontier_t: &Self::Frontier, _beta_t: &Self::Boundary) -> Self::Cost {
        frontier_t.costs.last().copied().unwrap_or(0)
    }
}

impl<'a> SummaryApply<EditDistanceFrontier> for EditDistanceSummary<'a> {
    fn apply(&self, frontier: &EditDistanceFrontier) -> EditDistanceFrontier {
        assert_eq!(
            frontier.costs.len(),
            self.t.len() + 1,
            "edit-distance frontier width mismatch"
        );
        let mut current = frontier.clone();
        let mut next = vec![0; self.t.len() + 1];
        for layer in self.start..self.end {
            let ch = self.s[layer];
            next[0] = current.costs[0] + 1;
            for col in 1..=self.t.len() {
                let subst = u32::from(ch != self.t[col - 1]);
                let diag = current.costs[col - 1] + subst;
                let delete = current.costs[col] + 1;
                let insert = next[col - 1] + 1;
                next[col] = diag.min(delete).min(insert);
            }
            std::mem::swap(&mut current.costs, &mut next);
        }
        current
    }
}

/// Exact Levenshtein distance using rolling-row dynamic programming.
///
/// This is the standard linear-space score-only baseline: `O(nm)` time and
/// `O(m)` memory for `n = x.len()` and `m = y.len()`.
pub fn distance_linear_space(x: &[u8], y: &[u8]) -> u32 {
    edit_last_row(x, y).last().copied().unwrap_or(0)
}

/// Exact Levenshtein distance within a fixed edit-distance band.
///
/// Returns `Some(distance)` when the true distance is at most `max_distance`.
/// Returns `None` when the distance is larger than that band. Runtime is
/// `O(n * max_distance)` for similarly sized strings, which is the useful
/// regime for near-duplicate sequence or text workloads.
pub fn distance_banded(x: &[u8], y: &[u8], max_distance: usize) -> Option<u32> {
    let n = x.len();
    let m = y.len();
    if n.abs_diff(m) > max_distance {
        return None;
    }
    if n == 0 {
        return (m <= max_distance).then_some(m as u32);
    }
    if m == 0 {
        return (n <= max_distance).then_some(n as u32);
    }

    let inf = max_distance.saturating_add(1) as u32;
    let mut prev_lo = 0usize;
    let mut prev_hi = m.min(max_distance);
    let mut prev: Vec<u32> = (prev_lo..=prev_hi).map(|col| col as u32).collect();

    for row in 1..=n {
        let lo = row.saturating_sub(max_distance);
        let hi = m.min(row.saturating_add(max_distance));
        let mut curr = vec![inf; hi - lo + 1];

        for col in lo..=hi {
            let idx = col - lo;
            if col == 0 {
                curr[idx] = row as u32;
                continue;
            }

            let subst = u32::from(x[row - 1] != y[col - 1]);
            let diag = band_get(&prev, prev_lo, prev_hi, col - 1, inf).saturating_add(subst);
            let delete = band_get(&prev, prev_lo, prev_hi, col, inf).saturating_add(1);
            let insert = if idx > 0 {
                curr[idx - 1].saturating_add(1)
            } else {
                inf
            };
            curr[idx] = diag.min(delete).min(insert).min(inf);
        }

        if curr.iter().all(|&cost| cost > max_distance as u32) {
            return None;
        }

        prev = curr;
        prev_lo = lo;
        prev_hi = hi;
    }

    let distance = band_get(&prev, prev_lo, prev_hi, m, inf);
    (distance <= max_distance as u32).then_some(distance)
}

/// Exact Levenshtein distance and traceback within a fixed edit-distance band.
///
/// Returns `Some(trace)` when the true distance is at most `max_distance`.
/// Returns `None` when the optimum lies outside that band.
pub fn trace_banded(x: &[u8], y: &[u8], max_distance: usize) -> Option<EditDistanceTrace> {
    let n = x.len();
    let m = y.len();
    if n.abs_diff(m) > max_distance {
        return None;
    }

    let inf = max_distance.saturating_add(1) as u32;
    let mut rows = Vec::with_capacity(n + 1);
    let row0_hi = m.min(max_distance);
    rows.push(BandRow {
        lo: 0,
        costs: (0..=row0_hi).map(|col| col as u32).collect(),
    });

    for row in 1..=n {
        let prev = &rows[row - 1];
        let lo = row.saturating_sub(max_distance);
        let hi = m.min(row.saturating_add(max_distance));
        let mut costs = vec![inf; hi - lo + 1];

        for col in lo..=hi {
            let idx = col - lo;
            if col == 0 {
                costs[idx] = row as u32;
                continue;
            }

            let subst = u32::from(x[row - 1] != y[col - 1]);
            let diag = prev.get(col - 1, inf).saturating_add(subst);
            let delete = prev.get(col, inf).saturating_add(1);
            let insert = if idx > 0 {
                costs[idx - 1].saturating_add(1)
            } else {
                inf
            };
            costs[idx] = diag.min(delete).min(insert).min(inf);
        }

        if costs.iter().all(|&cost| cost > max_distance as u32) {
            return None;
        }
        rows.push(BandRow { lo, costs });
    }

    let distance = rows[n].get(m, inf);
    if distance > max_distance as u32 {
        return None;
    }

    let mut row = n;
    let mut col = m;
    let mut rev_path = Vec::with_capacity(n + m + 1);
    rev_path.push((row, col));

    while row > 0 || col > 0 {
        let here = rows[row].get(col, inf);

        if row > 0 && col > 0 {
            let subst = u32::from(x[row - 1] != y[col - 1]);
            if here == rows[row - 1].get(col - 1, inf).saturating_add(subst) {
                row -= 1;
                col -= 1;
                rev_path.push((row, col));
                continue;
            }
        }

        if row > 0 && here == rows[row - 1].get(col, inf).saturating_add(1) {
            row -= 1;
            rev_path.push((row, col));
            continue;
        }

        assert!(
            col > 0,
            "banded edit-distance traceback must have predecessor"
        );
        col -= 1;
        assert_eq!(
            here,
            rows[row].get(col, inf).saturating_add(1),
            "banded edit-distance traceback left the optimal predecessor set"
        );
        rev_path.push((row, col));
    }

    rev_path.reverse();
    Some(EditDistanceTrace {
        distance,
        path: rev_path,
        band: max_distance,
    })
}

/// Exact adaptive-banded Levenshtein distance.
///
/// The band starts at the length difference lower bound and grows
/// geometrically until it contains the optimum. This keeps exactness while
/// giving `O(n * s)` behavior for final edit distance `s`, up to the standard
/// `O(nm)` worst case.
pub fn distance_adaptive_banded(x: &[u8], y: &[u8]) -> u32 {
    let max_len = x.len().max(y.len());
    if max_len == 0 {
        return 0;
    }

    let mut band = x.len().abs_diff(y.len());
    loop {
        if let Some(distance) = distance_banded(x, y, band) {
            return distance;
        }
        if band >= max_len {
            return distance_linear_space(x, y);
        }
        band = if band == 0 {
            1
        } else {
            band.saturating_mul(2).saturating_add(1).min(max_len)
        };
    }
}

/// Exact adaptive-banded Levenshtein distance with traceback.
///
/// The returned path is exact. Runtime is `O(n*s)` in low-edit regimes after
/// geometric band growth, with quadratic worst-case fallback when the optimum
/// is not near the main diagonal.
pub fn trace_adaptive_banded(x: &[u8], y: &[u8]) -> EditDistanceTrace {
    let max_len = x.len().max(y.len());
    if max_len == 0 {
        return EditDistanceTrace {
            distance: 0,
            path: vec![(0, 0)],
            band: 0,
        };
    }

    let mut band = x.len().abs_diff(y.len());
    loop {
        if let Some(trace) = trace_banded(x, y, band) {
            return trace;
        }
        if band >= max_len {
            return trace_banded(x, y, max_len)
                .expect("full-width band must contain the edit-distance optimum");
        }
        band = if band == 0 {
            1
        } else {
            band.saturating_mul(2).saturating_add(1).min(max_len)
        };
    }
}

/// Exact Myers bit-vector Levenshtein distance for arbitrary pattern lengths.
///
/// This is a distance-only backend. It computes the exact edit distance in
/// `O(ceil(pattern.len() / 64) * text.len())` word operations and stores no
/// traceback path. Use [`EditDistanceProblem`] with [`crate::HcpEngine`] when
/// an exact reconstructed path is required.
pub fn distance_myers(pattern: &[u8], text: &[u8]) -> u32 {
    const WORD_BITS: usize = u64::BITS as usize;

    let m = pattern.len();
    if m == 0 {
        return text.len() as u32;
    }
    if m <= WORD_BITS {
        return distance_myers_u64(pattern, text).expect("short pattern must fit in one word");
    }

    let word_count = m.div_ceil(WORD_BITS);
    let last_word = word_count - 1;
    let last_bits = m % WORD_BITS;
    let last_mask = if last_bits == 0 {
        u64::MAX
    } else {
        (1u64 << last_bits) - 1
    };
    let high_bit = 1u64 << ((m - 1) % u64::BITS as usize);

    let mut peq = vec![[0u64; 256]; word_count];
    for (idx, &byte) in pattern.iter().enumerate() {
        peq[idx / WORD_BITS][byte as usize] |= 1u64 << (idx % WORD_BITS);
    }

    let mut vp = vec![u64::MAX; word_count];
    vp[last_word] = last_mask;
    let mut vn = vec![0u64; word_count];
    let mut d0 = vec![0u64; word_count];
    let mut hp = vec![0u64; word_count];
    let mut hn = vec![0u64; word_count];
    let mut next_vp = vec![0u64; word_count];
    let mut next_vn = vec![0u64; word_count];
    let mut score = m as u32;

    for &byte in text {
        let mut carry = 0u128;
        for word in 0..word_count {
            let mask = if word == last_word {
                last_mask
            } else {
                u64::MAX
            };
            let eq = peq[word][byte as usize];
            let x = eq | vn[word];
            let sum = ((x & vp[word]) as u128) + (vp[word] as u128) + carry;
            carry = sum >> u64::BITS;

            let d = (((sum as u64) ^ vp[word]) | x) & mask;
            d0[word] = d;
            hp[word] = (vn[word] | !(d | vp[word])) & mask;
            hn[word] = (vp[word] & d) & mask;
        }

        if hp[last_word] & high_bit != 0 {
            score += 1;
        } else if hn[last_word] & high_bit != 0 {
            score -= 1;
        }

        let mut hp_shift_in = 1u64;
        let mut hn_shift_in = 0u64;
        for word in 0..word_count {
            let mask = if word == last_word {
                last_mask
            } else {
                u64::MAX
            };
            let shifted_hp = ((hp[word] << 1) | hp_shift_in) & mask;
            hp_shift_in = hp[word] >> (u64::BITS - 1);
            let shifted_hn = ((hn[word] << 1) | hn_shift_in) & mask;
            hn_shift_in = hn[word] >> (u64::BITS - 1);

            next_vp[word] = (shifted_hn | !(d0[word] | shifted_hp)) & mask;
            next_vn[word] = shifted_hp & d0[word];
        }

        std::mem::swap(&mut vp, &mut next_vp);
        std::mem::swap(&mut vn, &mut next_vn);
    }

    score
}

/// Exact Myers single-word bit-vector Levenshtein distance.
///
/// This is a fast exact distance-only backend for patterns up to 64 symbols.
/// It is useful as a small-pattern proof point and as a building block for a
/// future multiword bit-parallel summary backend.
pub fn distance_myers_u64(pattern: &[u8], text: &[u8]) -> Option<u32> {
    let m = pattern.len();
    if m > u64::BITS as usize {
        return None;
    }
    if m == 0 {
        return Some(text.len() as u32);
    }

    let active_mask = if m == u64::BITS as usize {
        u64::MAX
    } else {
        (1u64 << m) - 1
    };
    let high_bit = 1u64 << (m - 1);
    let mut peq = [0u64; 256];
    for (idx, &byte) in pattern.iter().enumerate() {
        peq[byte as usize] |= 1u64 << idx;
    }

    let mut vp = active_mask;
    let mut vn = 0u64;
    let mut score = m as u32;

    for &byte in text {
        let eq = peq[byte as usize];
        let x = eq | vn;
        let d0 = ((((x & vp).wrapping_add(vp)) ^ vp) | x) & active_mask;
        let hp = (vn | !(d0 | vp)) & active_mask;
        let hn = vp & d0;

        if hp & high_bit != 0 {
            score += 1;
        } else if hn & high_bit != 0 {
            score -= 1;
        }

        let shifted_hp = ((hp << 1) | 1) & active_mask;
        let shifted_hn = (hn << 1) & active_mask;
        vp = (shifted_hn | !(d0 | shifted_hp)) & active_mask;
        vn = shifted_hp & d0;
    }

    Some(score)
}

fn band_get(row: &[u32], lo: usize, hi: usize, col: usize, inf: u32) -> u32 {
    if (lo..=hi).contains(&col) {
        row[col - lo]
    } else {
        inf
    }
}

#[derive(Clone, Debug)]
struct BandRow {
    lo: usize,
    costs: Vec<u32>,
}

impl BandRow {
    fn hi(&self) -> usize {
        self.lo + self.costs.len() - 1
    }

    fn get(&self, col: usize, inf: u32) -> u32 {
        if (self.lo..=self.hi()).contains(&col) {
            self.costs[col - self.lo]
        } else {
            inf
        }
    }
}

fn edit_last_row(x: &[u8], y: &[u8]) -> Vec<u32> {
    let mut prev: Vec<u32> = (0..=y.len() as u32).collect();
    let mut curr = vec![0; y.len() + 1];
    for (row, &cx) in x.iter().enumerate() {
        curr[0] = row as u32 + 1;
        for col in 1..=y.len() {
            let subst = u32::from(cx != y[col - 1]);
            let diag = prev[col - 1] + subst;
            let delete = prev[col] + 1;
            let insert = curr[col - 1] + 1;
            curr[col] = diag.min(delete).min(insert);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HcpEngine;

    #[test]
    fn kitten_sitting_distance_is_three() {
        let problem = EditDistanceProblem::new(b"kitten", b"sitting");
        let (distance, path) = HcpEngine::new(problem.clone()).run();
        assert_eq!(distance, 3);
        assert_eq!(problem.score_path(&path), Some(distance));
    }

    #[test]
    fn empty_source_reconstructs_insertions() {
        let problem = EditDistanceProblem::new(b"", b"ABC");
        let (distance, path) = HcpEngine::new(problem.clone()).run();
        assert_eq!(distance, 3);
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(0, 3)));
        assert_eq!(problem.score_path(&path), Some(distance));
    }

    #[test]
    fn adaptive_banded_distance_matches_linear_space() {
        let cases: &[(&[u8], &[u8])] = &[
            (b"", b""),
            (b"", b"ABC"),
            (b"ABC", b""),
            (b"kitten", b"sitting"),
            (b"ACGTACGT", b"ACGTTCGT"),
            (b"AAAAAAAAAAAAAAAA", b"AAAAAAAAAAAAAAAT"),
            (b"ACGTACGT", b"TTTTACGTACGTAAAA"),
            (b"GATTACA", b"GCATGCU"),
        ];

        for (x, y) in cases {
            assert_eq!(
                distance_adaptive_banded(x, y),
                distance_linear_space(x, y),
                "adaptive banded mismatch for {:?} vs {:?}",
                std::str::from_utf8(x).unwrap_or("<bytes>"),
                std::str::from_utf8(y).unwrap_or("<bytes>")
            );
        }
    }

    #[test]
    fn adaptive_banded_trace_matches_linear_space_and_scores_path() {
        let cases: &[(&[u8], &[u8])] = &[
            (b"", b""),
            (b"", b"ABC"),
            (b"ABC", b""),
            (b"kitten", b"sitting"),
            (b"ACGTACGT", b"ACGTTCGT"),
            (b"AAAAAAAAAAAAAAAA", b"AAAAAAAAAAAAAAAT"),
            (b"ACGTACGT", b"TTTTACGTACGTAAAA"),
            (b"GATTACA", b"GCATGCU"),
        ];

        for (x, y) in cases {
            let trace = trace_adaptive_banded(x, y);
            let problem = EditDistanceProblem::new(x, y);
            assert_eq!(trace.distance, distance_linear_space(x, y));
            assert_eq!(trace.path.first(), Some(&(0, 0)));
            assert_eq!(trace.path.last(), Some(&(x.len(), y.len())));
            assert_eq!(problem.score_path(&trace.path), Some(trace.distance));
        }
    }

    #[test]
    fn fixed_band_reports_none_when_too_small() {
        assert_eq!(distance_banded(b"kitten", b"sitting", 2), None);
        assert_eq!(distance_banded(b"kitten", b"sitting", 3), Some(3));
        assert_eq!(distance_banded(b"AAAA", b"TTTT", 3), None);
        assert_eq!(distance_banded(b"AAAA", b"TTTT", 4), Some(4));
        assert!(trace_banded(b"kitten", b"sitting", 2).is_none());
        let trace = trace_banded(b"kitten", b"sitting", 3).expect("band contains optimum");
        assert_eq!(trace.distance, 3);
        assert_eq!(
            EditDistanceProblem::new(b"kitten", b"sitting").score_path(&trace.path),
            Some(3)
        );
    }

    #[test]
    fn myers_u64_matches_linear_space_for_short_patterns() {
        let cases: &[(&[u8], &[u8])] = &[
            (b"", b"ABC"),
            (b"A", b""),
            (b"A", b"A"),
            (b"A", b"T"),
            (b"kitten", b"sitting"),
            (b"GATTACA", b"GCATGCU"),
            (b"ACGTACGTACGT", b"ACGTTCGTACGA"),
            (
                b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                b"AAAAAAAATAAAAAAAAAAAAAAATAAAAA",
            ),
        ];

        for (pattern, text) in cases {
            assert_eq!(
                distance_myers_u64(pattern, text),
                Some(distance_linear_space(pattern, text)),
                "Myers mismatch for {:?} vs {:?}",
                std::str::from_utf8(pattern).unwrap_or("<bytes>"),
                std::str::from_utf8(text).unwrap_or("<bytes>")
            );
        }
    }

    #[test]
    fn myers_u64_declines_long_patterns() {
        let pattern = vec![b'A'; 65];
        assert_eq!(distance_myers_u64(&pattern, b"A"), None);
    }

    #[test]
    fn myers_arbitrary_matches_u64_for_short_patterns() {
        let cases: &[(&[u8], &[u8])] = &[
            (b"", b"ABC"),
            (b"A", b""),
            (b"kitten", b"sitting"),
            (b"GATTACA", b"GCATGCU"),
            (
                b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                b"AAAAAAAATAAAAAAAAAAAAAAATAAAAAAAAAAAAAAATAAAAAAAAAAAAAAATAAAA",
            ),
        ];

        for (pattern, text) in cases {
            assert_eq!(
                distance_myers(pattern, text),
                distance_myers_u64(pattern, text).unwrap(),
                "arbitrary Myers mismatch for {:?} vs {:?}",
                std::str::from_utf8(pattern).unwrap_or("<bytes>"),
                std::str::from_utf8(text).unwrap_or("<bytes>")
            );
        }
    }

    #[test]
    fn myers_arbitrary_matches_linear_space_for_long_patterns() {
        let cases: Vec<(Vec<u8>, Vec<u8>)> = vec![
            (patterned(65, 0), patterned(65, 1)),
            (patterned(127, 0), patterned(127, 2)),
            (patterned(128, 0), patterned(128, 3)),
            (patterned(129, 0), patterned(129, 1)),
            (patterned(191, 0), patterned(257, 2)),
            (vec![b'A'; 256], vec![b'T'; 256]),
            (
                [patterned(64, 0), vec![b'A'; 128], patterned(64, 1)].concat(),
                [patterned(64, 0), vec![b'A'; 64], patterned(64, 1)].concat(),
            ),
        ];

        for (pattern, text) in cases {
            assert_eq!(
                distance_myers(&pattern, &text),
                distance_linear_space(&pattern, &text),
                "arbitrary Myers mismatch for lengths {} and {}",
                pattern.len(),
                text.len()
            );
        }
    }

    #[test]
    fn myers_arbitrary_matches_linear_space_for_exhaustive_small_binary_strings() {
        let strings = binary_strings(5);
        for pattern in &strings {
            for text in &strings {
                assert_eq!(
                    distance_myers(pattern, text),
                    distance_linear_space(pattern, text),
                    "arbitrary Myers mismatch for {:?} vs {:?}",
                    std::str::from_utf8(pattern).unwrap_or("<bytes>"),
                    std::str::from_utf8(text).unwrap_or("<bytes>")
                );
            }
        }
    }

    fn patterned(len: usize, offset: usize) -> Vec<u8> {
        const ALPHABET: &[u8] = b"ACGT";
        (0..len)
            .map(|idx| ALPHABET[(idx.wrapping_mul(7) + offset) % ALPHABET.len()])
            .collect()
    }

    fn binary_strings(max_len: usize) -> Vec<Vec<u8>> {
        let mut strings = Vec::new();
        for len in 0..=max_len {
            for bits in 0..(1usize << len) {
                let mut s = Vec::with_capacity(len);
                for idx in 0..len {
                    let byte = if bits & (1usize << idx) == 0 {
                        b'A'
                    } else {
                        b'C'
                    };
                    s.push(byte);
                }
                strings.push(s);
            }
        }
        strings
    }
}
