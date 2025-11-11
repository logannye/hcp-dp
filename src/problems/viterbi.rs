//! Viterbi decoding as an HCP-DP instance.
//!
//! This module illustrates how to use the engine for long-chain probabilistic
//! models with small state spaces.

use crate::traits::{HcpProblem, SummaryApply};
use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

/// Hidden Markov Model with discrete states and emissions.
#[derive(Clone, Debug)]
pub struct Hmm {
    /// Number of states.
    pub n_states: usize,
    /// log initial probabilities [s]
    pub log_pi: Vec<f64>,
    /// log transition probabilities [s_from][s_to]
    pub log_a: Vec<Vec<f64>>,
    /// log emission probabilities [s][obs_symbol]
    pub log_b: Vec<Vec<f64>>,
}

/// Viterbi DP instance: fixed HMM + observation sequence.
#[derive(Clone)]
pub struct ViterbiProblem {
    pub hmm: Arc<Hmm>,
    /// Observations as discrete symbols in 0..V.
    pub obs: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct VitFrontier {
    /// log-probability of best path ending in each state at time t.
    pub log_delta: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct VitSummary {
    hmm: Arc<Hmm>,
    obs_tail: Vec<usize>,
    start_layer: usize,
    end_layer: usize,
}

#[derive(Clone, Debug)]
pub struct VitBoundary {
    /// Time index (layer)
    pub t: usize,
    /// Fixed state at this time
    pub state: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VitState {
    pub t: usize,
    pub state: usize,
}

/// Total-ordering wrapper for f64 to satisfy `Ord` (NaN-safe via `total_cmp`).
#[derive(Clone, Copy, Debug)]
pub struct TotalF64(pub f64);

impl PartialEq for TotalF64 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for TotalF64 {}
impl PartialOrd for TotalF64 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TotalF64 {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}
impl fmt::Display for TotalF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl ViterbiProblem {
    pub fn new(hmm: Hmm, obs: Vec<usize>) -> Self {
        assert_eq!(hmm.log_pi.len(), hmm.n_states);
        Self {
            hmm: Arc::new(hmm),
            obs,
        }
    }

    fn t_len(&self) -> usize {
        self.obs.len()
    }
}

impl HcpProblem for ViterbiProblem {
    type State = VitState;
    type Frontier = VitFrontier;
    type Summary = VitSummary;
    type Boundary = VitBoundary;
    type Cost = TotalF64;

    fn num_layers(&self) -> usize {
        self.t_len()
    }

    fn init_frontier(&self) -> Self::Frontier {
        let o0 = self.obs[0];
        let log_delta = (0..self.hmm.n_states)
            .map(|s| self.hmm.log_pi[s] + self.hmm.log_b[s][o0])
            .collect::<Vec<_>>();
        VitFrontier { log_delta }
    }

    fn forward_step(&self, layer: usize, f: &Self::Frontier) -> Self::Frontier {
        let t = layer + 1; // we already consumed obs[0] in init
        if t >= self.obs.len() {
            return f.clone();
        }
        let obs_sym = self.obs[t];
        let n = self.hmm.n_states;
        let mut next = vec![f64::NEG_INFINITY; n];
        for (s_to, val) in next.iter_mut().enumerate().take(n) {
            let emit = self.hmm.log_b[s_to][obs_sym];
            let mut best = f64::NEG_INFINITY;
            for s_from in 0..n {
                let cand = f.log_delta[s_from] + self.hmm.log_a[s_from][s_to] + emit;
                if cand > best {
                    best = cand;
                }
            }
            *val = best;
        }
        VitFrontier { log_delta: next }
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
        let mut segment = Vec::with_capacity(b.saturating_sub(a));
        let obs_len = self.obs.len();
        for layer in a..b {
            let t = layer + 1;
            if t < obs_len {
                segment.push(self.obs[t]);
            }
        }

        (
            f.clone(),
            VitSummary {
                hmm: Arc::clone(&self.hmm),
                obs_tail: segment,
                start_layer: a,
                end_layer: b,
            },
        )
    }

    fn merge_summary(&self, left: &Self::Summary, right: &Self::Summary) -> Self::Summary {
        debug_assert!(Arc::ptr_eq(&left.hmm, &right.hmm), "HMM must match");
        let mut obs_tail = left.obs_tail.clone();
        obs_tail.extend(&right.obs_tail);
        VitSummary {
            hmm: Arc::clone(&left.hmm),
            obs_tail,
            start_layer: left.start_layer,
            end_layer: right.end_layer,
        }
    }

    fn initial_boundary(&self) -> Self::Boundary {
        VitBoundary {
            t: 0,
            // dummy; actual start state is chosen by reconstruction
            state: 0,
        }
    }

    fn terminal_boundary(&self, frontier_t: &Self::Frontier) -> Self::Boundary {
        let mut best_s = 0;
        let mut best = f64::NEG_INFINITY;
        for (s, &v) in frontier_t.log_delta.iter().enumerate() {
            if v > best {
                best = v;
                best_s = s;
            }
        }
        VitBoundary {
            t: self.t_len() - 1,
            state: best_s,
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
        // For simplicity:
        // - ignore sigma_{left,right},
        // - recompute local Viterbi on [a,c] with constraints at ends,
        // - pick best state at m on a constrained optimal path.
        //
        // This is a reference implementation; production code would optimize.

        let n = self.hmm.n_states;
        let t0 = a;
        let t1 = c;
        let obs = &self.obs[t0..t1];

        // Forward DP with fixed start state (if meaningful).
        // Here, we allow any start and then filter by boundary in reconstruction.
        let mut fwd = vec![vec![f64::NEG_INFINITY; n]; obs.len()];
        for s in 0..n {
            fwd[0][s] = self.hmm.log_pi[s] + self.hmm.log_b[s][obs[0]];
        }
        for t in 1..obs.len() {
            let sym = obs[t];
            for s_to in 0..n {
                let mut best = f64::NEG_INFINITY;
                for s_from in 0..n {
                    let cand = fwd[t - 1][s_from]
                        + self.hmm.log_a[s_from][s_to]
                        + self.hmm.log_b[s_to][sym];
                    if cand > best {
                        best = cand;
                    }
                }
                fwd[t][s_to] = best;
            }
        }

        // Backward DP from c backwards.
        let mut bwd = vec![vec![f64::NEG_INFINITY; n]; obs.len()];
        for s in 0..n {
            bwd[obs.len() - 1][s] = 0.0;
        }
        for t in (0..obs.len() - 1).rev() {
            let sym_next = obs[t + 1];
            for s_from in 0..n {
                let mut best = f64::NEG_INFINITY;
                for s_to in 0..n {
                    let cand = self.hmm.log_a[s_from][s_to]
                        + self.hmm.log_b[s_to][sym_next]
                        + bwd[t + 1][s_to];
                    if cand > best {
                        best = cand;
                    }
                }
                bwd[t][s_from] = best;
            }
        }

        // Choose state at m that lies on some best path from a to c.
        let rel_m = m - t0;
        let mut best_s = 0;
        let mut best = f64::NEG_INFINITY;
        for s in 0..n {
            let cand = fwd[rel_m][s] + bwd[rel_m][s];
            if cand > best {
                best = cand;
                best_s = s;
            }
        }

        let _ = (beta_a, beta_c); // in a more constrained design we'd enforce them.

        VitBoundary {
            t: m,
            state: best_s,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn choose_boundary_with_frontiers(
        &self,
        _a: usize,
        m: usize,
        _c: usize,
        _frontier_a: &Self::Frontier,
        frontier_m_forward: &Self::Frontier,
        frontier_m_backward: &Self::Frontier,
        _frontier_c: &Self::Frontier,
        _sigma_left: &Self::Summary,
        _sigma_right: &Self::Summary,
        _beta_a: &Self::Boundary,
        _beta_c: &Self::Boundary,
    ) -> Self::Boundary {
        let n = self.hmm.n_states;
        let forward = &frontier_m_forward.log_delta;
        let backward = &frontier_m_backward.log_delta;
        let mut best_state = 0;
        let mut best = f64::NEG_INFINITY;

        for state in 0..n {
            let cand = forward[state] + backward[state];
            if cand > best {
                best = cand;
                best_state = state;
            }
        }

        VitBoundary {
            t: m,
            state: best_state,
        }
    }

    fn reconstruct_block(
        &self,
        a: usize,
        b: usize,
        _beta_a: &Self::Boundary,
        beta_b: &Self::Boundary,
    ) -> Vec<Self::State> {
        // Local Viterbi decode on [a,b], end state fixed by beta_b.
        let t0 = a;
        let t1 = b;
        let obs = &self.obs[t0..t1];
        let n = self.hmm.n_states;

        let len = obs.len();
        let mut dp = vec![vec![f64::NEG_INFINITY; n]; len];
        let mut back = vec![vec![0usize; n]; len];

        // init (naively, unconstrained start)
        for s in 0..n {
            dp[0][s] = self.hmm.log_pi[s] + self.hmm.log_b[s][obs[0]];
            back[0][s] = s;
        }

        for t in 1..len {
            let sym = obs[t];
            for s_to in 0..n {
                let mut best = f64::NEG_INFINITY;
                let mut arg = 0;
                for s_from in 0..n {
                    let cand = dp[t - 1][s_from]
                        + self.hmm.log_a[s_from][s_to]
                        + self.hmm.log_b[s_to][sym];
                    if cand > best {
                        best = cand;
                        arg = s_from;
                    }
                }
                dp[t][s_to] = best;
                back[t][s_to] = arg;
            }
        }

        let mut path = Vec::with_capacity(len);
        let mut s = beta_b.state;
        let mut t = len - 1;
        path.push(VitState {
            t: t0 + t,
            state: s,
        });

        while t > 0 {
            let prev = back[t][s];
            t -= 1;
            s = prev;
            path.push(VitState {
                t: t0 + t,
                state: s,
            });
        }

        path.reverse();
        path
    }

    fn extract_cost(&self, frontier_t: &Self::Frontier, beta_t: &Self::Boundary) -> Self::Cost {
        TotalF64(frontier_t.log_delta[beta_t.state])
    }
}

impl SummaryApply<VitFrontier> for VitSummary {
    fn apply(&self, frontier: &VitFrontier) -> VitFrontier {
        if self.obs_tail.is_empty() {
            return frontier.clone();
        }
        let hmm = &self.hmm;
        let n = hmm.n_states;
        let mut prev = frontier.log_delta.clone();
        let mut next = vec![f64::NEG_INFINITY; n];

        for &obs_sym in &self.obs_tail {
            next.iter_mut().for_each(|val| *val = f64::NEG_INFINITY);
            for (s_to, next_val) in next.iter_mut().enumerate().take(n) {
                let emit = hmm.log_b[s_to][obs_sym];
                let mut best = f64::NEG_INFINITY;
                for (s_from, &prev_val) in prev.iter().enumerate().take(n) {
                    let cand = prev_val + hmm.log_a[s_from][s_to] + emit;
                    if cand > best {
                        best = cand;
                    }
                }
                *next_val = best;
            }
            std::mem::swap(&mut prev, &mut next);
        }

        VitFrontier { log_delta: prev }
    }

    fn apply_reverse(&self, frontier: &VitFrontier) -> VitFrontier {
        if self.obs_tail.is_empty() {
            return frontier.clone();
        }
        let hmm = &self.hmm;
        let n = hmm.n_states;
        let mut prev = frontier.log_delta.clone();
        let mut next = vec![f64::NEG_INFINITY; n];

        for &obs_sym in self.obs_tail.iter().rev() {
            next.iter_mut().for_each(|val| *val = f64::NEG_INFINITY);
            for (s_from, next_val) in next.iter_mut().enumerate().take(n) {
                let mut best = f64::NEG_INFINITY;
                for (s_to, &prev_val) in prev.iter().enumerate().take(n) {
                    let cand = hmm.log_a[s_from][s_to] + hmm.log_b[s_to][obs_sym] + prev_val;
                    if cand > best {
                        best = cand;
                    }
                }
                *next_val = best;
            }
            std::mem::swap(&mut prev, &mut next);
        }

        VitFrontier { log_delta: prev }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HcpEngine;
    use std::sync::Arc;

    #[test]
    fn totalf64_ordering() {
        let a = TotalF64(0.0);
        let b = TotalF64(1.0);
        assert!(a < b);
        let nan = TotalF64(f64::NAN);
        // total order is defined; NaN compares deterministically; at least not equal to a
        assert!(nan != a);
    }

    #[test]
    fn init_frontier_first_obs() {
        let hmm = Hmm {
            n_states: 2,
            log_pi: vec![0.0, 0.0],         // log(1.0) for both states
            log_a: vec![vec![0.0, 0.0]; 2], // not used in init
            log_b: vec![vec![0.0, 0.0]; 2], // will be overridden below
        };
        let mut p = ViterbiProblem::new(hmm, vec![0]);
        let hmm_mut = Arc::make_mut(&mut p.hmm);
        hmm_mut.log_b = vec![vec![0.0, -1.0], vec![0.0, -2.0]];
        let f = p.init_frontier();
        assert_eq!(f.log_delta.len(), 2);
        assert!(f.log_delta[0].is_finite());
        assert!(f.log_delta[1].is_finite());
    }

    #[test]
    fn e2e_example_hmm() {
        let hmm = Hmm {
            n_states: 2,
            log_pi: vec![(0.5f64).ln(), (0.5f64).ln()],
            log_a: vec![
                vec![(0.9f64).ln(), (0.1f64).ln()],
                vec![(0.2f64).ln(), (0.8f64).ln()],
            ],
            log_b: vec![
                vec![(0.8f64).ln(), (0.2f64).ln()],
                vec![(0.3f64).ln(), (0.7f64).ln()],
            ],
        };
        let obs = vec![0, 0, 1, 1, 1, 0, 1];
        let problem = ViterbiProblem::new(hmm, obs.clone());
        let engine = HcpEngine::new(problem);
        let (cost, path) = engine.run();
        assert_eq!(path.len(), obs.len());
        // cost is TotalF64 wrapper
        let _ = cost.0;
    }
}
