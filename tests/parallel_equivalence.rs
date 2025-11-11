#![cfg(feature = "parallel")]

use hcp_dp::{
    builder::HcpEngineBuilder,
    problems::{
        dag_sp::DagLayered,
        lcs::LcsProblem,
        nw_align::NwProblem,
        viterbi::{Hmm, ViterbiProblem},
    },
    HcpEngine,
};
use proptest::prelude::*;

fn full_lcs(s: &[u8], t: &[u8]) -> u32 {
    let n = s.len();
    let m = t.len();
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            let up = dp[i - 1][j];
            let left = dp[i][j - 1];
            let diag = dp[i - 1][j - 1] + if s[i - 1] == t[j - 1] { 1 } else { 0 };
            dp[i][j] = up.max(left).max(diag);
        }
    }
    dp[n][m]
}

fn full_nw(s: &[u8], t: &[u8], ms: i32, mm: i32, gp: i32) -> i32 {
    let n = s.len();
    let m = t.len();
    let mut dp = vec![vec![0i32; m + 1]; n + 1];
    for i in 1..=n {
        dp[i][0] = dp[i - 1][0] + gp;
    }
    for j in 1..=m {
        dp[0][j] = dp[0][j - 1] + gp;
    }
    for i in 1..=n {
        for j in 1..=m {
            let score = if s[i - 1] == t[j - 1] { ms } else { -mm };
            let diag = dp[i - 1][j - 1] + score;
            let up = dp[i - 1][j] + gp;
            let left = dp[i][j - 1] + gp;
            dp[i][j] = diag.max(up).max(left);
        }
    }
    dp[n][m]
}

fn topo_relax(adjacency: &[Vec<Vec<(usize, i64)>>], widths: &[usize]) -> Vec<i64> {
    let mut dist = vec![i64::MAX / 4; widths[0]];
    if !dist.is_empty() {
        dist[0] = 0;
    }
    for (layer, edges) in adjacency.iter().enumerate() {
        let mut next = vec![i64::MAX / 4; widths[layer + 1]];
        for (u, &du) in dist.iter().enumerate() {
            if du >= i64::MAX / 8 {
                continue;
            }
            for &(v, w) in &edges[u] {
                let cand = du.saturating_add(w);
                if cand < next[v] {
                    next[v] = cand;
                }
            }
        }
        dist = next;
    }
    dist
}

fn viterbi_baseline(hmm: &Hmm, obs: &[usize]) -> (f64, Vec<usize>) {
    if obs.is_empty() {
        return (0.0, Vec::new());
    }
    let n = hmm.n_states;
    let t = obs.len();
    let mut dp = vec![vec![f64::NEG_INFINITY; n]; t];
    let mut back = vec![vec![0usize; n]; t];
    for s in 0..n {
        dp[0][s] = hmm.log_pi[s] + hmm.log_b[s][obs[0]];
    }
    for time in 1..t {
        for s_to in 0..n {
            let emit = hmm.log_b[s_to][obs[time]];
            let mut best = f64::NEG_INFINITY;
            let mut arg = 0;
            for s_from in 0..n {
                let cand = dp[time - 1][s_from] + hmm.log_a[s_from][s_to] + emit;
                if cand > best {
                    best = cand;
                    arg = s_from;
                }
            }
            dp[time][s_to] = best;
            back[time][s_to] = arg;
        }
    }
    let mut best_state = 0;
    let mut best = f64::NEG_INFINITY;
    for s in 0..n {
        if dp[t - 1][s] > best {
            best = dp[t - 1][s];
            best_state = s;
        }
    }
    let mut path = Vec::with_capacity(t);
    let mut cur = best_state;
    for time in (0..t).rev() {
        path.push(cur);
        if time > 0 {
            cur = back[time][cur];
        }
    }
    path.reverse();
    (best, path)
}

proptest! {
    #[test]
    fn lcs_parallel_matches_baseline(a in "[ACGT]{0,10}", b in "[ACGT]{0,10}") {
        let s = a.as_bytes();
        let t = b.as_bytes();
        let serial = HcpEngine::new(LcsProblem::new(s, t)).run();
        let parallel = HcpEngineBuilder::new(LcsProblem::new(s, t)).build().run();
        prop_assert_eq!(serial.0, full_lcs(s, t));
        if !serial.1.is_empty() {
            prop_assert_eq!(serial.1.first(), Some(&(0, 0)));
        }
        prop_assert_eq!(serial.0, parallel.0);
        prop_assert_eq!(serial.1, parallel.1);
    }

    #[test]
    fn nw_parallel_matches_baseline(a in "[ACGT]{0,8}", b in "[ACGT]{0,8}") {
        let s = a.as_bytes();
        let t = b.as_bytes();
        let ms = 1;
        let mm = 1;
        let gp = -1;
        let serial = HcpEngine::new(NwProblem::new(s, t, ms, mm, gp)).run();
        let parallel = HcpEngineBuilder::new(NwProblem::new(s, t, ms, mm, gp))
            .build()
            .run();
        prop_assert_eq!(serial.0, full_nw(s, t, ms, mm, gp));
        prop_assert_eq!(serial.0, parallel.0);
    }

    #[test]
    fn dag_parallel_matches_baseline(
        layers in 1usize..4,
        widths in prop::collection::vec(1usize..4, 0usize..5),
        weights in prop::collection::vec(-3i64..6, 0usize..40)
    ) {
        let mut widths = widths;
        if widths.len() < layers + 1 {
            widths.resize(layers + 1, 1);
        } else {
            widths.truncate(layers + 1);
        }
        let mut idx = 0;
        let mut adjacency = Vec::with_capacity(layers);
        for layer in 0..layers {
            let w = widths[layer];
            let next_w = widths[layer + 1];
            let mut layer_edges = Vec::with_capacity(w);
            for _ in 0..w {
                let mut edges = Vec::new();
                for v in 0..next_w {
                    let weight = if idx < weights.len() { weights[idx] } else { 1 };
                    idx += 1;
                    edges.push((v, weight));
                }
                layer_edges.push(edges);
            }
            adjacency.push(layer_edges);
        }
        let baseline = topo_relax(&adjacency, &widths);
        let serial = HcpEngine::new(DagLayered::new(adjacency.clone(), widths.clone())).run();
        let parallel = HcpEngineBuilder::new(DagLayered::new(adjacency, widths.clone()))
            .build()
            .run();
        let best = *baseline.iter().min().unwrap();
        prop_assert_eq!(serial.0, best);
        prop_assert_eq!(parallel.0, best);
    }
}

#[test]
fn viterbi_parallel_matches_baseline() {
    let hmm = Hmm {
        n_states: 3,
        log_pi: vec![(0.4f64).ln(), (0.4f64).ln(), (0.2f64).ln()],
        log_a: vec![
            vec![(0.5f64).ln(), (0.3f64).ln(), (0.2f64).ln()],
            vec![(0.3f64).ln(), (0.5f64).ln(), (0.2f64).ln()],
            vec![(0.2f64).ln(), (0.3f64).ln(), (0.5f64).ln()],
        ],
        log_b: vec![
            vec![(0.6f64).ln(), (0.2f64).ln()],
            vec![(0.3f64).ln(), (0.7f64).ln()],
            vec![(0.5f64).ln(), (0.5f64).ln()],
        ],
    };
    let obs = vec![0, 1, 0, 1, 1];
    let (logp, path) = HcpEngine::new(ViterbiProblem::new(hmm.clone(), obs.clone())).run();
    let (baseline_logp, baseline_path) = viterbi_baseline(&hmm, &obs);
    assert!((logp.0 - baseline_logp).abs() < 1e-9);
    assert_eq!(
        path.iter().map(|s| s.state).collect::<Vec<_>>(),
        baseline_path
    );
}
