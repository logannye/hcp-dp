use hcp_dp::{
    problems::{
        lcs::LcsProblem,
        nw_align::NwProblem,
        viterbi::{Hmm, ViterbiProblem},
    },
    HcpEngine,
};
use proptest::prelude::*;

fn path_monotone(path: &[(usize, usize)]) -> bool {
    path.windows(2).all(|w| {
        let (a, b) = (w[0], w[1]);
        let di = b.0 as isize - a.0 as isize;
        let dj = b.1 as isize - a.1 as isize;
        matches!((di, dj), (1, 0) | (0, 1) | (1, 1))
    })
}

fn full_lcs_len(s: &[u8], t: &[u8]) -> u32 {
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

fn full_nw_score(
    s: &[u8],
    t: &[u8],
    match_score: i32,
    mismatch_penalty: i32,
    gap_penalty: i32,
) -> i32 {
    let n = s.len();
    let m = t.len();
    let mut dp = vec![vec![0i32; m + 1]; n + 1];
    for i in 1..=n {
        dp[i][0] = dp[i - 1][0] + gap_penalty;
    }
    for j in 1..=m {
        dp[0][j] = dp[0][j - 1] + gap_penalty;
    }
    for i in 1..=n {
        for j in 1..=m {
            let score = if s[i - 1] == t[j - 1] {
                match_score
            } else {
                -mismatch_penalty
            };
            let diag = dp[i - 1][j - 1] + score;
            let up = dp[i - 1][j] + gap_penalty;
            let left = dp[i][j - 1] + gap_penalty;
            dp[i][j] = diag.max(up).max(left);
        }
    }
    dp[n][m]
}

fn demo_hmm() -> Hmm {
    Hmm {
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
    }
}

proptest! {
    #[test]
    fn lcs_matches_full_dp(a in "[ACGT]{0,10}", b in "[ACGT]{0,10}") {
        let s = a.as_bytes();
        let t = b.as_bytes();
        let problem = LcsProblem::new(s, t);
        let engine = HcpEngine::new(problem);
        let (len, _path) = engine.run();
        prop_assert_eq!(len, full_lcs_len(s, t));
    }

    #[test]
    fn nw_matches_full_dp(a in "[ACGT]{0,8}", b in "[ACGT]{0,8}") {
        let s = a.as_bytes();
        let t = b.as_bytes();
        let ms = 1;
        let mm = 1;
        let gp = -1;
        let problem = NwProblem::new(s, t, ms, mm, gp);
        let engine = HcpEngine::new(problem);
        let (score, path) = engine.run();
        prop_assert!(path_monotone(&path));
        prop_assert_eq!(score, full_nw_score(s, t, ms, mm, gp));
    }

    #[test]
    fn viterbi_path_len_consistent(obs in proptest::collection::vec(0usize..2, 1..20)) {
        let problem = ViterbiProblem::new(demo_hmm(), obs.clone());
        let engine = HcpEngine::new(problem);
        let (_logp, path) = engine.run();
        prop_assert_eq!(path.len(), obs.len());
    }
}
