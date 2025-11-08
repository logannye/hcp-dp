use hcp_dp::{
    problems::{nw_affine::NwAffineProblem, nw_align::NwProblem},
    HcpEngine,
};
use proptest::prelude::*;

fn full_linear_score(
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

fn full_affine_score(
    s: &[u8],
    t: &[u8],
    match_score: i32,
    mismatch_penalty: i32,
    gap_open_penalty: i32,
    gap_extend_penalty: i32,
) -> i32 {
    let n = s.len();
    let m = t.len();
    let neg_inf = i32::MIN / 4;
    let mut m_dp = vec![vec![neg_inf; m + 1]; n + 1];
    let mut ix_dp = vec![vec![neg_inf; m + 1]; n + 1];
    let mut iy_dp = vec![vec![neg_inf; m + 1]; n + 1];
    m_dp[0][0] = 0;
    for j in 1..=m {
        if j == 1 {
            iy_dp[0][j] = gap_open_penalty + gap_extend_penalty;
        } else {
            iy_dp[0][j] = iy_dp[0][j - 1] + gap_extend_penalty;
        }
    }
    for i in 1..=n {
        if i == 1 {
            ix_dp[i][0] = gap_open_penalty + gap_extend_penalty;
        } else {
            ix_dp[i][0] = ix_dp[i - 1][0] + gap_extend_penalty;
        }
    }
    for i in 1..=n {
        for j in 1..=m {
            let score = if s[i - 1] == t[j - 1] {
                match_score
            } else {
                -mismatch_penalty
            };
            m_dp[i][j] = (m_dp[i - 1][j - 1]
                .max(ix_dp[i - 1][j - 1])
                .max(iy_dp[i - 1][j - 1]))
                + score;
            ix_dp[i][j] = (m_dp[i - 1][j] + gap_open_penalty + gap_extend_penalty)
                .max(ix_dp[i - 1][j] + gap_extend_penalty);
            iy_dp[i][j] = (m_dp[i][j - 1] + gap_open_penalty + gap_extend_penalty)
                .max(iy_dp[i][j - 1] + gap_extend_penalty);
        }
    }
    *[m_dp[n][m], ix_dp[n][m], iy_dp[n][m]].iter().max().unwrap()
}

proptest! {
    #[test]
    fn linear_matches_full(a in "[ACGT]{0,8}", b in "[ACGT]{0,8}") {
        let s = a.as_bytes();
        let t = b.as_bytes();
        let ms = 1;
        let mm = 1;
        let gp = -1;
        let problem = NwProblem::new(s, t, ms, mm, gp);
        let (score, _) = HcpEngine::new(problem).run();
        prop_assert_eq!(score, full_linear_score(s, t, ms, mm, gp));
    }

    #[test]
    fn affine_matches_full(a in "[ACGT]{0,6}", b in "[ACGT]{0,6}") {
        let s = a.as_bytes();
        let t = b.as_bytes();
        let ms = 2;
        let mm = 1;
        let go = -3;
        let ge = -1;
        let problem = NwAffineProblem::new(s, t, ms, mm, go, ge);
        let (score, _) = HcpEngine::new(problem).run();
        prop_assert_eq!(score, full_affine_score(s, t, ms, mm, go, ge));
    }
}

#[test]
fn all_gaps_affine_edges() {
    let s = b"AAAA";
    let t = b"";
    let ms = 1;
    let mm = 1;
    let go = -3;
    let ge = -1;
    let (score, _) = HcpEngine::new(NwAffineProblem::new(s, t, ms, mm, go, ge)).run();
    assert_eq!(score, full_affine_score(s, t, ms, mm, go, ge));
}

#[test]
fn homopolymer_gap_extensions() {
    let s = b"AAAAAA";
    let t = b"AAA";
    let ms = 2;
    let mm = 1;
    let go = -5;
    let ge = -1;
    let (score, _) = HcpEngine::new(NwAffineProblem::new(s, t, ms, mm, go, ge)).run();
    assert_eq!(score, full_affine_score(s, t, ms, mm, go, ge));
}

#[test]
fn appending_identical_char_does_not_reduce_score() {
    let s = b"GATTACA";
    let t = b"GCATGCU";
    let ms = 1;
    let mm = 1;
    let gp = -1;
    let base = HcpEngine::new(NwProblem::new(s, t, ms, mm, gp)).run().0;
    let mut s_ext = s.to_vec();
    s_ext.push(b'A');
    let mut t_ext = t.to_vec();
    t_ext.push(b'A');
    let extended = HcpEngine::new(NwProblem::new(&s_ext, &t_ext, ms, mm, gp))
        .run()
        .0;
    assert!(extended >= base);
}
