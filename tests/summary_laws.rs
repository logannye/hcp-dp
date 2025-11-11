use hcp_dp::{
    problems::{
        dag_sp::DagLayered,
        lcs::LcsProblem,
        nw_align::NwProblem,
        viterbi::{Hmm, ViterbiProblem},
    },
    traits::{HcpProblem, SummaryApply},
};
use proptest::prelude::*;

fn frontier_at<P: HcpProblem>(problem: &P, layer: usize) -> P::Frontier {
    let mut frontier = problem.init_frontier();
    for i in 0..layer {
        frontier = problem.forward_step(i, &frontier);
    }
    frontier
}

fn split_points(
    total: usize,
    seeds: (usize, usize, usize),
) -> Option<(usize, usize, usize, usize)> {
    if total < 3 {
        return None;
    }
    let a = seeds.0 % (total - 2);
    let b = a + 1 + (seeds.1 % (total - a - 1));
    let c = b + 1 + (seeds.2 % (total - b));
    let d = total;
    Some((a, b, c, d))
}

mod lcs_laws {
    use super::*;

    proptest! {
        #[test]
        fn summary_laws(s in "[ACGT]{0,6}", t in "[ACGT]{0,6}", s0 in 0usize..6, s1 in 0usize..6, s2 in 0usize..6) {
            let s = s.as_bytes();
            let t = t.as_bytes();
            let problem = LcsProblem::new(s, t);
            let total = problem.num_layers();
            let Some((a,b,c,d)) = super::split_points(total, (s0,s1,s2)) else {
                return Ok(());
            };

            let frontier_a = frontier_at(&problem, a);
            let (frontier_b, sigma_ab) = problem.summarize_block(a, b, &frontier_a);
            let (frontier_c, sigma_bc) = problem.summarize_block(b, c, &frontier_b);
            let (frontier_d, sigma_cd) = problem.summarize_block(c, d, &frontier_c);

            let applied_ab = sigma_ab.apply(&frontier_a);
            let applied_bc = sigma_bc.apply(&frontier_b);
            prop_assert_eq!(applied_ab.scores.as_slice(), frontier_b.scores.as_slice());
            prop_assert_eq!(applied_bc.scores.as_slice(), frontier_c.scores.as_slice());

            let merged = problem.merge_summary(&sigma_ab, &sigma_bc);
            let merged_frontier = merged.apply(&frontier_a);
            prop_assert_eq!(
                merged_frontier.scores.as_slice(),
                frontier_c.scores.as_slice()
            );

            let direct_ac = problem.summarize_block(a, c, &frontier_a).1;
            let direct_frontier = direct_ac.apply(&frontier_a);
            prop_assert_eq!(
                direct_frontier.scores.as_slice(),
                frontier_c.scores.as_slice()
            );

            let merged_left = problem.merge_summary(&merged, &sigma_cd);
            let merged_left_frontier = merged_left.apply(&frontier_a);
            let merged_right = problem.merge_summary(&sigma_ab, &problem.merge_summary(&sigma_bc, &sigma_cd));
            let merged_right_frontier = merged_right.apply(&frontier_a);
            prop_assert_eq!(
                merged_left_frontier.scores.as_slice(),
                merged_right_frontier.scores.as_slice()
            );
            prop_assert_eq!(
                merged_left_frontier.scores.as_slice(),
                frontier_d.scores.as_slice()
            );
        }
    }
}

mod nw_laws {
    use super::*;

    proptest! {
        #[test]
        fn summary_laws(s in "[ACGT]{0,6}", t in "[ACGT]{0,6}", seeds in (0usize..6, 0usize..6, 0usize..6)) {
            let s = s.as_bytes();
            let t = t.as_bytes();
            let ms = 2;
            let mm = 1;
            let gp = -1;
            let problem = NwProblem::new(s, t, ms, mm, gp);
            let total = problem.num_layers();
            let Some((a,b,c,d)) = super::split_points(total, seeds) else {
                return Ok(());
            };

            let frontier_a = frontier_at(&problem, a);
            let (frontier_b, sigma_ab) = problem.summarize_block(a, b, &frontier_a);
            let (frontier_c, sigma_bc) = problem.summarize_block(b, c, &frontier_b);
            let (frontier_d, sigma_cd) = problem.summarize_block(c, d, &frontier_c);

            let applied_ab = sigma_ab.apply(&frontier_a);
            let applied_bc = sigma_bc.apply(&frontier_b);
            prop_assert_eq!(applied_ab.scores.as_slice(), frontier_b.scores.as_slice());
            prop_assert_eq!(applied_bc.scores.as_slice(), frontier_c.scores.as_slice());

            let merged = problem.merge_summary(&sigma_ab, &sigma_bc);
            let merged_frontier = merged.apply(&frontier_a);
            prop_assert_eq!(
                merged_frontier.scores.as_slice(),
                frontier_c.scores.as_slice()
            );

            let direct_ac = problem.summarize_block(a, c, &frontier_a).1;
            let direct_frontier = direct_ac.apply(&frontier_a);
            prop_assert_eq!(
                direct_frontier.scores.as_slice(),
                frontier_c.scores.as_slice()
            );

            let merged_left = problem.merge_summary(&merged, &sigma_cd);
            let merged_left_frontier = merged_left.apply(&frontier_a);
            let merged_right = problem.merge_summary(&sigma_ab, &problem.merge_summary(&sigma_bc, &sigma_cd));
            let merged_right_frontier = merged_right.apply(&frontier_a);
            prop_assert_eq!(
                merged_left_frontier.scores.as_slice(),
                merged_right_frontier.scores.as_slice()
            );
            prop_assert_eq!(
                merged_left_frontier.scores.as_slice(),
                frontier_d.scores.as_slice()
            );
        }
    }
}

mod viterbi_laws {
    use super::*;

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

    fn approx_eq(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() <= 1e-9 || (x.is_nan() && y.is_nan()))
    }

    proptest! {
        #[test]
        fn summary_laws(obs in proptest::collection::vec(0usize..2, 3..8), seeds in (0usize..6, 0usize..6, 0usize..6)) {
            let problem = ViterbiProblem::new(demo_hmm(), obs.clone());
            let total = problem.num_layers();
            let Some((a,b,c,d)) = split_points(total, seeds) else {
                return Ok(());
            };

            let frontier_a = frontier_at(&problem, a);
            let (frontier_b, sigma_ab) = problem.summarize_block(a, b, &frontier_a);
            let (frontier_c, sigma_bc) = problem.summarize_block(b, c, &frontier_b);
            let (frontier_d, sigma_cd) = problem.summarize_block(c, d, &frontier_c);

            let applied_ab = sigma_ab.apply(&frontier_a);
            let applied_bc = sigma_bc.apply(&frontier_b);
            prop_assert!(approx_eq(
                applied_ab.log_delta.as_slice(),
                frontier_b.log_delta.as_slice()
            ));
            prop_assert!(approx_eq(
                applied_bc.log_delta.as_slice(),
                frontier_c.log_delta.as_slice()
            ));

            let merged = problem.merge_summary(&sigma_ab, &sigma_bc);
            let merged_frontier = merged.apply(&frontier_a);
            prop_assert!(approx_eq(
                merged_frontier.log_delta.as_slice(),
                frontier_c.log_delta.as_slice()
            ));

            let direct_ac = problem.summarize_block(a, c, &frontier_a).1;
            let direct_frontier = direct_ac.apply(&frontier_a);
            prop_assert!(approx_eq(
                direct_frontier.log_delta.as_slice(),
                frontier_c.log_delta.as_slice()
            ));

            let merged_left = problem.merge_summary(&merged, &sigma_cd);
            let merged_left_frontier = merged_left.apply(&frontier_a);
            let merged_right = problem.merge_summary(&sigma_ab, &problem.merge_summary(&sigma_bc, &sigma_cd));
            let merged_right_frontier = merged_right.apply(&frontier_a);
            prop_assert!(approx_eq(
                merged_left_frontier.log_delta.as_slice(),
                merged_right_frontier.log_delta.as_slice()
            ));
            prop_assert!(approx_eq(
                merged_left_frontier.log_delta.as_slice(),
                frontier_d.log_delta.as_slice()
            ));
        }
    }
}

mod dag_laws {
    use super::*;

    fn make_dag(seed: (usize, usize)) -> DagLayered {
        let widths = vec![2, 2, 2, 2];
        let mut adjacency = Vec::new();
        for (layer, &width) in widths.iter().take(widths.len() - 1).enumerate() {
            let mut layer_edges = Vec::new();
            for u in 0..width {
                let w1 = 1 + ((seed.0 + layer + u) % 5) as i64;
                let w2 = 1 + ((seed.1 + layer + u * 7) % 7) as i64;
                layer_edges.push(vec![(0, w1), (1, w2)]);
            }
            adjacency.push(layer_edges);
        }
        DagLayered::new(adjacency, widths)
    }

    proptest! {
        #[test]
        fn summary_laws(seeds in (0usize..10, 0usize..10), splits in (0usize..3, 0usize..3, 0usize..3)) {
            let problem = make_dag(seeds);
            let total = problem.num_layers();
            let Some((a,b,c,d)) = super::split_points(total, splits) else {
                return Ok(());
            };

            let frontier_a = frontier_at(&problem, a);
            let (frontier_b, sigma_ab) = problem.summarize_block(a, b, &frontier_a);
            let (frontier_c, sigma_bc) = problem.summarize_block(b, c, &frontier_b);
            let (frontier_d, sigma_cd) = problem.summarize_block(c, d, &frontier_c);

            let applied_ab = sigma_ab.apply(&frontier_a);
            let applied_bc = sigma_bc.apply(&frontier_b);
            prop_assert_eq!(applied_ab.dist.as_slice(), frontier_b.dist.as_slice());
            prop_assert_eq!(applied_bc.dist.as_slice(), frontier_c.dist.as_slice());

            let merged = problem.merge_summary(&sigma_ab, &sigma_bc);
            let merged_frontier = merged.apply(&frontier_a);
            prop_assert_eq!(
                merged_frontier.dist.as_slice(),
                frontier_c.dist.as_slice()
            );

            let direct_ac = problem.summarize_block(a, c, &frontier_a).1;
            let direct_frontier = direct_ac.apply(&frontier_a);
            prop_assert_eq!(
                direct_frontier.dist.as_slice(),
                frontier_c.dist.as_slice()
            );

            let merged_left = problem.merge_summary(&merged, &sigma_cd);
            let merged_left_frontier = merged_left.apply(&frontier_a);
            let merged_right = problem.merge_summary(&sigma_ab, &problem.merge_summary(&sigma_bc, &sigma_cd));
            let merged_right_frontier = merged_right.apply(&frontier_a);
            prop_assert_eq!(
                merged_left_frontier.dist.as_slice(),
                merged_right_frontier.dist.as_slice()
            );
            prop_assert_eq!(
                merged_left_frontier.dist.as_slice(),
                frontier_d.dist.as_slice()
            );
        }
    }
}
