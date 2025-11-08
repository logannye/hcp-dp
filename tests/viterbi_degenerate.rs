use hcp_dp::problems::viterbi::{Hmm, ViterbiProblem};
use hcp_dp::HcpEngine;

fn make_sym_hmm() -> Hmm {
    Hmm {
        n_states: 2,
        log_pi: vec![(0.5f64).ln(), (0.5f64).ln()],
        log_a: vec![
            vec![(0.5f64).ln(), (0.5f64).ln()],
            vec![(0.5f64).ln(), (0.5f64).ln()],
        ],
        log_b: vec![
            vec![(0.6f64).ln(), (0.4f64).ln()],
            vec![(0.4f64).ln(), (0.6f64).ln()],
        ],
    }
}

#[test]
fn zero_probabilities_are_respected() {
    let hmm = Hmm {
        n_states: 2,
        log_pi: vec![0.0, f64::NEG_INFINITY], // only state 0 possible initially
        log_a: vec![
            vec![0.0, f64::NEG_INFINITY], // state 1 unreachable
            vec![f64::NEG_INFINITY, 0.0],
        ],
        log_b: vec![
            vec![0.0, f64::NEG_INFINITY], // state 0 can emit only symbol 0
            vec![0.0, 0.0],
        ],
    };
    let obs = vec![0, 0, 0];
    let (log_prob, path) = HcpEngine::new(ViterbiProblem::new(hmm, obs.clone())).run();
    assert!(log_prob.0.is_finite());
    assert_eq!(path.len(), obs.len());
    assert!(path.iter().all(|s| s.state == 0));
}

#[test]
fn ties_are_deterministic() {
    let hmm = make_sym_hmm();
    let obs = vec![0, 1, 0, 1];
    let engine = HcpEngine::new(ViterbiProblem::new(hmm.clone(), obs.clone()));
    let (log_p1, path1) = engine.run();
    let (log_p2, path2) = HcpEngine::new(ViterbiProblem::new(hmm, obs)).run();
    assert_eq!(log_p1, log_p2);
    assert_eq!(path1, path2);
}

#[test]
fn short_sequences_with_ties() {
    let hmm = make_sym_hmm();
    let obs = vec![1];
    let (log_prob, path) = HcpEngine::new(ViterbiProblem::new(hmm, obs.clone())).run();
    assert_eq!(path.len(), obs.len());
    assert!(log_prob.0.is_finite());
}
