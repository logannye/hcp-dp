use hcp_dp::{
    problems::viterbi::{Hmm, ViterbiProblem},
    HcpEngine,
};

#[test]
fn viterbi_example_integration() {
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
    let _ = cost.0;
}
