use hcp_dp::{problems::matrix_chain::MatrixChainProblem, HcpEngine};

#[test]
fn matrix_chain_clrs_integration() {
    let p = vec![30, 35, 15, 5, 10, 20, 25];
    let problem = MatrixChainProblem::new(p);
    let engine = HcpEngine::new(problem);
    let (cost, states) = engine.run();
    assert_eq!(cost, 15125);
    // States list split decisions; it should be non-empty for n>2
    assert!(!states.is_empty());
}
