use hcp_dp::{problems::lcs::LcsProblem, HcpEngine};

#[test]
fn zero_layers_no_op() {
    struct EmptyProblem;
    #[derive(Clone)]
    struct UnitSummary;
    #[allow(clippy::unused_unit)]
    impl hcp_dp::traits::SummaryApply<()> for UnitSummary {
        fn apply(&self, frontier: &()) -> () {
            *frontier
        }
    }
    impl hcp_dp::HcpProblem for EmptyProblem {
        type State = usize;
        type Frontier = ();
        type Summary = UnitSummary;
        type Boundary = usize;
        type Cost = i32;
        fn num_layers(&self) -> usize {
            0
        }
        fn init_frontier(&self) -> Self::Frontier {}
        fn forward_step(&self, _layer: usize, _frontier_i: &Self::Frontier) -> Self::Frontier {}
        fn summarize_block(
            &self,
            _a: usize,
            _b: usize,
            frontier_a: &Self::Frontier,
        ) -> (Self::Frontier, Self::Summary) {
            (*frontier_a, UnitSummary)
        }
        fn merge_summary(&self, _left: &Self::Summary, _right: &Self::Summary) -> Self::Summary {
            UnitSummary
        }
        fn initial_boundary(&self) -> Self::Boundary {
            0
        }
        fn terminal_boundary(&self, _frontier_t: &Self::Frontier) -> Self::Boundary {
            0
        }
        fn choose_boundary(
            &self,
            _a: usize,
            _m: usize,
            _c: usize,
            _sigma_left: &Self::Summary,
            _sigma_right: &Self::Summary,
            beta_a: &Self::Boundary,
            _beta_c: &Self::Boundary,
        ) -> Self::Boundary {
            *beta_a
        }
        fn reconstruct_block(
            &self,
            _a: usize,
            _b: usize,
            beta_a: &Self::Boundary,
            _beta_b: &Self::Boundary,
        ) -> Vec<Self::State> {
            vec![*beta_a]
        }
        fn extract_cost(
            &self,
            _frontier_t: &Self::Frontier,
            _beta_t: &Self::Boundary,
        ) -> Self::Cost {
            0
        }
    }
    let engine = HcpEngine::new(EmptyProblem);
    let (cost, path) = engine.run();
    assert_eq!(cost, 0);
    assert!(path.is_empty() || path == vec![0]);
}

#[test]
fn concatenation_drops_single_midpoint() {
    let s = b"ACCG";
    let t = b"ACCG";
    let engine = HcpEngine::with_block_size(LcsProblem::new(s, t), 2);
    let (_cost, path) = engine.run();
    // Path must be monotone and start at (0,0), end at (4,4)
    assert_eq!(path.first(), Some(&(0, 0)));
    assert_eq!(path.last(), Some(&(4, 4)));
    for window in path.windows(2) {
        let (prev, next) = (window[0], window[1]);
        assert!(next.0 >= prev.0 && next.1 >= prev.1);
        let di = next.0 - prev.0;
        let dj = next.1 - prev.1;
        assert!(di <= 1 && dj <= 1, "violates monotone step");
    }
}
