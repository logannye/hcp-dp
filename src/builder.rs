use crate::utils::default_block_size;
use crate::{HcpEngine, HcpProblem};

pub struct HcpEngineBuilder<P: HcpProblem> {
    problem: P,
    block_size: Option<usize>,
}

impl<P: HcpProblem> HcpEngineBuilder<P> {
    pub fn new(problem: P) -> Self {
        Self {
            problem,
            block_size: None,
        }
    }
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = Some(block_size);
        self
    }
    pub fn build(self) -> HcpEngine<P> {
        match self.block_size {
            Some(b) => HcpEngine::with_block_size(self.problem, b),
            None => {
                let t = self.problem.num_layers().max(1);
                let b = default_block_size(t).max(1);
                HcpEngine::with_block_size(self.problem, b)
            }
        }
    }
}
