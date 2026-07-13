//! Bounded, deterministic batch execution and recoverable-record contracts.

use crate::oracle::{
    AlignmentRecord, ComparisonReport, Producer, VerificationReport, SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;

/// Stage at which one batch record failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorStage {
    InputParse,
    RecordParse,
    Resolve,
    Solve,
    Verify,
    Compare,
    Serialize,
    Worker,
}

/// Best-effort location of a failed batch item.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ErrorContext {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pair_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_line: Option<usize>,
}

/// Machine-readable failure details.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ErrorDetail {
    pub stage: ErrorStage,
    pub code: String,
    pub message: String,
    pub recoverable: bool,
}

/// A recoverable `hcp-align.v2` batch failure.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ErrorRecord {
    pub schema_version: String,
    pub record_type: String,
    pub producer: Box<Producer>,
    pub context: Box<ErrorContext>,
    pub error: Box<ErrorDetail>,
}

impl ErrorRecord {
    pub fn new(
        stage: ErrorStage,
        code: impl Into<String>,
        message: impl Into<String>,
        context: ErrorContext,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            record_type: "error".to_string(),
            producer: Box::new(Producer {
                name: "hcp-align".to_string(),
                version: crate::VERSION.to_string(),
                commit: option_env!("HCP_DP_GIT_COMMIT").map(str::to_string),
                backend: "batch".to_string(),
            }),
            context: Box::new(context),
            error: Box::new(ErrorDetail {
                stage,
                code: code.into(),
                message: message.into(),
                recoverable: true,
            }),
        }
    }
}

/// Any stable machine record emitted by the oracle product.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OracleRecord {
    Alignment(Box<AlignmentRecord>),
    Verification(Box<VerificationReport>),
    Comparison(Box<ComparisonReport>),
    Error(Box<ErrorRecord>),
}

/// Common resource limits for batch commands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchOptions {
    pub threads: usize,
    pub max_in_flight: usize,
    pub max_in_flight_bases: usize,
    pub fail_fast: bool,
}

impl Default for BatchOptions {
    fn default() -> Self {
        let threads = thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(8);
        Self {
            threads,
            max_in_flight: threads.saturating_mul(2).max(1),
            max_in_flight_bases: 64_000_000,
            fail_fast: false,
        }
    }
}

impl BatchOptions {
    pub fn validate(&self) -> Result<(), String> {
        if self.threads == 0 {
            return Err("--threads must be greater than zero".to_string());
        }
        if self.max_in_flight == 0 {
            return Err("--max-in-flight must be greater than zero".to_string());
        }
        if self.max_in_flight_bases == 0 {
            return Err("--max-in-flight-bases must be greater than zero".to_string());
        }
        Ok(())
    }
}

/// Observable batch counts, including high-water marks used by acceptance tests.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchSummary {
    pub records: u64,
    pub error_records: u64,
    pub max_pending_jobs: usize,
    pub max_pending_bases: usize,
}

/// One ordered result from [`run_ordered`].
#[derive(Clone, Debug)]
pub(crate) enum BatchItem<T> {
    Record(T),
    Error(ErrorRecord),
}

/// Execute jobs concurrently while strictly bounding submitted-but-not-emitted work.
///
/// The coordinator refills only after it emits the next consecutive result. A
/// slow first job therefore cannot grow the reorder buffer beyond
/// `max_in_flight`. Jobs larger than the base budget are admitted only when no
/// other job is pending.
pub(crate) fn run_ordered<I, J, T, Weight, Work, Emit>(
    input: I,
    options: &BatchOptions,
    weight: Weight,
    work: Work,
    mut emit: Emit,
) -> Result<BatchSummary, String>
where
    I: IntoIterator<Item = Result<J, ErrorRecord>>,
    J: Send + 'static,
    T: Send + 'static,
    Weight: Fn(&J) -> usize,
    Work: Fn(J) -> Result<T, ErrorRecord> + Send + Sync + 'static,
    Emit: FnMut(BatchItem<T>) -> Result<(), String>,
{
    options.validate()?;
    struct JobQueue<J> {
        jobs: VecDeque<(usize, J)>,
        closed: bool,
    }
    let mut iterator = input.into_iter();
    let queue = Arc::new((
        Mutex::new(JobQueue {
            jobs: VecDeque::new(),
            closed: false,
        }),
        Condvar::new(),
    ));
    let (result_tx, result_rx) = mpsc::channel::<(usize, BatchItem<T>)>();
    let work = Arc::new(work);
    let mut handles = Vec::with_capacity(options.threads);
    for _ in 0..options.threads {
        let queue = Arc::clone(&queue);
        let result_tx = result_tx.clone();
        let work = Arc::clone(&work);
        handles.push(thread::spawn(move || loop {
            let job = {
                let (lock, available) = &*queue;
                let mut state = lock.lock().expect("batch queue mutex poisoned");
                while state.jobs.is_empty() && !state.closed {
                    state = available
                        .wait(state)
                        .expect("batch queue mutex poisoned while waiting");
                }
                match state.jobs.pop_front() {
                    Some(job) => job,
                    None if state.closed => return,
                    None => continue,
                }
            };
            let (index, job) = job;
            let item = match catch_unwind(AssertUnwindSafe(|| work(job))) {
                Ok(Ok(record)) => BatchItem::Record(record),
                Ok(Err(error)) => BatchItem::Error(error),
                Err(_) => BatchItem::Error(ErrorRecord::new(
                    ErrorStage::Worker,
                    "worker_panic",
                    "batch worker panicked",
                    ErrorContext {
                        pair_index: Some(index),
                        ..ErrorContext::default()
                    },
                )),
            };
            if result_tx.send((index, item)).is_err() {
                return;
            }
        }));
    }
    drop(result_tx);

    let coordinator = (|| {
        let mut summary = BatchSummary::default();
        let mut next_index = 0usize;
        let mut deferred: Option<Result<J, ErrorRecord>> = None;

        // Execute bounded windows on a persistent worker pool. Draining each
        // window before admitting the next one deliberately trades a little
        // pipeline depth for a hard bound on both queued work and the reorder
        // buffer, including when the first job is arbitrarily slow.
        loop {
            let mut buffered = BTreeMap::<usize, BatchItem<T>>::new();
            let mut window_items = 0usize;
            let mut worker_jobs = 0usize;
            let mut window_bases = 0usize;

            while window_items < options.max_in_flight {
                let next = deferred.take().or_else(|| iterator.next());
                let Some(next) = next else {
                    break;
                };
                match next {
                    Ok(job) => {
                        let job_weight = weight(&job);
                        if window_items > 0
                            && window_bases.saturating_add(job_weight) > options.max_in_flight_bases
                        {
                            deferred = Some(Ok(job));
                            break;
                        }
                        window_bases = window_bases.saturating_add(job_weight);
                        let (lock, available) = &*queue;
                        lock.lock()
                            .map_err(|_| "batch queue mutex was poisoned".to_string())?
                            .jobs
                            .push_back((next_index, job));
                        available.notify_one();
                        worker_jobs += 1;
                    }
                    Err(error) => {
                        buffered.insert(next_index, BatchItem::Error(error));
                    }
                }
                next_index += 1;
                window_items += 1;
            }

            if window_items == 0 {
                break;
            }
            summary.max_pending_jobs = summary.max_pending_jobs.max(window_items);
            summary.max_pending_bases = summary.max_pending_bases.max(window_bases);

            for _ in 0..worker_jobs {
                let (index, item) = result_rx
                    .recv()
                    .map_err(|_| "batch workers stopped before returning every job".to_string())?;
                buffered.insert(index, item);
            }
            if buffered.len() != window_items {
                return Err("batch window completed with missing results".to_string());
            }

            for (_, item) in buffered {
                let is_error = matches!(item, BatchItem::Error(_));
                let is_fatal = matches!(&item, BatchItem::Error(error) if !error.error.recoverable);
                emit(item)?;
                summary.records += 1;
                summary.error_records += u64::from(is_error);
                if is_fatal || (is_error && options.fail_fast) {
                    return Ok(summary);
                }
            }
        }
        Ok(summary)
    })();

    {
        let (lock, available) = &*queue;
        if let Ok(mut state) = lock.lock() {
            if coordinator.is_err() || options.fail_fast {
                state.jobs.clear();
            }
            state.closed = true;
        }
        available.notify_all();
    }
    for handle in handles {
        let _ = handle.join();
    }
    coordinator
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn error_records_use_the_oracle_version() {
        let error = ErrorRecord::new(
            ErrorStage::RecordParse,
            "invalid_record",
            "fixture",
            ErrorContext::default(),
        );
        assert_eq!(error.producer.name, "hcp-align");
        assert_eq!(error.producer.version, crate::VERSION);
    }

    #[test]
    fn slow_first_job_is_ordered_and_window_bounded() {
        let options = BatchOptions {
            threads: 4,
            max_in_flight: 8,
            max_in_flight_bases: 32,
            fail_fast: false,
        };
        let mut output = Vec::new();
        let summary = run_ordered(
            (0usize..100).map(Ok),
            &options,
            |_| 4,
            |value| {
                if value == 0 {
                    thread::sleep(Duration::from_millis(20));
                }
                Ok(value)
            },
            |item| {
                if let BatchItem::Record(value) = item {
                    output.push(value);
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(output, (0..100).collect::<Vec<_>>());
        assert!(summary.max_pending_jobs <= 8);
        assert!(summary.max_pending_bases <= 32);
    }

    #[test]
    fn worker_panic_becomes_an_ordered_error_record() {
        let options = BatchOptions {
            threads: 2,
            max_in_flight: 2,
            max_in_flight_bases: 10,
            fail_fast: false,
        };
        let mut kinds = Vec::new();
        let summary = run_ordered(
            [Ok(0usize), Ok(1), Ok(2)],
            &options,
            |_| 1,
            |value| {
                if value == 1 {
                    panic!("injected worker panic");
                }
                Ok(value)
            },
            |item| {
                kinds.push(match item {
                    BatchItem::Record(value) => format!("record:{value}"),
                    BatchItem::Error(error) => error.error.code,
                });
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(kinds, ["record:0", "worker_panic", "record:2"]);
        assert_eq!(summary.error_records, 1);
    }

    #[test]
    fn oversized_pair_runs_in_its_own_window() {
        let options = BatchOptions {
            threads: 2,
            max_in_flight: 4,
            max_in_flight_bases: 10,
            fail_fast: false,
        };
        let mut output = Vec::new();
        let summary = run_ordered(
            [Ok(100usize), Ok(1), Ok(1)],
            &options,
            |value| *value,
            Ok,
            |item| {
                if let BatchItem::Record(value) = item {
                    output.push(value);
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(output, [100, 1, 1]);
        assert_eq!(summary.max_pending_bases, 100);
        assert_eq!(summary.max_pending_jobs, 2);
    }

    #[test]
    fn unrecoverable_error_stops_jsonl_style_continuation() {
        let options = BatchOptions {
            threads: 1,
            max_in_flight: 3,
            max_in_flight_bases: 10,
            fail_fast: false,
        };
        let mut fatal = ErrorRecord::new(
            ErrorStage::InputParse,
            "fatal_input",
            "record boundaries are unavailable",
            ErrorContext::default(),
        );
        fatal.error.recoverable = false;
        let mut emitted = Vec::new();
        let summary = run_ordered(
            [Ok(0usize), Err(fatal), Ok(2)],
            &options,
            |_| 1,
            Ok,
            |item| {
                emitted.push(matches!(item, BatchItem::Error(_)));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(emitted, [false, true]);
        assert_eq!(summary.records, 2);
        assert_eq!(summary.error_records, 1);
    }
}
