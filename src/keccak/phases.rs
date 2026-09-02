//! A [`tracing::Subscriber`] that keeps the wall clock of binius64's own phase spans.
//!
//! binius64 instruments `Prover::prove` and `Verifier::verify` with `INFO` spans — `Commit
//! witness`, `[phase] BitAnd check`, `[phase] Shift Reduction`, `[phase] PCS Opening` and their
//! verifier counterparts — so the stock pipeline is measured at binius64's own granularity rather
//! than split by hand. Everything below `INFO` is disabled, which costs those spans nothing.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Level, Metadata, Subscriber};

#[derive(Clone, Default)]
pub struct Phases(Arc<Mutex<Spans>>);

#[derive(Default)]
pub struct Spans {
    next: u64,
    name: HashMap<u64, &'static str>,
    entered: HashMap<u64, Instant>,
    total: HashMap<&'static str, Duration>,
}

impl Phases {
    /// Runs `f` with this subscriber installed for the current thread.
    pub fn record<T>(&self, f: impl FnOnce() -> T) -> T {
        tracing::subscriber::with_default(self.clone(), f)
    }

    /// Milliseconds spent in the span of that name, zero if it never ran.
    pub fn milliseconds(&self, name: &str) -> f64 {
        self.0
            .lock()
            .unwrap()
            .total
            .get(name)
            .map_or(0.0, |d| d.as_secs_f64() * 1e3)
    }
}

impl Subscriber for Phases {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        *metadata.level() <= Level::INFO
    }

    fn new_span(&self, span: &Attributes<'_>) -> Id {
        let mut spans = self.0.lock().unwrap();
        spans.next += 1;
        let id = spans.next;
        spans.name.insert(id, span.metadata().name());
        Id::from_u64(id)
    }

    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, _event: &Event<'_>) {}

    fn enter(&self, span: &Id) {
        self.0
            .lock()
            .unwrap()
            .entered
            .insert(span.into_u64(), Instant::now());
    }

    fn exit(&self, span: &Id) {
        let mut spans = self.0.lock().unwrap();
        let Some(start) = spans.entered.remove(&span.into_u64()) else {
            return;
        };
        let name = spans.name[&span.into_u64()];
        *spans.total.entry(name).or_default() += start.elapsed();
    }
}
