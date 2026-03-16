//! Simplified event stream implementation.

use futures::Stream;
use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

/// A generic event stream that supports async iteration and final result retrieval.
pub struct EventStream<T, R = T> {
    queue: Arc<Mutex<VecDeque<T>>>,
    done: Arc<AtomicBool>,
    condvar: Arc<Condvar>,
    result: Arc<Mutex<Option<R>>>,
    is_complete: fn(&T) -> bool,
    extract_result: fn(T) -> R,
}

impl<T, R> EventStream<T, R>
where
    T: Send + 'static,
    R: Send + 'static,
{
    /// Create a new event stream.
    pub fn new(is_complete: fn(&T) -> bool, extract_result: fn(T) -> R) -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            done: Arc::new(AtomicBool::new(false)),
            condvar: Arc::new(Condvar::new()),
            result: Arc::new(Mutex::new(None)),
            is_complete,
            extract_result,
        }
    }

    /// Push an event to the stream.
    pub fn push(&self, event: T) {
        let mut queue = self.queue.lock();

        // Check if this is a completion event
        let is_complete = (self.is_complete)(&event);
        if is_complete {
            self.done.store(true, Ordering::SeqCst);
            let result = (self.extract_result)(event);
            *self.result.lock() = Some(result);
            self.condvar.notify_all();
            return;
        }

        queue.push_back(event);
        self.condvar.notify_one();
    }

    /// End the stream with an optional result.
    pub fn end(&self, result: Option<R>) {
        self.done.store(true, Ordering::SeqCst);
        if let Some(r) = result {
            *self.result.lock() = Some(r);
        }
        self.condvar.notify_all();
    }

    /// Check if the stream has ended.
    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::SeqCst)
    }

    /// Get the final result (blocks until ready).
    pub async fn result(&self) -> R {
        // Wait for result
        loop {
            {
                let mut result = self.result.lock();
                if let Some(r) = result.take() {
                    return r;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }
}

impl<T, R> Stream for EventStream<T, R>
where
    T: Send + Unpin,
{
    type Item = T;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let mut queue = this.queue.lock();

        if let Some(event) = queue.pop_front() {
            return Poll::Ready(Some(event));
        }

        if this.done.load(Ordering::SeqCst) {
            return Poll::Ready(None);
        }

        // Wait for more events
        let _ = this.condvar.wait(&mut queue);
        if let Some(event) = queue.pop_front() {
            Poll::Ready(Some(event))
        } else if this.done.load(Ordering::SeqCst) {
            Poll::Ready(None)
        } else {
            Poll::Pending
        }
    }
}

impl<T, R> Clone for EventStream<T, R> {
    fn clone(&self) -> Self {
        Self {
            queue: Arc::clone(&self.queue),
            done: Arc::clone(&self.done),
            condvar: Arc::clone(&self.condvar),
            result: Arc::clone(&self.result),
            is_complete: self.is_complete,
            extract_result: self.extract_result,
        }
    }
}

/// Assistant message event stream type alias.
pub type AssistantMessageEventStream = EventStream<crate::types::AssistantMessageEvent, crate::types::AssistantMessage>;

impl AssistantMessageEventStream {
    /// Create a new assistant message event stream.
    pub fn new_assistant_stream() -> Self {
        Self::new(
            |event| event.is_complete(),
            |event| {
                match event {
                    crate::types::AssistantMessageEvent::Done { message, .. } => message.clone(),
                    crate::types::AssistantMessageEvent::Error { error, .. } => error.clone(),
                    _ => unreachable!("is_complete should only return true for Done/Error"),
                }
            },
        )
    }
}
