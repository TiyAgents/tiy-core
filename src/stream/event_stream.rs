//! Simplified event stream implementation using async-safe primitives.

use futures::Stream;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

/// Shared inner state for the event stream.
struct EventStreamInner<T, R> {
    /// Event queue.
    events: Mutex<VecDeque<T>>,
    /// Whether the stream is done.
    done: AtomicBool,
    /// The final result.
    result: Mutex<Option<R>>,
    /// Waker to notify when new events are available.
    waker: Mutex<Option<Waker>>,
}

/// A generic event stream that supports async iteration and final result retrieval.
pub struct EventStream<T, R = T> {
    inner: Arc<EventStreamInner<T, R>>,
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
            inner: Arc::new(EventStreamInner {
                events: Mutex::new(VecDeque::new()),
                done: AtomicBool::new(false),
                result: Mutex::new(None),
                waker: Mutex::new(None),
            }),
            is_complete,
            extract_result,
        }
    }

    /// Wake the stream consumer.
    fn wake(&self) {
        if let Some(waker) = self.inner.waker.lock().take() {
            waker.wake();
        }
    }

    /// Push an event to the stream.
    pub fn push(&self, event: T) {
        if self.inner.done.load(Ordering::SeqCst) {
            // Stream is already done, ignore further events
            return;
        }

        let is_complete = (self.is_complete)(&event);
        if is_complete {
            // Extract the result and store it for result() callers
            let result = (self.extract_result)(event);
            *self.inner.result.lock() = Some(result);
            self.inner.done.store(true, Ordering::SeqCst);
        } else {
            self.inner.events.lock().push_back(event);
        }
        self.wake();
    }

    /// End the stream with an optional result.
    pub fn end(&self, result: Option<R>) {
        if result.is_some() {
            *self.inner.result.lock() = result;
        }
        self.inner.done.store(true, Ordering::SeqCst);
        self.wake();
    }

    /// Check if the stream has ended.
    pub fn is_done(&self) -> bool {
        self.inner.done.load(Ordering::SeqCst)
    }

    /// Get the final result (async, non-blocking wait).
    pub async fn result(&self) -> R {
        loop {
            {
                let mut result = self.inner.result.lock();
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

        // Try to get an event from the queue
        {
            let mut queue = this.inner.events.lock();
            if let Some(event) = queue.pop_front() {
                return Poll::Ready(Some(event));
            }
        }

        // Queue is empty — check if we're done
        if this.inner.done.load(Ordering::SeqCst) {
            return Poll::Ready(None);
        }

        // Not done, no events: register waker and return Pending
        *this.inner.waker.lock() = Some(cx.waker().clone());

        // Double-check after registering waker to avoid race condition
        {
            let mut queue = this.inner.events.lock();
            if let Some(event) = queue.pop_front() {
                return Poll::Ready(Some(event));
            }
        }
        if this.inner.done.load(Ordering::SeqCst) {
            return Poll::Ready(None);
        }

        Poll::Pending
    }
}

impl<T, R> Clone for EventStream<T, R> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
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
