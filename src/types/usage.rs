//! Token usage and cost tracking.
//!
//! ## Field semantics across protocols
//!
//! Providers report cache in two distinct shapes. All protocol modules in
//! `src/protocol/` normalize the wire values into a single internal model so
//! downstream code can use a uniform formula:
//!
//! | Field        | OpenAI / Google                                       | Anthropic                              |
//! | ------------ | ----------------------------------------------------- | -------------------------------------- |
//! | `input`      | prompt tokens **excluding** the cached slice          | non-cached input (mutually exclusive)  |
//! | `cache_read` | cached slice of the prompt (subset of original total) | cache hit tokens (mutually exclusive)  |
//! | `cache_write`| always 0 (no write-side API on these providers)      | tokens written to cache this request   |
//! | `output`     | generated tokens                                      | generated tokens                       |
//!
//! After normalization:
//! - OpenAI / Google: `input + cache_read` equals the original
//!   `prompt_tokens` / `prompt_token_count` from the wire response.
//! - Anthropic: `input + cache_read + cache_write` is the three non-overlapping
//!   components of billable input tokens.
//!
//! In **every** case the true context footprint of one request is
//! `input + output + cache_read + cache_write` — see
//! [`Usage::context_size`].

use serde::{Deserialize, Serialize};

/// Token usage information.
///
/// Field semantics are documented at the module level — see the table in
/// `src/types/usage.rs` for the per-protocol meaning of each counter.
///
/// The recommended way to get the true context footprint of a request across
/// all providers is [`Usage::context_size`], which always returns
/// `input + output + cache_read + cache_write` regardless of the original
/// protocol's accounting convention.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Usage {
    /// Number of non-cached input tokens.
    ///
    /// - OpenAI / Google: `prompt_tokens - cached_tokens` (cached slice
    ///   has been split out into [`Self::cache_read`]).
    /// - Anthropic: raw `input_tokens` (cache hits / writes are reported
    ///   separately and do **not** overlap with this counter).
    pub input: u64,
    /// Number of output tokens (model-generated).
    pub output: u64,
    /// Number of cached tokens read.
    ///
    /// - OpenAI / Google: subset of the original prompt token total — when
    ///   added back to [`Self::input`] it reproduces the wire `prompt_tokens`.
    /// - Anthropic: mutually exclusive with [`Self::input`] and
    ///   [`Self::cache_write`].
    pub cache_read: u64,
    /// Number of tokens written to cache. Only Anthropic populates this
    /// counter; it is always `0` for OpenAI / Google.
    pub cache_write: u64,
    /// Total tokens used.
    ///
    /// This field is the **wire-level** total (when the provider exposes one)
    /// or a synthesized sum for protocols that don't emit one. Its exact
    /// meaning depends on the source protocol:
    ///
    /// - OpenAI Chat Completions / Responses: `prompt_tokens + completion_tokens`.
    ///   `cache_read` is *implicit* in `prompt_tokens` and is **not** added
    ///   again here.
    /// - Google Generative AI: `totalTokenCount` from the API.
    /// - Anthropic: `input + output + cache_read + cache_write`
    ///   (synthesized client-side).
    ///
    /// For a protocol-agnostic context footprint, use
    /// [`Usage::context_size`] instead. Do not sum `total_tokens` across
    /// different providers.
    pub total_tokens: u64,
    /// Cost breakdown.
    pub cost: UsageCost,
}

impl Usage {
    /// Create a new usage with zero values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a usage from input and output tokens.
    pub fn from_tokens(input: u64, output: u64) -> Self {
        Self {
            input,
            output,
            cache_read: 0,
            cache_write: 0,
            total_tokens: input + output,
            cost: UsageCost::default(),
        }
    }

    /// Protocol-agnostic context footprint of the request.
    ///
    /// Returns `input + output + cache_read + cache_write`. After the
    /// normalization performed in each protocol module, this is the true
    /// "how many tokens does this exchange touch" number for **every**
    /// provider — including the cached portion for OpenAI / Google and the
    /// full three-segment billable input for Anthropic.
    pub fn context_size(&self) -> u64 {
        self.input + self.output + self.cache_read + self.cache_write
    }

    /// Add another usage to this one.
    ///
    /// The four token counters are summed component-wise. The cached
    /// `total_tokens` field is **left untouched** because it carries the
    /// wire-level semantics of whichever request populated it last and
    /// is not safe to recompute across protocols (or even within the same
    /// provider, where `total_tokens` is `prompt + completion` and would
    /// double-count the cached slice if summed as
    /// `input + output + cache_read + cache_write`). Use
    /// [`Usage::context_size`] for the protocol-agnostic footprint.
    pub fn add(&mut self, other: &Usage) {
        self.input += other.input;
        self.output += other.output;
        self.cache_read += other.cache_read;
        self.cache_write += other.cache_write;
        self.cost.input += other.cost.input;
        self.cost.output += other.cost.output;
        self.cost.cache_read += other.cost.cache_read;
        self.cost.cache_write += other.cost.cache_write;
        self.cost.recalculate_total();
    }

    /// Calculate the total cost.
    pub fn total_cost(&self) -> f64 {
        self.cost.total
    }
}

/// Cost breakdown for usage.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct UsageCost {
    /// Cost for input tokens.
    pub input: f64,
    /// Cost for output tokens.
    pub output: f64,
    /// Cost for cached tokens read.
    pub cache_read: f64,
    /// Cost for tokens written to cache.
    pub cache_write: f64,
    /// Total cost.
    pub total: f64,
}

impl UsageCost {
    /// Create a new cost with zero values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a cost from input and output costs.
    pub fn from_costs(input: f64, output: f64) -> Self {
        let total = input + output;
        Self {
            input,
            output,
            cache_read: 0.0,
            cache_write: 0.0,
            total,
        }
    }

    /// Calculate the total cost from component costs.
    pub fn total(&self) -> f64 {
        self.input + self.output + self.cache_read + self.cache_write
    }

    /// Recalculate and update the stored total from component costs.
    pub fn recalculate_total(&mut self) {
        self.total = self.input + self.output + self.cache_read + self.cache_write;
    }
}
