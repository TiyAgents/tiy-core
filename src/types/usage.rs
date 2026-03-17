//! Token usage and cost tracking.

use serde::{Deserialize, Serialize};

/// Token usage information.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Number of input tokens.
    pub input: u64,
    /// Number of output tokens.
    pub output: u64,
    /// Number of cached tokens read.
    pub cache_read: u64,
    /// Number of tokens written to cache.
    pub cache_write: u64,
    /// Total tokens used.
    pub total_tokens: u64,
    /// Cost breakdown.
    pub cost: UsageCost,
}

impl Default for Usage {
    fn default() -> Self {
        Self {
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            total_tokens: 0,
            cost: UsageCost::default(),
        }
    }
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

    /// Add another usage to this one.
    pub fn add(&mut self, other: &Usage) {
        self.input += other.input;
        self.output += other.output;
        self.cache_read += other.cache_read;
        self.cache_write += other.cache_write;
        self.total_tokens = self.input + self.output + self.cache_read + self.cache_write;
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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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

impl Default for UsageCost {
    fn default() -> Self {
        Self {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        }
    }
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
