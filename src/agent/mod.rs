//! Agent module for stateful conversation handling.

#[allow(clippy::module_inception)]
mod agent;
mod state;
mod types;

pub use agent::{Agent, AgentError, SubscriberId};
pub use state::{AgentState, AgentStateSnapshot};
pub use types::*;
