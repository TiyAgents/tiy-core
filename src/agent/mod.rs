//! Agent module for stateful conversation handling.

mod agent;
mod state;
mod types;

pub use agent::{Agent, AgentError};
pub use state::{AgentState, AgentStateSnapshot};
pub use types::*;
