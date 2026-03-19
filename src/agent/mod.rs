//! Agent module for stateful conversation handling.

#[allow(clippy::module_inception)]
mod agent;
mod state;
mod types;

pub use agent::{
    agent_loop, agent_loop_continue, run_agent_loop, run_agent_loop_continue, Agent, AgentError,
    AgentEventStream, SubscriberId,
};
pub use state::{AgentState, AgentStateSnapshot};
pub use types::*;
