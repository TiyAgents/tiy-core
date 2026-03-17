//! Agent example for tiy-core.

use tiy_core::agent::Agent;

fn main() {
    // Create an agent with default model
    let agent = Agent::new();

    println!("Agent created successfully");

    // Get current state
    let state = agent.state();
    println!("Initial state:");
    println!("  Messages: {}", state.messages.read().len());
    println!("  Tools: {}", state.tools.read().len());

    // Set a system prompt
    agent.set_system_prompt("You are a helpful coding assistant.");

    // Set tools (example)
    let tool = tiy_core::agent::AgentTool::new(
        "get_weather",
        "Get Weather",
        "Get the current weather for a location",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }),
    );

    agent.set_tools(vec![tool]);

    println!("\nAfter setting up:");
    println!("  Tools: {}", state.tools.read().len());
}
