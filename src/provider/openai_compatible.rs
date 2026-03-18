//! Generic OpenAI-compatible provider facade.
//!
//! This provider reuses the OpenAI Chat Completions protocol and lets callers
//! point `model.base_url` (or `StreamOptions.base_url`) at any compatible API.
//! For convenience, it reuses `OPENAI_API_KEY` as the default environment key.

use crate::stream::AssistantMessageEventStream;
use crate::types::*;

define_openai_delegation_provider! {
    name: OpenAICompatibleProvider,
    doc: "Generic OpenAI-compatible provider facade.",
    provider_type: Provider::OpenAICompatible,
    env_var: "OPENAI_API_KEY",
}
