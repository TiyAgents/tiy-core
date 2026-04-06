use chrono::Utc;
use reqwest::Client;
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use tiycore::catalog::{
    apply_model_patches, build_catalog_snapshot, build_catalog_snapshot_manifest,
    CatalogModelMetadata, ModelPatchConfig,
};

const DEFAULT_OPENROUTER_MODELS_URL: &str = "https://openrouter.ai/api/v1/models";
const DEFAULT_OPENROUTER_EMBEDDINGS_MODELS_URL: &str =
    "https://openrouter.ai/api/v1/embeddings/models";
const DEFAULT_PATCH_CONFIG_PATH: &str = "catalog/patches.json";

struct Args {
    output: PathBuf,
    manifest_output: PathBuf,
    snapshot_url: String,
    source_url: String,
    embeddings_source_url: String,
    patch_config: PathBuf,
    version: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let client = Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| Client::new());

    let response = client
        .get(&args.source_url)
        .header("accept", "application/json")
        .send()
        .await?
        .error_for_status()?;
    let payload = response.json::<Value>().await?;

    let embeddings_response = client
        .get(&args.embeddings_source_url)
        .header("accept", "application/json")
        .send()
        .await?
        .error_for_status()?;
    let embeddings_payload = embeddings_response.json::<Value>().await?;

    let models = merge_openrouter_payloads(&payload, Some(&embeddings_payload))?;
    let patch_config = load_patch_config(&args.patch_config)?;
    let models = apply_model_patches(models, &patch_config);

    let generated_at = Utc::now().to_rfc3339();
    let version = args
        .version
        .unwrap_or_else(|| Utc::now().format("%Y%m%d%H%M%S").to_string());
    let snapshot = build_catalog_snapshot(version.clone(), generated_at.clone(), models);
    let snapshot_bytes = serde_json::to_vec_pretty(&snapshot)?;
    let manifest =
        build_catalog_snapshot_manifest(version, generated_at, args.snapshot_url, &snapshot_bytes);

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = args.manifest_output.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(&args.output, snapshot_bytes)?;
    fs::write(&args.manifest_output, serde_json::to_vec_pretty(&manifest)?)?;
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut output = None;
    let mut manifest_output = None;
    let mut snapshot_url = None;
    let mut source_url = DEFAULT_OPENROUTER_MODELS_URL.to_string();
    let mut embeddings_source_url = DEFAULT_OPENROUTER_EMBEDDINGS_MODELS_URL.to_string();
    let mut patch_config = PathBuf::from(DEFAULT_PATCH_CONFIG_PATH);
    let mut version = None;

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--output" => output = iter.next().map(PathBuf::from),
            "--manifest-output" => manifest_output = iter.next().map(PathBuf::from),
            "--snapshot-url" => snapshot_url = iter.next(),
            "--source-url" => {
                source_url = iter.next().ok_or("--source-url requires a value")?;
            }
            "--embeddings-source-url" => {
                embeddings_source_url = iter
                    .next()
                    .ok_or("--embeddings-source-url requires a value")?;
            }
            "--patch-config" => {
                patch_config = iter
                    .next()
                    .map(PathBuf::from)
                    .ok_or("--patch-config requires a value")?;
            }
            "--version" => version = iter.next(),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                return Err(format!("unknown argument: {other}").into());
            }
        }
    }

    let output = output.ok_or("--output is required")?;
    let manifest_output = manifest_output.ok_or("--manifest-output is required")?;
    let snapshot_url = snapshot_url.ok_or("--snapshot-url is required")?;

    Ok(Args {
        output,
        manifest_output,
        snapshot_url,
        source_url,
        embeddings_source_url,
        patch_config,
        version,
    })
}

fn load_patch_config(path: &PathBuf) -> Result<ModelPatchConfig, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let config = serde_json::from_slice::<ModelPatchConfig>(&bytes)?;
    Ok(config)
}

fn print_help() {
    eprintln!(
        "\
tiy-catalog-sync

Build a catalog snapshot and manifest from OpenRouter's model catalog.

Required:
  --output <path>           Output path for catalog.json
  --manifest-output <path>  Output path for manifest.json
  --snapshot-url <url>      Public URL or relative path for catalog.json in manifest.json

Optional:
  --source-url <url>        Source models endpoint (default: {DEFAULT_OPENROUTER_MODELS_URL})
  --embeddings-source-url <url>
                            Source embeddings endpoint (default: {DEFAULT_OPENROUTER_EMBEDDINGS_MODELS_URL})
  --patch-config <path>     Patch config path (default: {DEFAULT_PATCH_CONFIG_PATH})
  --version <value>         Override snapshot version
"
    );
}

pub(crate) fn merge_openrouter_payloads(
    models_payload: &Value,
    embeddings_payload: Option<&Value>,
) -> Result<Vec<CatalogModelMetadata>, Box<dyn std::error::Error>> {
    let mut models = extract_openrouter_models(models_payload)?;
    if let Some(embeddings_payload) = embeddings_payload {
        append_unique_metadata(&mut models, extract_openrouter_models(embeddings_payload)?);
    }
    Ok(models)
}

pub(crate) fn extract_openrouter_models(
    payload: &Value,
) -> Result<Vec<CatalogModelMetadata>, Box<dyn std::error::Error>> {
    let data = payload
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| std::io::Error::other("payload is missing `data` array"))?;

    Ok(data
        .iter()
        .filter_map(|item| {
            let id = item.get("id")?.as_str()?.to_string();
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .or_else(|| Some(id.clone()));
            let description = item
                .get("description")
                .and_then(Value::as_str)
                .map(ToString::to_string);
            let context_window = item.get("context_length").and_then(parse_u64).or_else(|| {
                item.get("top_provider")
                    .and_then(|v| v.get("context_length"))
                    .and_then(parse_u64)
            });
            let max_output_tokens = item
                .get("top_provider")
                .and_then(|v| v.get("max_completion_tokens"))
                .and_then(parse_u64);
            let modalities = collect_modalities(item);
            let capabilities = collect_capabilities(item);
            let canonical_model_key = canonical_key_for(&id);

            Some(CatalogModelMetadata {
                canonical_model_key,
                aliases: vec![id.clone()],
                display_name: name,
                description,
                context_window,
                max_output_tokens,
                max_input_tokens: None,
                modalities,
                capabilities,
                pricing: item.get("pricing").cloned(),
                source: "openrouter".to_string(),
                raw: item.clone(),
            })
        })
        .collect())
}

fn parse_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.parse::<u64>().ok(),
        _ => None,
    }
}

fn canonical_key_for(id: &str) -> String {
    let mut chars = String::with_capacity(id.len());
    let mut last_sep = false;
    for ch in id.trim().to_lowercase().chars() {
        let mapped = match ch {
            'a'..='z' | '0'..='9' | '.' => Some(ch),
            '/' => Some(':'),
            '_' | ' ' | '-' => Some('-'),
            _ => None,
        };
        if let Some(mapped) = mapped {
            let is_sep = mapped == '-' || mapped == ':';
            if is_sep && last_sep {
                continue;
            }
            last_sep = is_sep;
            chars.push(mapped);
        }
    }
    chars.trim_matches(['-', ':']).to_string()
}

fn collect_modalities(item: &Value) -> Option<Vec<String>> {
    let architecture = item.get("architecture")?.as_object()?;
    let mut modalities = Vec::new();
    for key in ["input_modalities", "output_modalities"] {
        if let Some(values) = architecture.get(key).and_then(Value::as_array) {
            for value in values {
                if let Some(text) = value.as_str() {
                    modalities.push(text.to_string());
                }
            }
        }
    }

    let mut deduped = Vec::new();
    let mut seen = HashSet::new();
    for modality in modalities {
        if seen.insert(modality.clone()) {
            deduped.push(modality);
        }
    }

    if deduped.is_empty() {
        None
    } else {
        Some(deduped)
    }
}

fn collect_capabilities(item: &Value) -> Option<Vec<String>> {
    let parameters = item
        .get("supported_parameters")
        .and_then(parse_string_array)?;

    let mut capabilities = Vec::new();
    for parameter in parameters {
        match parameter.as_str() {
            "reasoning" | "include_reasoning" => capabilities.push("reasoning".to_string()),
            "tools" | "tool_choice" | "parallel_tool_calls" => {
                capabilities.push("tools".to_string())
            }
            "response_format" | "structured_outputs" => {
                capabilities.push("structured_outputs".to_string())
            }
            _ => {}
        }
    }

    dedupe_strings(capabilities)
}

fn parse_string_array(value: &Value) -> Option<Vec<String>> {
    match value {
        Value::Array(values) => {
            let items: Vec<String> = values
                .iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect();
            if items.is_empty() {
                None
            } else {
                Some(items)
            }
        }
        Value::String(text) => Some(vec![text.to_string()]),
        _ => None,
    }
}

fn dedupe_strings(values: Vec<String>) -> Option<Vec<String>> {
    let mut deduped = Vec::new();
    let mut seen = HashSet::new();
    for value in values {
        if seen.insert(value.clone()) {
            deduped.push(value);
        }
    }

    if deduped.is_empty() {
        None
    } else {
        Some(deduped)
    }
}

fn append_unique_metadata(
    target: &mut Vec<CatalogModelMetadata>,
    incoming: Vec<CatalogModelMetadata>,
) {
    let mut seen: HashSet<String> = target
        .iter()
        .map(|item| item.canonical_model_key.clone())
        .collect();
    for item in incoming {
        if seen.insert(item.canonical_model_key.clone()) {
            target.push(item);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_openrouter_payload_shape_into_catalog_metadata() {
        let payload = json!({
            "data": [
                {
                    "id": "openai/gpt-5.4-nano",
                    "name": "OpenAI: GPT-5.4 Nano",
                    "description": "Fast small GPT-5.4 variant",
                    "context_length": 400000,
                    "pricing": {
                        "prompt": "0.050",
                        "completion": "0.400"
                    },
                    "architecture": {
                        "input_modalities": ["text", "image", "file"],
                        "output_modalities": ["text"]
                    },
                    "top_provider": {
                        "max_completion_tokens": 128000
                    }
                },
                {
                    "id": "x-ai/grok-4.20-beta",
                    "name": "xAI: Grok 4.20 Beta",
                    "context_length": 2000000,
                    "architecture": {
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["text"]
                    }
                }
            ]
        });

        let models = extract_openrouter_models(&payload).expect("payload should parse");
        assert_eq!(models.len(), 2);

        let first = &models[0];
        assert_eq!(first.canonical_model_key, "openai:gpt-5.4-nano");
        assert_eq!(first.aliases, vec!["openai/gpt-5.4-nano".to_string()]);
        assert_eq!(first.display_name.as_deref(), Some("OpenAI: GPT-5.4 Nano"));
        assert_eq!(
            first.description.as_deref(),
            Some("Fast small GPT-5.4 variant")
        );
        assert_eq!(first.context_window, Some(400000));
        assert_eq!(first.max_output_tokens, Some(128000));
        assert_eq!(
            first.modalities.as_ref(),
            Some(&vec![
                "text".to_string(),
                "image".to_string(),
                "file".to_string()
            ])
        );
        assert_eq!(first.source, "openrouter");
        assert!(first.pricing.is_some());

        let second = &models[1];
        assert_eq!(second.canonical_model_key, "x-ai:grok-4.20-beta");
        assert_eq!(second.max_output_tokens, None);
        assert_eq!(second.context_window, Some(2000000));
    }

    #[test]
    fn extracts_reasoning_capability_from_openrouter_supported_parameters() {
        let payload = json!({
            "data": [
                {
                    "id": "google/gemini-2.5-flash-image",
                    "name": "Google: Gemini 2.5 Flash Image",
                    "supported_parameters": ["max_tokens", "seed", "stop"]
                },
                {
                    "id": "minimax/minimax-m2.7",
                    "name": "MiniMax: M2.7",
                    "supported_parameters": [
                        "max_tokens",
                        "include_reasoning",
                        "reasoning",
                        "tool_choice",
                        "tools"
                    ]
                }
            ]
        });

        let models = extract_openrouter_models(&payload).expect("payload should parse");
        assert_eq!(models.len(), 2);

        assert_eq!(models[0].capabilities, None);
        assert_eq!(
            models[1].capabilities,
            Some(vec!["reasoning".to_string(), "tools".to_string()])
        );
    }

    #[test]
    fn merges_embedding_models_into_snapshot_payload() {
        let models_payload = json!({
            "data": [
                {
                    "id": "openai/gpt-5.4-nano",
                    "name": "OpenAI: GPT-5.4 Nano",
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": ["text"]
                    }
                }
            ]
        });
        let embeddings_payload = json!({
            "data": [
                {
                    "id": "openai/text-embedding-3-small",
                    "name": "OpenAI: Text Embedding 3 Small",
                    "context_length": 8192,
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": ["embeddings"]
                    }
                }
            ]
        });

        let models = merge_openrouter_payloads(&models_payload, Some(&embeddings_payload))
            .expect("payloads should merge");

        assert_eq!(models.len(), 2);
        assert_eq!(
            models[1].canonical_model_key,
            "openai:text-embedding-3-small"
        );
        assert_eq!(
            models[1].modalities,
            Some(vec!["text".to_string(), "embeddings".to_string()])
        );
    }

    #[test]
    fn applies_patch_config_before_snapshot_generation() {
        let models = vec![CatalogModelMetadata {
            canonical_model_key: "z-ai:glm-5".to_string(),
            aliases: vec!["z-ai/glm-5".to_string()],
            display_name: Some("Z.AI: GLM-5".to_string()),
            description: None,
            context_window: Some(80_000),
            max_output_tokens: Some(16_384),
            max_input_tokens: None,
            modalities: None,
            capabilities: None,
            pricing: None,
            source: "openrouter".to_string(),
            raw: json!({}),
        }];

        let patched = apply_model_patches(
            models,
            &ModelPatchConfig {
                patches: vec![tiycore::catalog::ModelPatch {
                    source: "openrouter".to_string(),
                    alias: Some("z-ai/glm-5".to_string()),
                    canonical_model_key: None,
                    display_name: None,
                    description: None,
                    context_window: Some(200_000),
                    max_output_tokens: None,
                    max_input_tokens: None,
                    modalities: None,
                    capabilities: None,
                    pricing: None,
                    patch_source: Some("catalog-patch:openrouter:z-ai/glm-5".to_string()),
                }],
            },
        );

        assert_eq!(patched.len(), 1);
        assert_eq!(patched[0].context_window, Some(200_000));
        assert_eq!(patched[0].max_output_tokens, Some(16_384));
        assert_eq!(patched[0].source, "catalog-patch:openrouter:z-ai/glm-5");
    }
}
