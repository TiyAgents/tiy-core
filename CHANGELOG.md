# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] - 2026-04-14
### :sparkles: New Features
- [`d2d77c1`](https://github.com/TiyAgents/tiycore/commit/d2d77c13d8b5edee4315cb4baa2509efee286597) - **url-policy**: ✨ allow HTTP for configured HTTPS-exempt hosts *(commit by [@jorben](https://github.com/jorben))*


## [0.1.4] - 2026-04-12
### :bug: Bug Fixes
- [`5c60ce0`](https://github.com/TiyAgents/tiycore/commit/5c60ce0358fdea6e9a231347f7af3f7e1bafacfd) - 🐛 Use TIY_CACHE_RETENTION for cache retention *(commit by [@jorben](https://github.com/jorben))*


## [0.1.3] - 2026-04-07
### :sparkles: New Features
- [`303f1a2`](https://github.com/TiyAgents/tiycore/commit/303f1a208106734d425d896d51d4b2aec57f9e6b) - **catalog**: ✨ add scheduled snapshot patching via catalog/patches.json *(commit by [@jorben](https://github.com/jorben))*
- [`1f975ea`](https://github.com/TiyAgents/tiycore/commit/1f975ea3e3e82071420dfcd8d14ff68f58e32652) - **agent**: ✨ add custom HTTP headers support for LLM requests *(commit by [@jorben](https://github.com/jorben))*


## [0.1.2] - 2026-04-05
### :sparkles: New Features
- [`b862e72`](https://github.com/TiyAgents/tiycore/commit/b862e723059a39d4871786eb9ebdc135a651e644) - **provider**: ✨ implement Anthropic, Google, OpenAI Responses and add 7 new providers
- [`74d4970`](https://github.com/TiyAgents/tiycore/commit/74d49702c459ec583ef7b94b65a37347d7160995) - **agent**: ✨ implement full agent conversation loop with tool execution
- [`2fc925a`](https://github.com/TiyAgents/tiycore/commit/2fc925ac8df0f4cd1a0cd23280e3485b18db8ec8) - **provider**: ✨ add base_url override, tracing, and Google auth header *(commit by [@jorben](https://github.com/jorben))*
- [`13d64e1`](https://github.com/TiyAgents/tiycore/commit/13d64e17a9a84a718b69667d652ac003e65e6098) - **provider**: add Provider::OpenAIResponses variant and use registry pattern in example *(commit by [@jorben](https://github.com/jorben))*
- [`1841c36`](https://github.com/TiyAgents/tiycore/commit/1841c3639a760eb8cf9fb6735f66b7ac6e534f16) - **provider**: ✨ add get_registered_providers() and clarify model registry docs *(commit by [@jorben](https://github.com/jorben))*
- [`e7cd4ea`](https://github.com/TiyAgents/tiycore/commit/e7cd4ea26490db630075cd33469f2ae7e88023dd) - **google**: ✨ add Vertex AI URL format and auth header support *(commit by [@jorben](https://github.com/jorben))*
- [`96b1097`](https://github.com/TiyAgents/tiycore/commit/96b1097658b80f93563b1b3492df0510eadd42e4) - **security**: ✨ add SecurityConfig and comprehensive hardening across providers and agent *(commit by [@jorben](https://github.com/jorben))*
- [`7665729`](https://github.com/TiyAgents/tiycore/commit/7665729b4f0ee4fec939ca3c452294eb0b31e27a) - **agent**: ✨ implement full Agent capability set with provider integration *(commit by [@jorben](https://github.com/jorben))*
- [`c1f1203`](https://github.com/TiyAgents/tiycore/commit/c1f1203a09b4e0c755cff8c354a36bc69bcf99c4) - **provider**: ✨ wire onPayload hook into all protocol providers *(commit by [@jorben](https://github.com/jorben))*
- [`4293551`](https://github.com/TiyAgents/tiycore/commit/4293551d5717c509e9120503acf367bb2b344ac8) - **provider**: ✨ add DeepSeek provider (OpenAI-compatible delegation) *(commit by [@jorben](https://github.com/jorben))*
- [`cb7c8ce`](https://github.com/TiyAgents/tiycore/commit/cb7c8ce979f0da30181563ed32dc85615efc7997) - **protocol**: ✨ implement full compat-aware request building for OpenAI Completions *(commit by [@jorben](https://github.com/jorben))*
- [`04c032b`](https://github.com/TiyAgents/tiycore/commit/04c032b01826ccff692edf10fb386e81af88f21d) - **provider**: ✨ add OpenAI-compatible facade *(commit by [@jorben](https://github.com/jorben))*
- [`2b37988`](https://github.com/TiyAgents/tiycore/commit/2b37988ee02e73f559dd2a05b5d3b7795ee9eb61) - **catalog**: ✨ add model catalog snapshots and sync workflow *(commit by [@jorben](https://github.com/jorben))*
- [`f7b83de`](https://github.com/TiyAgents/tiycore/commit/f7b83dec5a8677f191a232613624b64190d91545) - **catalog**: ✨ add manual model enrichment API *(commit by [@jorben](https://github.com/jorben))*
- [`e229331`](https://github.com/TiyAgents/tiycore/commit/e2293318b7defad74de1ba4122c406dd9cd97201) - **catalog**: ✨ merge embedding and vertex model lists *(commit by [@jorben](https://github.com/jorben))*
- [`5cb8aa3`](https://github.com/TiyAgents/tiycore/commit/5cb8aa3da504918c9ec64592fcbcbe21eb47d7c7) - **agent**: ✨ align loop semantics *(commit by [@jorben](https://github.com/jorben))*
- [`873e309`](https://github.com/TiyAgents/tiycore/commit/873e309ed7fb60be9df45ce93c2ce2768a701524) - **agent**: ✨ add pi-mono runtime parity *(commit by [@jorben](https://github.com/jorben))*
- [`6f0e5e5`](https://github.com/TiyAgents/tiycore/commit/6f0e5e5505c1d0eea8f6afff644078be0da186c4) - **retry**: ✨ add transparent protocol-layer retries with Retry-After support *(commit by [@jorben](https://github.com/jorben))*
- [`2c028da`](https://github.com/TiyAgents/tiycore/commit/2c028dadba90ea62473dea6859a685bd7f1ed7a4) - **agent**: ✨ retry incomplete LLM streams from stable context *(commit by [@jorben](https://github.com/jorben))*
- [`87b6666`](https://github.com/TiyAgents/tiycore/commit/87b66669c8b40688a0d8c6d0e8324ded9612a038) - **catalog**: ✨ add vendor prefixes for additional AI providers *(commit by [@jorben](https://github.com/jorben))*

### :bug: Bug Fixes
- [`a7ed89c`](https://github.com/TiyAgents/tiycore/commit/a7ed89cd94f91d4f509f6a0a5b509df47486f120) - **openai-responses**: 🐛 extract event type from JSON data instead of SSE event line *(commit by [@jorben](https://github.com/jorben))*
- [`ad9e192`](https://github.com/TiyAgents/tiycore/commit/ad9e1920a4c58c76156a8757451c306c6f1611c6) - **catalog**: 🐛 detect reasoning from OpenRouter parameters *(commit by [@jorben](https://github.com/jorben))*
- [`4dcc8b7`](https://github.com/TiyAgents/tiycore/commit/4dcc8b78771900d6dba774dcd05f1ac0ca12a1a3) - **protocol**: 🐛 harden provider edge cases *(commit by [@jorben](https://github.com/jorben))*
- [`5c88c3a`](https://github.com/TiyAgents/tiycore/commit/5c88c3a92a42b59f8c1e20f84fa1f18606fd3ace) - **protocol**: 🐛 map simple-stream reasoning *(commit by [@jorben](https://github.com/jorben))*
- [`cf03c20`](https://github.com/TiyAgents/tiycore/commit/cf03c20157a98ac1a2f4ef9d28f99018d1ba3357) - **protocol**: 🐛 align replay parity with pi-mono *(commit by [@jorben](https://github.com/jorben))*
- [`79942b3`](https://github.com/TiyAgents/tiycore/commit/79942b37259f50978d3f6b5a5f1c2631ec67e2ba) - **protocol**: 🐛 align provider protocol options *(commit by [@jorben](https://github.com/jorben))*
- [`055708d`](https://github.com/TiyAgents/tiycore/commit/055708d4d5070bf45029c815e54f8b6adcde294e) - **protocol**: 🐛 close remaining pi-mono parity gaps *(commit by [@jorben](https://github.com/jorben))*
- [`3923e35`](https://github.com/TiyAgents/tiycore/commit/3923e3553f30a91b5762ace0b9c0df55fe2d1d8b) - **openai**: 🐛 Strip unstored response item IDs *(commit by [@jorben](https://github.com/jorben))*
- [`1433615`](https://github.com/TiyAgents/tiycore/commit/1433615eea340663a39779d0a9b5303ccb2347b9) - **agent**: 🐛 error on max turn limit exhaustion *(commit by [@jorben](https://github.com/jorben))*

### :recycle: Refactors
- [`2c850f8`](https://github.com/TiyAgents/tiycore/commit/2c850f81edf844e7e874aac894b8b2110343dde2) - ♻️ improve core types, stream, and model foundations
- [`d5eb78d`](https://github.com/TiyAgents/tiycore/commit/d5eb78dc2e04b63dce42eb2554355956cc565c61) - **provider**: ♻️ key ProviderRegistry by Provider instead of Api *(commit by [@jorben](https://github.com/jorben))*
- [`c3c27ee`](https://github.com/TiyAgents/tiycore/commit/c3c27ee95614114bdd0e88d37735290c31bb1111) - **model**: ♻️ make base_url optional, let providers own defaults *(commit by [@jorben](https://github.com/jorben))*
- [`024f1c6`](https://github.com/TiyAgents/tiycore/commit/024f1c6586321539c5b26e66b5f55240fd537f63) - **zenmux**: ♻️ adaptive 3-way protocol routing with Vertex AI support *(commit by [@jorben](https://github.com/jorben))*
- [`0d63c68`](https://github.com/TiyAgents/tiycore/commit/0d63c68da80fbc22eb122811c87834998d771ee0) - ♻️ modular architecture optimization (P0/P1) *(commit by [@jorben](https://github.com/jorben))*
- [`439a09b`](https://github.com/TiyAgents/tiycore/commit/439a09bbf0a4567c78f2ba6afea2329af6b35ad0) - ♻️ separate protocol and provider naming to eliminate ambiguity *(commit by [@jorben](https://github.com/jorben))*
- [`4736540`](https://github.com/TiyAgents/tiycore/commit/47365404350e7376b14ce6a030328c7e1c69a86d) - ♻️ move registry and delegation from protocol/ to provider/ *(commit by [@jorben](https://github.com/jorben))*

### :white_check_mark: Tests
- [`88199ef`](https://github.com/TiyAgents/tiycore/commit/88199effbfccfbae2d9476451ff92f5f7b435812) - ✅ add provider tests for Ollama, Zenmux, and delegation providers *(commit by [@jorben](https://github.com/jorben))*
- [`647f02d`](https://github.com/TiyAgents/tiycore/commit/647f02d3eef43774f83d261092983fae9cab289b) - ✅ improve test coverage from 75.59% to 85.80% *(commit by [@jorben](https://github.com/jorben))*

### :wrench: Chores
- [`5180071`](https://github.com/TiyAgents/tiycore/commit/518007167af688b1265b7823d6c0bb200ce5e71e) - 🔧 clean up basic_usage example
- [`cfa9803`](https://github.com/TiyAgents/tiycore/commit/cfa9803aff7522192d0dfea9d0dbbb9eafa381f4) - ✨ rename crate ti y-core to tiycore *(commit by [@jorben](https://github.com/jorben))*


## [0.1.1] - 2026-04-05
### :sparkles: New Features
- [`b862e72`](https://github.com/TiyAgents/tiycore/commit/b862e723059a39d4871786eb9ebdc135a651e644) - **provider**: ✨ implement Anthropic, Google, OpenAI Responses and add 7 new providers
- [`74d4970`](https://github.com/TiyAgents/tiycore/commit/74d49702c459ec583ef7b94b65a37347d7160995) - **agent**: ✨ implement full agent conversation loop with tool execution
- [`2fc925a`](https://github.com/TiyAgents/tiycore/commit/2fc925ac8df0f4cd1a0cd23280e3485b18db8ec8) - **provider**: ✨ add base_url override, tracing, and Google auth header *(commit by [@jorben](https://github.com/jorben))*
- [`13d64e1`](https://github.com/TiyAgents/tiycore/commit/13d64e17a9a84a718b69667d652ac003e65e6098) - **provider**: add Provider::OpenAIResponses variant and use registry pattern in example *(commit by [@jorben](https://github.com/jorben))*
- [`1841c36`](https://github.com/TiyAgents/tiycore/commit/1841c3639a760eb8cf9fb6735f66b7ac6e534f16) - **provider**: ✨ add get_registered_providers() and clarify model registry docs *(commit by [@jorben](https://github.com/jorben))*
- [`e7cd4ea`](https://github.com/TiyAgents/tiycore/commit/e7cd4ea26490db630075cd33469f2ae7e88023dd) - **google**: ✨ add Vertex AI URL format and auth header support *(commit by [@jorben](https://github.com/jorben))*
- [`96b1097`](https://github.com/TiyAgents/tiycore/commit/96b1097658b80f93563b1b3492df0510eadd42e4) - **security**: ✨ add SecurityConfig and comprehensive hardening across providers and agent *(commit by [@jorben](https://github.com/jorben))*
- [`7665729`](https://github.com/TiyAgents/tiycore/commit/7665729b4f0ee4fec939ca3c452294eb0b31e27a) - **agent**: ✨ implement full Agent capability set with provider integration *(commit by [@jorben](https://github.com/jorben))*
- [`c1f1203`](https://github.com/TiyAgents/tiycore/commit/c1f1203a09b4e0c755cff8c354a36bc69bcf99c4) - **provider**: ✨ wire onPayload hook into all protocol providers *(commit by [@jorben](https://github.com/jorben))*
- [`4293551`](https://github.com/TiyAgents/tiycore/commit/4293551d5717c509e9120503acf367bb2b344ac8) - **provider**: ✨ add DeepSeek provider (OpenAI-compatible delegation) *(commit by [@jorben](https://github.com/jorben))*
- [`cb7c8ce`](https://github.com/TiyAgents/tiycore/commit/cb7c8ce979f0da30181563ed32dc85615efc7997) - **protocol**: ✨ implement full compat-aware request building for OpenAI Completions *(commit by [@jorben](https://github.com/jorben))*
- [`04c032b`](https://github.com/TiyAgents/tiycore/commit/04c032b01826ccff692edf10fb386e81af88f21d) - **provider**: ✨ add OpenAI-compatible facade *(commit by [@jorben](https://github.com/jorben))*
- [`2b37988`](https://github.com/TiyAgents/tiycore/commit/2b37988ee02e73f559dd2a05b5d3b7795ee9eb61) - **catalog**: ✨ add model catalog snapshots and sync workflow *(commit by [@jorben](https://github.com/jorben))*
- [`f7b83de`](https://github.com/TiyAgents/tiycore/commit/f7b83dec5a8677f191a232613624b64190d91545) - **catalog**: ✨ add manual model enrichment API *(commit by [@jorben](https://github.com/jorben))*
- [`e229331`](https://github.com/TiyAgents/tiycore/commit/e2293318b7defad74de1ba4122c406dd9cd97201) - **catalog**: ✨ merge embedding and vertex model lists *(commit by [@jorben](https://github.com/jorben))*
- [`5cb8aa3`](https://github.com/TiyAgents/tiycore/commit/5cb8aa3da504918c9ec64592fcbcbe21eb47d7c7) - **agent**: ✨ align loop semantics *(commit by [@jorben](https://github.com/jorben))*
- [`873e309`](https://github.com/TiyAgents/tiycore/commit/873e309ed7fb60be9df45ce93c2ce2768a701524) - **agent**: ✨ add pi-mono runtime parity *(commit by [@jorben](https://github.com/jorben))*
- [`6f0e5e5`](https://github.com/TiyAgents/tiycore/commit/6f0e5e5505c1d0eea8f6afff644078be0da186c4) - **retry**: ✨ add transparent protocol-layer retries with Retry-After support *(commit by [@jorben](https://github.com/jorben))*
- [`2c028da`](https://github.com/TiyAgents/tiycore/commit/2c028dadba90ea62473dea6859a685bd7f1ed7a4) - **agent**: ✨ retry incomplete LLM streams from stable context *(commit by [@jorben](https://github.com/jorben))*
- [`87b6666`](https://github.com/TiyAgents/tiycore/commit/87b66669c8b40688a0d8c6d0e8324ded9612a038) - **catalog**: ✨ add vendor prefixes for additional AI providers *(commit by [@jorben](https://github.com/jorben))*

### :bug: Bug Fixes
- [`a7ed89c`](https://github.com/TiyAgents/tiycore/commit/a7ed89cd94f91d4f509f6a0a5b509df47486f120) - **openai-responses**: 🐛 extract event type from JSON data instead of SSE event line *(commit by [@jorben](https://github.com/jorben))*
- [`ad9e192`](https://github.com/TiyAgents/tiycore/commit/ad9e1920a4c58c76156a8757451c306c6f1611c6) - **catalog**: 🐛 detect reasoning from OpenRouter parameters *(commit by [@jorben](https://github.com/jorben))*
- [`4dcc8b7`](https://github.com/TiyAgents/tiycore/commit/4dcc8b78771900d6dba774dcd05f1ac0ca12a1a3) - **protocol**: 🐛 harden provider edge cases *(commit by [@jorben](https://github.com/jorben))*
- [`5c88c3a`](https://github.com/TiyAgents/tiycore/commit/5c88c3a92a42b59f8c1e20f84fa1f18606fd3ace) - **protocol**: 🐛 map simple-stream reasoning *(commit by [@jorben](https://github.com/jorben))*
- [`cf03c20`](https://github.com/TiyAgents/tiycore/commit/cf03c20157a98ac1a2f4ef9d28f99018d1ba3357) - **protocol**: 🐛 align replay parity with pi-mono *(commit by [@jorben](https://github.com/jorben))*
- [`79942b3`](https://github.com/TiyAgents/tiycore/commit/79942b37259f50978d3f6b5a5f1c2631ec67e2ba) - **protocol**: 🐛 align provider protocol options *(commit by [@jorben](https://github.com/jorben))*
- [`055708d`](https://github.com/TiyAgents/tiycore/commit/055708d4d5070bf45029c815e54f8b6adcde294e) - **protocol**: 🐛 close remaining pi-mono parity gaps *(commit by [@jorben](https://github.com/jorben))*
- [`3923e35`](https://github.com/TiyAgents/tiycore/commit/3923e3553f30a91b5762ace0b9c0df55fe2d1d8b) - **openai**: 🐛 Strip unstored response item IDs *(commit by [@jorben](https://github.com/jorben))*
- [`1433615`](https://github.com/TiyAgents/tiycore/commit/1433615eea340663a39779d0a9b5303ccb2347b9) - **agent**: 🐛 error on max turn limit exhaustion *(commit by [@jorben](https://github.com/jorben))*

### :recycle: Refactors
- [`2c850f8`](https://github.com/TiyAgents/tiycore/commit/2c850f81edf844e7e874aac894b8b2110343dde2) - ♻️ improve core types, stream, and model foundations
- [`d5eb78d`](https://github.com/TiyAgents/tiycore/commit/d5eb78dc2e04b63dce42eb2554355956cc565c61) - **provider**: ♻️ key ProviderRegistry by Provider instead of Api *(commit by [@jorben](https://github.com/jorben))*
- [`c3c27ee`](https://github.com/TiyAgents/tiycore/commit/c3c27ee95614114bdd0e88d37735290c31bb1111) - **model**: ♻️ make base_url optional, let providers own defaults *(commit by [@jorben](https://github.com/jorben))*
- [`024f1c6`](https://github.com/TiyAgents/tiycore/commit/024f1c6586321539c5b26e66b5f55240fd537f63) - **zenmux**: ♻️ adaptive 3-way protocol routing with Vertex AI support *(commit by [@jorben](https://github.com/jorben))*
- [`0d63c68`](https://github.com/TiyAgents/tiycore/commit/0d63c68da80fbc22eb122811c87834998d771ee0) - ♻️ modular architecture optimization (P0/P1) *(commit by [@jorben](https://github.com/jorben))*
- [`439a09b`](https://github.com/TiyAgents/tiycore/commit/439a09bbf0a4567c78f2ba6afea2329af6b35ad0) - ♻️ separate protocol and provider naming to eliminate ambiguity *(commit by [@jorben](https://github.com/jorben))*
- [`4736540`](https://github.com/TiyAgents/tiycore/commit/47365404350e7376b14ce6a030328c7e1c69a86d) - ♻️ move registry and delegation from protocol/ to provider/ *(commit by [@jorben](https://github.com/jorben))*

### :white_check_mark: Tests
- [`88199ef`](https://github.com/TiyAgents/tiycore/commit/88199effbfccfbae2d9476451ff92f5f7b435812) - ✅ add provider tests for Ollama, Zenmux, and delegation providers *(commit by [@jorben](https://github.com/jorben))*
- [`647f02d`](https://github.com/TiyAgents/tiycore/commit/647f02d3eef43774f83d261092983fae9cab289b) - ✅ improve test coverage from 75.59% to 85.80% *(commit by [@jorben](https://github.com/jorben))*

### :wrench: Chores
- [`5180071`](https://github.com/TiyAgents/tiycore/commit/518007167af688b1265b7823d6c0bb200ce5e71e) - 🔧 clean up basic_usage example
- [`cfa9803`](https://github.com/TiyAgents/tiycore/commit/cfa9803aff7522192d0dfea9d0dbbb9eafa381f4) - ✨ rename crate ti y-core to tiycore *(commit by [@jorben](https://github.com/jorben))*

[0.1.1]: https://github.com/TiyAgents/tiycore/compare/0.0.1...0.1.1
[0.1.2]: https://github.com/TiyAgents/tiycore/compare/0.0.1...0.1.2
[0.1.3]: https://github.com/TiyAgents/tiycore/compare/0.1.2...0.1.3
[0.1.4]: https://github.com/TiyAgents/tiycore/compare/0.1.3...0.1.4
[0.1.6]: https://github.com/TiyAgents/tiycore/compare/0.1.5...0.1.6
