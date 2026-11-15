# Collaborative Scientific Multi-Agent System (MAS) in Rust

An enterprise-grade, audit-ready **Multi-Agent System (MAS)** built in **Rust** using the `rig-core` LLM framework and `tokio` asynchronous runtime. The system orchestrates specialized autonomous agents (`Researcher` and `Coder`) through a deterministic state-driven execution graph to solve symbolic mathematics, perform Retrieval-Augmented Generation (RAG) over scientific literature and documentation, and generate high-resolution data visualizations.

---

## 1. Theoretical Foundations & Paradigm Shift

### 1.1 From Monolithic Prompts to Multi-Agent Systems (MAS)
Traditional Large Language Model (LLM) workflows often rely on a single monolithic prompt to handle information retrieval, mathematical reasoning, and code/artifact generation simultaneously. This approach suffers from **context degradation**, **hallucinations**, and **attention drift** (where an LLM loses track of tool-calling constraints when overwhelmed by long contextual histories or raw binary payloads).

A **Multi-Agent System (MAS)** decomposes complex workflows into specialized, decoupled roles:
* **Separation of Concerns:** Each agent possesses a strictly scoped system preamble, a tailored subset of tools, and a clear termination/handoff condition.
* **Deterministic State Routing:** Instead of letting an LLM decide the entire application lifecycle freely, transitions between agents are governed by an explicit finite-state router (`NextNode`), preventing infinite loops and ensuring predictable execution paths.

### 1.2 ReAct & Tool-Augmented Reasoning
Both agents operate under the **ReAct (Reason + Act)** paradigm:
1. **Thought:** The model analyzes the current conversation state (`ScientificMasState`) and determines what external knowledge or computation is required.
2. **Action:** The model invokes strongly-typed Rust tools via JSON Schema definitions (`schemars`).
3. **Observation:** The deterministic Rust backend executes the tool (e.g., querying Wikipedia, scraping docs, solving an ODE via Computer Algebra System, or rendering a PNG) and returns structured output.
4. **Synthesis / Handoff:** The agent either iterates or hands control over to the next node in the graph.

### 1.3 Explainable AI (XAI) & Auditability
In scientific and engineering domains, black-box answers are unacceptable. This architecture embeds **Explainable AI (XAI)** directly into its state management:
* Every state transition, agent reasoning step, and routing decision is recorded chronologically in a `TraceStep` audit log.
* The complete execution trace is serializable to structured JSON, enabling full post-hoc verification, debugging, and compliance auditing.

### 1.4 Neuro-Symbolic AI Integration
LLMs are probabilistic token predictors and notoriously unreliable at exact algebraic manipulation. By coupling the LLM (`Coder` agent) with a deterministic **Computer Algebra System (CAS)** (`mathcore`), the system achieves **Neuro-Symbolic AI**:
* The LLM translates natural language intent and retrieved syntax into formal mathematical arguments.
* The symbolic engine computes exact analytical derivatives, integrals, algebraic roots, and separable Ordinary Differential Equations (ODEs) without arithmetic hallucination.

---

## 2. System Architecture

The application is structured as a modular Rust library (`gfrig_collaborative_scientific_mas`) paired with an executable runner (`src/main.rs`) and an integration test suite (`tests/mas_integration_test.rs`).

### 2.1 Execution Graph Topology

```text
+-------+      +------------+      +-------------------+      +-------+
| START | ---> | Researcher | ---> | Route Researcher  | ---> | Coder |
+-------+      +------------+      +-------------------+      +-------+
                     ^                       |                    |
                     |                       v (Empty/Error)      v
                     |                    +-----+          +-------------+
                     +--------------------| END |<---------| Route Coder |
                   (Need Researcher)      +-----+          +-------------+
                                                                  |
                                                                  v (FINAL ANSWER)
                                                               +-----+
                                                               | END |
                                                               +-----+
```

### 2.2 Project Directory Structure

```text
gfrig_collaborative_scientific_mas/
├── Cargo.toml                      # Crate manifest, dependencies, and target definitions
├── README.md                       # Comprehensive architectural & theoretical documentation
├── src/
│   ├── lib.rs                      # Public library exports
│   ├── main.rs                     # Entry point executing Scenario 1 (Math) & Scenario 2 (Charts)
│   ├── state.rs                    # Shared state, message history, XAI trace, Base64 extractor
│   ├── graph.rs                    # Multi-agent orchestration loop, routing logic, anti-loop guards
│   ├── tools_researcher.rs         # Fact-finding & RAG tools (Wikipedia, arXiv, Docs Scraper)
│   └── tools_coder.rs              # Symbolic CAS engine (MathCore) & Plotters PNG renderer
└── tests/
    └── mas_integration_test.rs     # Unit and async integration tests for state and tools
```

---

## 3. Module-by-Module Code Breakdown

### 3.1 `src/state.rs` — State Management & XAI Tracing
Defines the single source of truth passed across the agent graph:
* **`AgentRole` (`User`, `Researcher`, `Coder`):** Strongly-typed enum identifying the author of each message in the conversation history.
* **`MasMessage` & `TraceStep`:** Data structures capturing conversational turns and low-level graph execution events (`step`, `node`, `action`, `details`).
* **`ScientificMasState`:** Holds the `original_query`, chronological `messages`, `loop_count` (for cycle detection), `generated_chart_b64` (extracted binary artifact), `trace` log, and `final_answer`.
* **`extract_base64_from_output`:** A deterministic parser that extracts Base64-encoded PNG payloads delimited by `---B64-START---` and `---B64-END---` markers, isolating binary data from conversational text.

### 3.2 `src/graph.rs` — Graph Orchestrator & Deterministic Routers
Implements `CollaborativeScientificMas`, which manages the lifecycle of a query:
* **Agent Initialization:** Builds two distinct `rig` agents backed by a local Ollama LLM instance (e.g., `qwen3.5:9b`).
  * **Researcher Preamble:** Constrained to single-step empirical lookup (`wikipedia`, `arxiv`, `search_library_docs`) and mandatory handoff via the trigger phrase `"Pass to Coder."`.
  * **Coder Preamble:** Constrained to execute `symbolic_calculate` or `render_bar_chart` immediately upon receiving data, prohibiting off-topic prose, and terminating with `"FINAL ANSWER:"`.
* **Execution Loop (`run`):** Iterates up to `max_graph_steps` (default: `10`), formatting the shared history via `state.format_history_for_prompt()` at every turn.
* **Routing & Safety Guards (`route_researcher`, `route_coder`):**
  * **Anti-Loop Protection:** Tracks consecutive `Researcher` iterations via `loop_count`. If `loop_count >= 3`, the router forcefully hands off control to the `Coder` to prevent infinite research loops.
  * **Bidirectional Handoff:** Allows the `Coder` to request missing information (`"need researcher"`) or conclude the workflow (`"FINAL ANSWER:"`).

### 3.3 `src/tools_researcher.rs` — Empirical & Documentation RAG Tools
Equips the `Researcher` agent with three `rig::tool::Tool` implementations:
1. **`WikipediaTool` (`wikipedia`):** Queries the official MediaWiki REST/Query API (`en.wikipedia.org/w/api.php`), strips HTML search-match tags, and returns top factual summaries.
2. **`ArxivTool` (`arxiv`):** Queries the arXiv preprint export API (`export.arxiv.org/api/query`), parses the Atom XML response using `roxmltree`, and extracts paper titles and abstracts sorted by submission date.
3. **`DocsRagTool` (`search_library_docs`):** Hybrid retrieval tool supporting `sympy`, `scipy`, `numpy`, `pandas`, and `mathcore`. For differential equation queries (`dsolve`, `dy/dx`), it provides verified symbolic mapping instructions; for general queries, it scrapes live documentation search pages via `reqwest` and `scraper`.

### 3.4 `src/tools_coder.rs` — Neuro-Symbolic Math & Visualization Tools
Equips the `Coder` agent with deterministic computational capabilities:
1. **`SymbolicCalculateTool` (`symbolic_calculate`):** Wraps the native Rust `mathcore` CAS engine to execute:
   * `differentiate` / `diff`: Symbolic differentiation with respect to $x$ ($\frac{d}{dx} f(x)$).
   * `integrate` / `int`: Symbolic indefinite integration ($\int f(x)\,dx + C$).
   * `solve` / `roots`: Algebraic root finding ($f(x) = 0$).
   * `evaluate` / `calc`: Exact numerical expression evaluation.
   * `ode_dy_dx_eq_y_fx` / `ode`: Analytical solver for first-order separable Ordinary Differential Equations of the form $\frac{dy}{dx} = y \cdot f(x)$, yielding $y(x) = C_1 \cdot e^{\int f(x)\,dx}$.
2. **`RenderBarChartTool` (`render_bar_chart`):** Uses the `plotters` crate (`BitMapBackend`) to render a $900 \times 540$ anti-aliased PNG bar chart to disk, reads the resulting binary file, encodes it into Base64, and returns both the file confirmation and delimited Base64 payload.

### 3.5 `src/main.rs` — Demonstration Scenarios
Executes two end-to-end multi-agent workflows sequentially:
* **Scenario 1 (Symbolic Mathematics):** *"Find the MathCore syntax for solving dy/dx = y * x^2. Then execute it."* — Demonstrates RAG syntax lookup followed by exact analytical ODE resolution ($y(x) = C_1 \cdot e^{x^3/3}$).
* **Scenario 2 (Empirical Research + Visualization):** *"Find the 5 largest countries by area. Then create a bar chart saved as 'top_5_countries.png'."* — Demonstrates Wikipedia data extraction followed by automated chart generation and full JSON XAI trace output.

---

## 4. Prerequisites & Installation

### 4.1 System Requirements
* **Rust Toolchain:** Edition 2024 compatible (`rustc 1.85+` recommended).
* **Local LLM Inference Server:** [Ollama](https://ollama.com/) running locally on default port `11434`.
* **Model Weights:** A tool-calling capable model pulled in Ollama (default configured in `main.rs` is `qwen3.5:9b`):
  ```bash
  ollama pull qwen3.5:9b
  ```

### 4.2 Build & Execution
Clone the repository and build the project using Cargo:

```bash
# Check compilation and dependencies
cargo check

# Run the full unit and integration test suite
cargo test

# Execute the multi-agent scenarios
cargo run
```

---

## 5. Verification & Testing Strategy

The test suite in `tests/mas_integration_test.rs` validates deterministic system components independently of LLM stochasticity:
* **`test_state_history_and_base64_extraction`:** Verifies message formatting, role attribution, and Base64 boundary parsing.
* **`test_symbolic_calculate_tool_ode_and_diff`:** Asserts exact analytical output for $\frac{dy}{dx} = y \cdot x^2$ and polynomial differentiation $\frac{d}{dx}(x^3 + 2x)$.
* **`test_render_bar_chart_tool_creates_png_and_base64`:** Uses `tempfile::tempdir` to verify physical PNG creation on disk and validates PNG magic bytes (`iVBORw0K`) in the Base64 stream.
* **`test_render_bar_chart_validation_error`:** Ensures mismatched label/value vector lengths return a structured `CoderToolError::Chart` error.
* **`test_docs_rag_tool_ode_syntax`:** Validates RAG retrieval of ODE solver syntax.

---

## 6. Prompt Engineering & Small-Model Guardrails

When orchestrating local 7B–9B parameter models (such as `qwen3.5:9b`), two common failure modes occur:
1. **Context Bleed / Attention Drift:** Long tool outputs or open-ended prompts can cause smaller models to abandon their agent persona and generate unrelated technical tutorials.
2. **Tool-Call Looping:** Without strict turn limits, a researcher agent may repeatedly query search APIs.

This codebase mitigates both issues through:
* **Imperative Negative Constraints** in `graph.rs` (explicitly forbidding off-topic prose and requiring immediate tool invocation).
* **Graph-Level Anti-Loop Guards** (`state.loop_count >= 3`) combined with bounded `max_turns` per agent invocation.

---

## 7. Future Roadmap

* **Reviewer / Verifier Agent:** Introduce a third `Reviewer` node in `graph.rs` to cross-check numerical values and symbolic derivations before emitting `FINAL ANSWER:`.
* **Persistent SQLite XAI Storage:** Utilize the bundled `rusqlite` dependency to persist `TraceStep` records and Base64 artifacts into a queryable audit database.
* **Expanded CAS Operations:** Extend `SymbolicCalculateTool` to support multivariable partial derivatives, matrix operations, and definite numerical integration.