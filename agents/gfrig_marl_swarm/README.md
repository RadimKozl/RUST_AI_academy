# AGON MARL Swarm (`gfrig_marl_swarm`)

An autonomous, formal-methods-aligned Multi-Agent System (MAS) written in Rust. This framework orchestrates collaborative and competitive AI agents using **Multi-Agent Reinforcement Learning (MARL)**, **GraphRAG (G-RAG)**, and **LLM-driven semantic synthesis** (`rig-core` + `graph-flow`) to evolve and verify the **AGON** programming language architecture.

---

## 📖 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [System Architecture & Codebase](#3-system-architecture--codebase)
4. [Step-by-Step Tutorial](#4-step-by-step-tutorial)
5. [Running Tests & Simulation](#5-running-tests--simulation)
6. [Evolutionary Output Example](#6-evolutionary-output-example)

---

## 1. Project Overview

Designing domain-specific languages (DSLs) for distributed AI systems requires balancing **compile-time safety** (preventing undefined behavior and unsafe predicate leakage) with **runtime adaptability** (allowing agents to dynamically negotiate weights and policies).

`gfrig_marl_swarm` solves this by pitting three specialized autonomous agents against—and alongside—one another in a general-sum MARL environment:
* **ArchitectAgent**: Focuses on semantic type safety, abstract syntax trees (`AgonAST`), Higher-Ranked Trait Bounds (HRTBs), and `PhantomData` contracts.
* **SystemsAgent**: Focuses on zero-copy message passing, Tokio actor runtimes (`ActorModelTokio`), and runtime performance.
* **GameTheoryCritic**: Audits proposals using game-theoretic principles (Nash equilibrium stability, QMIX monotonicity, and regret minimization), penalizing naive or unsafe designs.

---

## 2. Theoretical Foundations

### A. General-Sum MARL: JAL-AM & Regret Matching (`src/marl.rs`)
The agents do not rely on hardcoded workflows. Instead, each agent operates an independent `MarlPolicyEngine` combining:
1. **Joint-Action Learning with Agent Modeling (JAL-AM)**: Each agent tracks the historical action frequencies of its peers $C(s, j, a_j)$ to estimate opponent policies $\hat{\pi}_j(a_j \mid s)$ and adapt to non-stationarity.
2. **Temporal-Difference (TD) Q-Learning**: Updates action values using general-sum payoffs (a shared cooperative reward for high swarm confidence plus competitive bonuses/penalties):
   $$Q_i(s, a_i) \leftarrow Q_i(s, a_i) + \alpha \left[ r_i + \gamma \max_{a'_i} Q_i(s', a'_i) - Q_i(s, a_i) \right]$$
3. **Cumulative Regret Matching**: Balances $\epsilon$-greedy exploration with regret-weighted action selection, shifting probability mass toward actions that would have yielded higher historical payoffs.

### B. QMIX Value Decomposition & Semantic Type Safety
A central theoretical constraint enforced by the `GameTheoryCritic` is **QMIX monotonicity** in Centralized Training with Decentralized Execution (CTDE):
$$Q_{\text{tot}}(\tau, \mathbf{a}) = \sum_{i=1}^{N} \alpha_i Q_i(\tau_i, a_i), \quad \text{where } \frac{\partial Q_{\text{tot}}}{\partial Q_i} \ge 0 \iff \alpha_i \ge 0$$
Through iterative episodes, the swarm discovers that statically locking weight vectors ($\alpha_i$) at compile-time causes **Nash Equilibrium Collapse** in dynamic topologies. Consequently, the agents converge on **Runtime-Adaptive Semantic Types**—using Rust trait bounds and `PhantomData` to verify contracts at compile-time while computing gradient weights dynamically at runtime.

### C. Cross-Agent Synchronized GraphRAG (`src/memory.rs`)
Each agent maintains a multi-layered memory architecture:
* **Short-Term Memory**: A shared sliding window of typed `AgonIntentMessage` structs exchanged within the current episode.
* **Reasoning Trace**: An auditable log of `ReasoningStep` entries recording thoughts, tool invocations, and confidence scores.
* **Long-Term Memory (`PersonalGraphRag`)**: A vector-graph hybrid store using cosine similarity over embeddings followed by **1-hop graph expansion** along typed edges (`ENFORCED_BY`, `OPTIMIZES`, `INFORMS_DESIGN_OF`). At the end of every episode, `MarlConsensusEvaluatorNode` performs **cross-agent memory synchronization**, injecting all peer proposals into every agent's graph so knowledge accumulates evolutionarily across episodes.

---

## 3. System Architecture & Codebase

```text
gfrig_marl_swarm/
├── Cargo.toml                 # Dependencies (rig-core, graph-flow, tokio, reqwest, serde)
├── task.txt                   # Core AGON language specification & architectural pillars
├── agon_final_spec.txt        # Auto-generated winning specification from consensus
├── src/
│   ├── main.rs                # Entry point: initializes G-RAGs, MARL engines, and 12-episode loop
│   ├── lib.rs                 # Graph topology builder (Architect -> Systems -> Critic -> Evaluator)
│   ├── workflow.rs            # Task nodes, Rig LLM integration, fallback logic, and G-RAG sync
│   ├── marl.rs                # JAL-AM opponent modeling, TD Q-learning, and Regret Matching
│   ├── memory.rs              # PersonalGraphRag, vector similarity, and 1-hop graph traversal
│   └── tools.rs               # Rig tools: live arXiv API fetcher & personal G-RAG query tool
└── tests/
    └── gfrig_swarm_tests.rs   # Unit and end-to-end integration tests (offline & online)
```

### Key Components Explained
* **`AgonSwarmNode` (`src/workflow.rs`)**: Wraps a `rig::agent::Agent` connected to a local Ollama instance (`qwen3.5:4b` by default). It enforces `.max_turns(3)` to prevent tool-calling loops on small local models, strips `<think>...</think>` reasoning blocks, parses `CONFIDENCE: 0.XX` scores, and gracefully degrades to an autonomous G-RAG synthesis fallback if the LLM times out or hallucinates a tool name.
* **`ArxivSearchTool` & `GraphRagQueryTool` (`src/tools.rs`)**: Implements the `rig::tool::Tool` trait, allowing agents to query live scientific papers from `export.arxiv.org` and persist summaries directly into their isolated knowledge graphs.
* **`MarlConsensusEvaluatorNode` (`src/workflow.rs`)**: Evaluates net scores (`confidence - critique_penalties`), awards the round winner (`+0.75` competitive bonus), penalizes flawed proposals (`-0.50`), updates MARL policies, and synchronizes episode findings across all three G-RAG instances.

---

## 4. Step-by-Step Tutorial

### Prerequisites
1. **Rust Toolchain**: Install Rust (1.80+ recommended) via [rustup.rs](https://rustup.rs/).
2. **Ollama (Local LLM Runtime)**:
   * Install Ollama from [ollama.com](https://ollama.com/).
   * Pull the default model used by the swarm:
     ```bash
     ollama pull qwen3.5:4b
     ```
   * Ensure the Ollama server is running (`ollama serve`).

### Step 1: Configure Environment Variables (Optional)
Create a `.env` file in the root directory to customize the LLM model or toggle offline mode:
```env
# Override the default Ollama model (e.g., qwen3.5:4b, llama3.1, mistral)
OLLAMA_MODEL=qwen3.5:4b

# Set to "off" to bypass Ollama and run purely on deterministic G-RAG synthesis
AGON_LLM=on
```

### Step 2: Review `task.txt`
The swarm reads its primary architectural mission from `task.txt` at startup. You can modify `task.txt` to steer the swarm toward new language features, compiler passes, or formal verification goals.

### Step 3: Run the Multi-Episode MARL Simulation
Execute the swarm across 12 training episodes:
```bash
cargo run
```
During execution, you will observe:
1. **Action Selection**: Each agent selects between `DeepArxivResearch`, `ProposeSpecification`, `CritiqueAndCompete`, and `FormConsensus`.
2. **Epsilon Decay**: Exploration ($\epsilon$) decays from `0.250` down to `0.092` as agents converge on high-reward strategies.
3. **Consensus & Artifact Generation**: Whenever the winning net score exceeds `0.85`, consensus is reached and the winning specification is written to `agon_final_spec.txt`.
4. **Memory Growth**: At the end of the run, the terminal prints the node/edge growth of each agent's `PersonalGraphRag` (typically growing to 65–85+ nodes).

---

## 5. Running Tests & Simulation

The project includes a comprehensive test suite covering G-RAG isolation, JAL-AM Q-value math, Rig tool execution, general-sum payoff calculation, and full `graph-flow` execution.

Run all tests offline (without requiring Ollama):
```bash
cargo test
```

---

## 6. Evolutionary Output Example

After 12 episodes of adversarial critique and cross-agent G-RAG synchronization, the swarm converges on a **Runtime-Adaptive Type System** (`agon_final_spec.txt`) that avoids static weight locking while preserving Rust's compile-time trait guarantees:

```rust
pub trait MarlMessage {
    type Action;
    fn to_vector(&self) -> Vec<f32>; // Dynamically computed at runtime
}

#[derive(Debug, Clone)]
pub struct Intent<T> 
where
    T: MarlMessage + Send + Sync,
{
    pub action: PhantomData<T>, // Compile-time contract, no static weights
    pub sender_id: usize,
}

pub trait IntentSafety {
    fn validate_runtime_confidence(&self) -> f32;
}
```