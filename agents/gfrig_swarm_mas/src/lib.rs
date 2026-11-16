// src/lib.rs
pub mod state;
pub mod tools_researcher;
pub mod tools_coder;
pub mod graph_swarm;
pub mod visualizer;

pub use graph_swarm::SwarmScientificMas;
pub use state::{AgentRole, SwarmMasState};