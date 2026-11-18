use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SwarmAction {
    DeepArxivResearch,
    ProposeSpecification,
    CritiqueAndCompete,
    FormConsensus,
}

impl SwarmAction {
    pub fn all() -> &'static [SwarmAction] {
        &[
            SwarmAction::DeepArxivResearch,
            SwarmAction::ProposeSpecification,
            SwarmAction::CritiqueAndCompete,
            SwarmAction::FormConsensus,
        ]
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            SwarmAction::DeepArxivResearch => "DeepArxivResearch",
            SwarmAction::ProposeSpecification => "ProposeSpecification",
            SwarmAction::CritiqueAndCompete => "CritiqueAndCompete",
            SwarmAction::FormConsensus => "FormConsensus",
        }
    }
}

/// JSON only allows string map keys, but the MARL tables are keyed by tuples.
/// `graph-flow` stores its context as JSON, so these maps are (de)serialized as lists of entries.
mod entries_as_vec {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;
    use std::hash::Hash;

    pub fn serialize<K, V, S>(map: &HashMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
    where
        K: Serialize,
        V: Serialize,
        S: Serializer,
    {
        serializer.collect_seq(map.iter())
    }

    pub fn deserialize<'de, K, V, D>(deserializer: D) -> Result<HashMap<K, V>, D::Error>
    where
        K: Deserialize<'de> + Eq + Hash,
        V: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        let entries: Vec<(K, V)> = Vec::deserialize(deserializer)?;
        Ok(entries.into_iter().collect())
    }
}

/// MARL agent status with support for Joint-Action Learning & Agent Modeling (JAL-AM)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarlPolicyEngine {
    pub agent_name: String,
    /// Q_i(state, my_action)
    #[serde(with = "entries_as_vec")]
    pub q_values: HashMap<(String, SwarmAction), f64>,
    /// Agent Modeling: C(state, other_agent, other_action) to estimate \hat{\pi}_j(a_j | s)
    #[serde(with = "entries_as_vec")]
    pub opponent_action_counts: HashMap<(String, String, SwarmAction), usize>,
    /// Cumulative Regret Matching (Regret Matching, per Chapter 6.5 of the book MARL)
    pub cumulative_regret: HashMap<SwarmAction, f64>,
    pub alpha: f64,   // Learning rate
    pub gamma: f64,   // Discount factor
    pub epsilon: f64, // Exploration rate
    pub total_reward: f64,
}

impl MarlPolicyEngine {
    pub fn new(agent_name: &str) -> Self {
        let mut cumulative_regret = HashMap::new();
        for &act in SwarmAction::all() {
            cumulative_regret.insert(act, 0.0);
        }
        Self {
            agent_name: agent_name.to_string(),
            q_values: HashMap::new(),
            opponent_action_counts: HashMap::new(),
            cumulative_regret,
            alpha: 0.15,
            gamma: 0.95,
            epsilon: 0.25,
            total_reward: 0.0,
        }
    }

    /// Action selection combining Epsilon-Greedy and Regret-Matching probabilities
    pub fn select_action(&self, state: &str) -> SwarmAction {
        let mut rng = rand::rng();
        let actions = SwarmAction::all();

        if rng.random::<f64>() < self.epsilon {
            return actions[rng.random_range(0..actions.len())];
        }

        // Calculation of probabilities from positive regret (Regret Matching) + Q-values
        let mut weights = Vec::with_capacity(actions.len());
        let mut sum_weights = 0.0;

        for &act in actions {
            let q = *self.q_values.get(&(state.to_string(), act)).unwrap_or(&0.5);
            let regret = self.cumulative_regret.get(&act).copied().unwrap_or(0.0).max(0.0);
            let w = (q + 0.3 * regret).max(0.05);
            weights.push((act, w));
            sum_weights += w;
        }

        let mut pick = rng.random::<f64>() * sum_weights;
        for (act, w) in weights {
            if pick <= w {
                return act;
            }
            pick -= w;
        }
        SwarmAction::ProposeSpecification
    }

    /// Opponent Model Update (Policy Reconstruction) and TD-update of Q-values
    pub fn update_policy(
        &mut self,
        state: &str,
        my_action: SwarmAction,
        other_actions: &[(String, SwarmAction)],
        reward: f64,
        next_state: &str,
    ) {
        self.total_reward += reward;

        // 1. Updating counters for modeling other agents (JAL-AM)
        for (other_name, other_act) in other_actions {
            if other_name != &self.agent_name {
                let key = (state.to_string(), other_name.clone(), *other_act);
                *self.opponent_action_counts.entry(key).or_insert(0) += 1;
            }
        }

        // 2. Estimation of the maximum future value max_{a'} Q(s', a')
        let max_next_q = SwarmAction::all()
            .iter()
            .map(|&a| *self.q_values.get(&(next_state.to_string(), a)).unwrap_or(&0.0))
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.0);

        // 3. Temporal-Difference Update
        let current_q = *self
            .q_values
            .get(&(state.to_string(), my_action))
            .unwrap_or(&0.0);
        let td_target = reward + self.gamma * max_next_q;
        let new_q = current_q + self.alpha * (td_target - current_q);
        self.q_values
            .insert((state.to_string(), my_action), new_q);

        // 4. Updating Regret with respect to other actions in a given state
        for &alt_act in SwarmAction::all() {
            let alt_q = *self
                .q_values
                .get(&(state.to_string(), alt_act))
                .unwrap_or(&0.0);
            let instant_regret = alt_q - new_q;
            let entry = self.cumulative_regret.entry(alt_act).or_insert(0.0);
            *entry = (*entry + instant_regret).max(0.0);
        }

        // Gradual reduction of exploration (epsilon decay)
        self.epsilon = (self.epsilon * 0.92).max(0.05);
    }
}