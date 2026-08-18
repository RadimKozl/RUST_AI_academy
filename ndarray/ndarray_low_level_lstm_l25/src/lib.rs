use ndarray::{concatenate, Array1, Array2, Axis};
use rand_distr::{Distribution, Normal};

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh(x: f64) -> f64 {
    x.tanh()
}

pub struct LstmCell {
    // Weights for the merged input [h_{t-1}, x_t]
    // Forget gate
    wf: Array2<f64>,
    bf: Array1<f64>,
    // Input gate
    wi: Array2<f64>,
    bi: Array1<f64>,
    // Candidate cell state
    wc: Array2<f64>,
    bc: Array1<f64>,
    // Output gate
    wo: Array2<f64>,
    bo: Array1<f64>,

    hidden_size: usize,
}

impl LstmCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.1).expect("Failed to create normal distribution");

        let concat_size = hidden_size + input_size;

        // Helper function for initializing with weights    
        let mut init_w = || Array2::from_shape_fn((hidden_size, concat_size), |_| normal.sample(&mut rng));
        let init_b = || Array1::zeros(hidden_size);

        Self {
            wf: init_w(),
            bf: init_b(),
            wi: init_w(),
            bi: init_b(),
            wc: init_w(),
            bc: init_b(),
            wo: init_w(),
            bo: init_b(),
            hidden_size,
        }
    }

    /// Take one time step (Forward Step)
    pub fn step(
        &self,
        x_t: &Array1<f64>,
        h_prev: &Array1<f64>,
        c_prev: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        // Merge h_{t-1} and x_t into one vector
        let concat_input = concatenate(Axis(0), &[h_prev.view(), x_t.view()])
            .expect("Concatenation failed along Axis(0)");

        // 1. Forget Gate
        let f_t = (&self.wf.dot(&concat_input) + &self.bf).mapv(sigmoid);
        
        // 2. Input Gate & Cell Candidate
        let i_t = (&self.wi.dot(&concat_input) + &self.bi).mapv(sigmoid);
        let c_tilde = (&self.wc.dot(&concat_input) + &self.bc).mapv(tanh);

        // 3. New Cell State: C_t = f_t * C_{t-1} + i_t * C_tilde
        let c_t = (&f_t * c_prev) + (&i_t * &c_tilde);

        // 4. Output Gate & New Hidden State: h_t = o_t * tanh(C_t)
        let o_t = (&self.wo.dot(&concat_input) + &self.bo).mapv(sigmoid);
        let h_t = &o_t * &c_t.mapv(tanh);

        (h_t, c_t)
    }

    /// Processing the entire sequence step by step
    pub fn forward_sequence(&self, sequence: &Array2<f64>) -> Vec<Array1<f64>> { // Shape: [seq_len, input_size]
        let seq_len = sequence.nrows();
        let mut hidden_states = Vec::with_capacity(seq_len);

        let mut h_t = Array1::zeros(self.hidden_size);
        let mut c_t = Array1::zeros(self.hidden_size);

        for t in 0..seq_len {
            let x_t = sequence.row(t).to_owned();
            let (h_next, c_next) = self.step(&x_t, &h_t, &c_t);

            h_t = h_next;
            c_t = c_next;
            hidden_states.push(h_t.clone());
        }

        hidden_states
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}