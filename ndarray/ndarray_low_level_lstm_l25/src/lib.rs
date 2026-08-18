use ndarray::{concatenate, Array1, Array2, Axis};
use rand_distr::{Distribution, Normal};

/// Activation function: Sigmoid
/// Maps input values to the range (0, 1). Used for LSTM gates.
fn sigmoid(x: f64) -> f64 { 
    1.0 / (1.0 + (-x).exp()) 
}

/// Derivative of Sigmoid given its output y = sigmoid(x)
/// Formula: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
fn dsigmoid(y: f64) -> f64 { 
    y * (1.0 - y) 
}

/// Derivative of Tanh given its output y = tanh(x)
/// Formula: d/dx tanh(x) = 1 - tanh^2(x)
fn dtanh(y: f64) -> f64 { 
    1.0 - y * y 
}

/// Holds the internal activation states for a single time step `t`.
/// Required during the Backward Pass (BPTT) to calculate exact gradients.
#[derive(Clone)]
pub struct LstmStepCache {
    /// Input vector x_t at time step t. Shape: (input_size)
    pub x_t: Array1<f64>,
    /// Previous hidden state h_{t-1}. Shape: (hidden_size)
    pub h_prev: Array1<f64>,
    /// Previous cell state c_{t-1}. Shape: (hidden_size)
    pub c_prev: Array1<f64>,
    /// Forget gate activation vector f_t. Shape: (hidden_size)
    pub f_t: Array1<f64>,
    /// Input gate activation vector i_t. Shape: (hidden_size)
    pub i_t: Array1<f64>,
    /// Candidate cell state c_tilde_t. Shape: (hidden_size)
    pub c_tilde: Array1<f64>,
    /// Updated cell state c_t. Shape: (hidden_size)
    pub c_t: Array1<f64>,
    /// Output gate activation vector o_t. Shape: (hidden_size)
    pub o_t: Array1<f64>,
    /// Updated hidden state h_t. Shape: (hidden_size)
    pub h_t: Array1<f64>,
    /// Concatenated vector [h_{t-1}, x_t]. Shape: (hidden_size + input_size)
    pub concat_input: Array1<f64>,
}

/// Container for all parameter gradients accumulated during Backpropagation Through Time.
pub struct LstmGradients {
    /// Gradient for Forget Gate weights Wf. Shape: (hidden_size, hidden_size + input_size)
    pub dwf: Array2<f64>, 
    /// Gradient for Forget Gate bias bf. Shape: (hidden_size)
    pub dbf: Array1<f64>,
    /// Gradient for Input Gate weights Wi. Shape: (hidden_size, hidden_size + input_size)
    pub dwi: Array2<f64>, 
    /// Gradient for Input Gate bias bi. Shape: (hidden_size)
    pub dbi: Array1<f64>,
    /// Gradient for Candidate Cell weights Wc. Shape: (hidden_size, hidden_size + input_size)
    pub dwc: Array2<f64>, 
    /// Gradient for Candidate Cell bias bc. Shape: (hidden_size)
    pub dbc: Array1<f64>,
    /// Gradient for Output Gate weights Wo. Shape: (hidden_size, hidden_size + input_size)
    pub dwo: Array2<f64>, 
    /// Gradient for Output Gate bias bo. Shape: (hidden_size)
    pub dbo: Array1<f64>,
    /// Gradient for Dense Output Layer weights Wy. Shape: (output_size, hidden_size)
    pub dwy: Array2<f64>, 
    /// Gradient for Dense Output Layer bias by. Shape: (output_size)
    pub dby: Array1<f64>,
}

/// Low-level LSTM network model with a Linear Dense Output layer.
pub struct LstmModel {
    /// Forget gate weight matrix Wf. Shape: (hidden_size, hidden_size + input_size)
    pub wf: Array2<f64>, 
    /// Forget gate bias vector bf. Shape: (hidden_size)
    pub bf: Array1<f64>,
    /// Input gate weight matrix Wi. Shape: (hidden_size, hidden_size + input_size)
    pub wi: Array2<f64>, 
    /// Input gate bias vector bi. Shape: (hidden_size)
    pub bi: Array1<f64>,
    /// Candidate cell weight matrix Wc. Shape: (hidden_size, hidden_size + input_size)
    pub wc: Array2<f64>, 
    /// Candidate cell bias vector bc. Shape: (hidden_size)
    pub bc: Array1<f64>,
    /// Output gate weight matrix Wo. Shape: (hidden_size, hidden_size + input_size)
    pub wo: Array2<f64>, 
    /// Output gate bias vector bo. Shape: (hidden_size)
    pub bo: Array1<f64>,
    /// Output projection weight matrix Wy. Shape: (output_size, hidden_size)
    pub wy: Array2<f64>, 
    /// Output projection bias vector by. Shape: (output_size)
    pub by: Array1<f64>,
    
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
}

impl LstmModel {
    /// Initializes an LSTM model with random normal weight initialization ~ N(0, 0.1) and zero biases.
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.1).expect("Normal distribution creation failed");
        let concat_size = hidden_size + input_size;

        // Helper closure to create weight matrices filled with values from a normal distribution
        let mut init_w = |r, c| Array2::from_shape_fn((r, c), |_| normal.sample(&mut rng));

        Self {
            wf: init_w(hidden_size, concat_size), bf: Array1::zeros(hidden_size),
            wi: init_w(hidden_size, concat_size), bi: Array1::zeros(hidden_size),
            wc: init_w(hidden_size, concat_size), bc: Array1::zeros(hidden_size),
            wo: init_w(hidden_size, concat_size), bo: Array1::zeros(hidden_size),
            wy: init_w(output_size, hidden_size), by: Array1::zeros(output_size),
            input_size, hidden_size, output_size,
        }
    }

    /// Performs the Forward Pass over a sequence of inputs.
    ///
    /// # Arguments
    /// * `sequence` - Matrix of shape (seq_len, input_size) containing input time steps.
    ///
    /// # Returns
    /// * `predictions` - Output predictions for each step t, Shape: Vec<(output_size)>
    /// * `hidden_states` - Hidden states h_t for each step t, Shape: Vec<(hidden_size)>
    /// * `cache` - Step caches storing intermediate activation states for BPTT.
    pub fn forward(&self, sequence: &Array2<f64>) -> (Vec<Array1<f64>>, Vec<Array1<f64>>, Vec<LstmStepCache>) {
        let seq_len = sequence.nrows();
        let mut cache = Vec::with_capacity(seq_len);
        let mut predictions = Vec::with_capacity(seq_len);
        let mut hidden_states = Vec::with_capacity(seq_len);

        // Initial hidden and cell states initialized to zero
        let mut h_t = Array1::zeros(self.hidden_size);
        let mut c_t = Array1::zeros(self.hidden_size);

        for t in 0..seq_len {
            let x_t = sequence.row(t).to_owned();
            // Concatenate h_{t-1} and x_t along Axis 0 -> Shape: (hidden_size + input_size)
            let concat_input = concatenate(Axis(0), &[h_t.view(), x_t.view()]).unwrap();

            // 1. Forget Gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
            let f_t = (&self.wf.dot(&concat_input) + &self.bf).mapv(sigmoid);
            // 2. Input Gate: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
            let i_t = (&self.wi.dot(&concat_input) + &self.bi).mapv(sigmoid);
            // 3. Candidate Cell State: c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
            let c_tilde = (&self.wc.dot(&concat_input) + &self.bc).mapv(f64::tanh);
            // 4. Update Cell State: c_t = f_t * c_{t-1} + i_t * c̃_t
            let c_next = (&f_t * &c_t) + (&i_t * &c_tilde);

            // 5. Output Gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
            let o_t = (&self.wo.dot(&concat_input) + &self.bo).mapv(sigmoid);
            // 6. Updated Hidden State: h_t = o_t * tanh(c_t)
            let h_next = &o_t * &c_next.mapv(f64::tanh);

            // 7. Linear Output Projection: y_hat = W_y · h_t + b_y
            let y_hat = &self.wy.dot(&h_next) + &self.by;

            cache.push(LstmStepCache {
                x_t, h_prev: h_t.clone(), c_prev: c_t.clone(),
                f_t, i_t, c_tilde, c_t: c_next.clone(), o_t, h_t: h_next.clone(), concat_input,
            });

            h_t = h_next;
            c_t = c_next;
            hidden_states.push(h_t.clone());
            predictions.push(y_hat);
        }

        (predictions, hidden_states, cache)
    }

    /// Performs Backpropagation Through Time (BPTT) to compute gradients for all model parameters.
    ///
    /// Iterates backwards from time step T-1 down to 0, accumulating gradients through time and cell states.
    pub fn backward(&self, cache: &[LstmStepCache], targets: &Array2<f64>, predictions: &[Array1<f64>]) -> LstmGradients {
        let seq_len = cache.len();
        let concat_size = self.hidden_size + self.input_size;

        // Initialize zero gradient accumulators for all weights and biases
        let mut dwf = Array2::zeros((self.hidden_size, concat_size));
        let mut dbf = Array1::zeros(self.hidden_size);
        let mut dwi = Array2::zeros((self.hidden_size, concat_size));
        let mut dbi = Array1::zeros(self.hidden_size);
        let mut dwc = Array2::zeros((self.hidden_size, concat_size));
        let mut dbc = Array1::zeros(self.hidden_size);
        let mut dwo = Array2::zeros((self.hidden_size, concat_size));
        let mut dbo = Array1::zeros(self.hidden_size);
        let mut dwy = Array2::zeros((self.output_size, self.hidden_size));
        let mut dby = Array1::zeros(self.output_size);

        // Gradients passed backward from step t+1 to step t
        let mut dh_next = Array1::zeros(self.hidden_size);
        let mut dc_next = Array1::zeros(self.hidden_size);

        for t in (0..seq_len).rev() {
            let step = &cache[t];
            let target = targets.row(t);
            let pred = &predictions[t];

            // Mean Squared Error Loss derivative: dL/dy_hat = 2 * (y_hat - y) / N
            let dy = (pred - &target) * (2.0 / seq_len as f64);

            // Gradients for Output Projection Layer
            dwy += &dy.split_into_shapes().0.dot(&step.h_t.split_into_shapes().1);
            dby += &dy;

            // Gradient flow into current hidden state h_t (from dense output layer + future hidden state dh_next)
            let dh = self.wy.t().dot(&dy) + &dh_next;

            let tanh_c = step.c_t.mapv(f64::tanh);
            
            // Output Gate Gradient: do_t = dh * tanh(c_t) * σ'(o_t)
            let do_g = &dh * &tanh_c * step.o_t.mapv(dsigmoid);

            // Cell State Gradient: dc_t = dh * o_t * (1 - tanh^2(c_t)) + dc_next
            let dc = &dh * &step.o_t * tanh_c.mapv(dtanh) + &dc_next;

            // Candidate Cell Gradient: dc̃_t = dc * i_t * (1 - c̃_t^2)
            let dc_tilde = &dc * &step.i_t * step.c_tilde.mapv(dtanh);

            // Input Gate Gradient: di_t = dc * c̃_t * σ'(i_t)
            let di_g = &dc * &step.c_tilde * step.i_t.mapv(dsigmoid);

            // Forget Gate Gradient: df_t = dc * c_{t-1} * σ'(f_t)
            let df_g = &dc * &step.c_prev * step.f_t.mapv(dsigmoid);

            // Outer products for accumulated gate weight gradients: dW = d_gate ⊗ [h_{t-1}, x_t]^T
            let concat_t = step.concat_input.split_into_shapes();
            dwf += &df_g.split_into_shapes().0.dot(&concat_t.1); dbf += &df_g;
            dwi += &di_g.split_into_shapes().0.dot(&concat_t.1); dbi += &di_g;
            dwc += &dc_tilde.split_into_shapes().0.dot(&concat_t.1); dbc += &dc_tilde;
            dwo += &do_g.split_into_shapes().0.dot(&concat_t.1); dbo += &do_g;

            // Compute gradient for concatenated vector [h_{t-1}, x_t]
            let dconcat = self.wf.t().dot(&df_g) + self.wi.t().dot(&di_g) + self.wc.t().dot(&dc_tilde) + self.wo.t().dot(&do_g);

            // Split dconcat to extract dh_{t-1} for the next step iteration (stored in dh_next)
            dh_next = dconcat.slice(ndarray::s![0..self.hidden_size]).to_owned();

            // Cell state gradient for previous time step: dc_{t-1} = dc_t * f_t
            dc_next = &dc * &step.f_t;
        }

        LstmGradients { dwf, dbf, dwi, dbi, dwc, dbc, dwo, dbo, dwy, dby }
    }

    /// Updates model weights using Gradient Descent with explicit Gradient Clipping.
    pub fn apply_gradients(&mut self, grads: &LstmGradients, lr: f64) {
        let max_norm = 1.0;

        // Function to clamp gradient values within [-max_norm, max_norm] to prevent exploding gradients
        let clip = |g: f64| g.clamp(-max_norm, max_norm) * lr;

        self.wf -= &(grads.dwf.mapv(clip)); 
        self.bf -= &(grads.dbf.mapv(clip));
        self.wi -= &(grads.dwi.mapv(clip)); 
        self.bi -= &(grads.dbi.mapv(clip));
        self.wc -= &(grads.dwc.mapv(clip)); 
        self.bc -= &(grads.dbc.mapv(clip));
        self.wo -= &(grads.dwo.mapv(clip)); 
        self.bo -= &(grads.dbo.mapv(clip));
        self.wy -= &(grads.dwy.mapv(clip)); 
        self.by -= &(grads.dby.mapv(clip));
    }
}

/// Helper trait to simplify reshaping 1D arrays into 2D column/row matrices for dot product operations.
pub trait MatrixReshape {
    fn split_into_shapes(&self) -> (Array2<f64>, Array2<f64>);
}

impl MatrixReshape for Array1<f64> {
    fn split_into_shapes(&self) -> (Array2<f64>, Array2<f64>) {
        // Returns (Column Vector (N, 1), Row Vector (1, N))
        (self.to_owned().insert_axis(Axis(1)), self.to_owned().insert_axis(Axis(0)))
    }
}