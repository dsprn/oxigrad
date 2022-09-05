use crate::oxigrad::engine::Value;
use rand::{thread_rng, Rng};

// BASE TRAIT
pub trait Base {
    fn zero_grad(&self) {
        for p in self.params().iter_mut() {
            p.core.borrow().grad.set(0.0);
        }
    }

    fn params(&self) -> Vec<Value>;
}

// NEURON IMPLEMENTATION
struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    nonlin: bool,
}

impl Base for Neuron {
    fn params(&self) -> Vec<Value> {
        let mut ps = vec![];

        for w in self.weights.clone() {
            ps.push(w.clone());
        }
        ps.push(self.bias.clone());

        ps
    }
}

impl Neuron {
    fn new(num_weights: usize, nonlin: bool) -> Self {
        Neuron {
            weights: (0..num_weights)
                .map(|_| thread_rng().gen_range::<f64>(-1.0, 1.0))
                .map(|v| Value::new(v))
                .collect(),
            bias: Value::new(0.0),
            nonlin,
        }
    }

    fn forward(&self, inputs: &Vec<Value>) -> Value {
        let mut dot = inputs.iter()
            .zip(self.weights.iter())
            .fold(
                Value::new(0.0),
                |mut s, (x, w)| { s = s + &(x * w); s },
            );
        dot = dot + &self.bias;

        if self.nonlin {
            dot.relu()
        } else {
            dot
        }
    }
}

// LAYER IMPLEMENTATION
struct Layer {
    neurons: Vec<Neuron>,
}

impl Base for Layer {
    fn params(&self) -> Vec<Value> {
        let mut ps = vec![];

        for n in self.neurons.iter() {
            for p in n.params() {
                ps.push(p);
            }
        }

        ps
    }
}

impl Layer {
    fn new(num_weights: usize, neurons: usize, nonlin: bool) -> Self {
        let mut l = Layer {
            neurons: Vec::<Neuron>::new(),
        };

        for _n in 0..neurons {
            l.neurons.push(Neuron::new(num_weights, nonlin));
        }

        l
    }

    fn forward(&self, inputs: Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(&inputs)).collect()
    }
}

// MODEL IMPLEMENTATION
pub struct Model {
    layers: Vec<Layer>,
}

impl Base for Model {
    fn params(&self) -> Vec<Value> {
        let mut ps = vec![];

        for l in self.layers.iter() {
            for p in l.params() {
                ps.push(p);
            }
        }

        ps
    }
}

impl Model {
    pub fn new(input_size: usize, arch: &Vec<usize>) -> Self {
        // initialize NN architecture
        let mut nn_arch = Vec::new();
        nn_arch.push(input_size);
        nn_arch.extend(arch);

        // initialize model
        let mut m = Model {
            layers: Vec::<Layer>::new(),
        };

        // initialize model's layers
        for l in 0..arch.len() {
            m.layers.push(Layer::new(
                nn_arch[l],
                nn_arch[l+1],
                l!=arch.len()-1,
            ))
        }

        m
    }

    // pub fn forward(&self, inputs: &[f64; 2]) -> Vec<Value> {
    pub fn forward(&self, inputs: &[f64; 2]) -> Value {
        // multiply inputs for each layers and collect results
        let mut is: Vec<Value> = inputs
            .iter()
            .map(|v| Value::new(*v))
            .collect();

        for l in &self.layers {
            is = l.forward(is);
        }

        is[0].clone()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn grad_sum(params: Vec<Value>) -> f64 {
        let grads = params.iter()
            .fold(0.0, |mut s, v| { s += v.core.borrow().grad.get(); s });
        
        grads
    }

    #[test]
    fn test_neuron() {
        let n = Neuron::new(10, true);

        assert!(n.weights.len() == 10);
        assert!(n.bias.core.borrow().data.get() == 0.0);

        n.zero_grad();
        assert!(grad_sum(n.params()) == 0.0);
    }

    #[test]
    fn test_layer() {
        let l = Layer::new(8, 2, false);

        assert!(l.neurons.len() == 2);
        assert!(l.neurons.first().unwrap().bias.core.borrow().data.get() == 0.0);

        l.zero_grad();
        assert!(grad_sum(l.params()) == 0.0);
    }

    #[test]
    fn test_model() {
        let m = Model::new(8, &vec![4, 2]);

        assert!(m.layers.first().unwrap().neurons.len() == 4);
        assert!(m.layers.last().unwrap().neurons.len() == 2);

        m.zero_grad();
        assert!(grad_sum(m.params()) == 0.0);
    }
}
