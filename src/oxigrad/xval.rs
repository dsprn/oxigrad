use std::collections::HashMap;
use std::fmt::Display;
use super::engine::Value;
use super::nn::Model;
use super::nn::Base;
use super::utils::{l2, group};


// RANGE IMPLEMENTATION WITH FLOATING VALUES
#[derive(Clone, Copy)]
pub(crate) struct FloatingRange {
    start: f64,
    end: f64,
    step: f64,
    current: f64,
}

impl FloatingRange {
    pub(crate) fn new(start: f64, end: f64, step: f64) -> Self {
        FloatingRange {
            start,
            end,
            step,
            current: start,
        }
    }
}

impl Iterator for FloatingRange {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current <= self.end {
            self.current = self.current + self.step;
            Some(self.current)
        } else {
            None
        }
    }
}


// CROSS VALIDATION IMPLEMENTATION AS struct
pub(crate) struct XVal<'a> {
    model: Option<Model>,
    model_arch: &'a Vec<usize>,
    k: usize,
    // alpha: f64,
    alpha: fn(i32, i32) -> f64,
    loss_fn: fn(&Value,f64) -> Value,
    values: Vec<Vec<[f64; 2]>>,
    labels: Vec<Vec<f64>>,
    hyper_range: FloatingRange,
    cv_scores: HashMap<String, Vec<f64>>,
}

impl<'a> XVal<'a> {
    pub(crate) fn new(
        data_ds: Vec<[f64; 2]>,
        labels_ds: Vec<f64>,
        model_arch: &'a Vec<usize>,
        hyper_range: FloatingRange,
        // alpha: f64,
        alpha: fn(i32, i32) -> f64,
        loss_fn: fn(&Value,f64) -> Value,
        k: usize,
    ) -> Self {
        let (values, labels) = group(data_ds, labels_ds, Some(k));
        
        XVal {
            model: None,
            model_arch,
            k: 10,
            alpha,
            loss_fn,
            values,
            labels,
            hyper_range,
            cv_scores: HashMap::new(),
        }
    }

    pub(crate) fn search_best_hyperpar(&mut self) -> f64 {
        println!("==> Using Cross Validation to look for the best L2 lambda hyperparameter in values ranging from {} to {}", 
            self.hyper_range.start, 
            self.hyper_range.end);
        
        let mut hyperpar = 0.0;

        for h in self.hyper_range {
            let mut scores: Vec<f64> = Vec::new();

            for ki in 0..self.k {
                // prepping data and holdouts
                let mut training_values = self.values.clone();
                let mut training_labels = self.labels.clone();
                let holdout_values = training_values.remove(ki);
                let holdout_labels = training_labels.remove(ki);

                // small training session (each time on a newly initialized model)
                self.mini_train(&training_values, &training_labels, Value::new(h));
                
                // holdout testing on the small training session to compute accuracy metric w.r.t current hyperpar
                let acc = self.holdout_test(&holdout_values, &holdout_labels);
                scores.push(acc);
            }

            let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
            println!("hyperpar={:.4}, accuracy={:.0}%", h, avg_score*100.0);

            // checking if score's already present in HashMap
            // if not add it with the respective value (i.e. the loss and the hypervalue)
            if !self.cv_scores.contains_key(&avg_score.to_string()) {
                self.cv_scores.insert(avg_score.to_string(), vec![h]);
            } else if let Some(v) = self.cv_scores.get_mut(&avg_score.to_string()) {
                v.push(h);
            }
        }

        // get the hyperpar associated with the highest accuracy (first of the list if there are more than 1)
        if let Some(best_score) = self.cv_scores.keys().max() {
            let hyperpars_list = self.cv_scores.get(best_score);

            match hyperpars_list {
                // get the first element from vector associated with best score
                // these elements are all the same as they all lead to the same score
                Some(v) => {
                    hyperpar = v[0];
                }
                // if the vector's empty return a default cross validation value
                None => {
                    hyperpar = 1e-4;
                }
            }
        }

        hyperpar
    }

    fn mini_train(&mut self, inputs: &Vec<Vec<[f64; 2]>>, expectations: &Vec<Vec<f64>>, hyperpar: Value) -> () {
        self.model = Some(Model::new(self.model_arch[0], self.model_arch));

        // train the new model on each of the training groups
        for (inps, exps) in inputs.iter().zip(expectations) {
            // train for 10 times on the same input group
            let iterations = 10;
            for pass in 0..iterations {
                // prepping for new forward pass
                self.model.as_ref().unwrap().zero_grad();

                // getting predictions and losses
                let preds: Vec<Value> = inps.iter()
                    // .map(|i| Model::forward(self.model.as_ref().unwrap(), i))
                    .map(|i| self.model.as_ref().unwrap().forward(i))
                    .collect();
                let losses: Vec<Value> = preds.iter()
                    .zip(exps)
                    .map(|(p, e)| (self.loss_fn)(p, *e))
                    .collect();
                let loss = losses.iter().sum::<Value>() / losses.len() as f64;

                // normalize loss with L2
                let reg = l2(&self.model.as_ref().unwrap().params(), Some(&hyperpar));
                let tot_loss = loss + &reg;
                
                // backward pass
                tot_loss.backward();
                for p in self.model.as_ref().unwrap().params().iter_mut() {
                    p.set_data(p.get_data() - ((self.alpha)(pass, iterations) * p.get_data()));
                }            
            }
        }
    }

    fn holdout_test(&self, inputs: &Vec<[f64; 2]>, expectations: &Vec<f64>) -> f64 {
        // computing prediction on holdout value
        let preds: Vec<Value> = inputs.iter()
            // .map(|x| Model::forward(self.model.as_ref().unwrap(), x))
            .map(|x| self.model.as_ref().unwrap().forward(x))
            .collect();
        
        // computing accuracy
        let directions = preds.iter()
            .zip(expectations)
            .map(|(p, e)| 
                if (p.get_data()>0.0) == (*e>0.0) {
                    1.0
                } else { 
                    0.0
                }
            ).collect::<Vec<f64>>();
        let acc: f64 = directions.iter().sum::<f64>() / directions.len() as f64;

        acc
    }
}

impl<'a> Display for XVal<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XVAL")
            .field("Model Arch", &self.model_arch)
            .field("Alpha", &self.alpha)
            .field("Values", &self.values)
            .field("Labels", &self.labels)
            // .field("CHILDREN", &self.core.borrow().children) // not printing this field as it could be pretty long, depending on the architecture of the network
            .finish()
    }
}