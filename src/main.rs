// TODO: implement model export to some kind of file
mod oxigrad;

use oxigrad::nn::Model;
use oxigrad::utils::{mse, svm_maxmargin, l2, alpha};
use oxigrad::data::{INP_DATASET, LBLS_DATASET};
use rand::Rng;
use crate::oxigrad::nn::Base;
use crate::oxigrad::engine::Value;
use crate::oxigrad::xval::{XVal, FloatingRange};

fn main() {
    // WATCH OUT, changing the following hyperparameter (i.e. the NN architecture)
    // could require to change other hyperparameters as well like the alpha
    // and, in general, to do some tuning before training the resulting NN
    let arch = vec![5, 5, 1];
    let m = Model::new(2, &arch);

    // cross validation to find best L2 lambda hyperparameter
    // data generated with scikit-learn's make_moon method (n_samples=100, noise=0.1)
    let mut xv = XVal::new(
        INP_DATASET.to_vec(),
        LBLS_DATASET.to_vec(), 
        &arch, 
        FloatingRange::new(0.0, 0.01, 0.0005), 
        alpha, 
        mse,
        10,
    );
    let l2_lambda = xv.search_best_hyperpar();
    println!("==> L2 lambda value={:.4}", l2_lambda);

    println!("\n==> Choosing inputs and relative label from a preloaded dataset...");
    let data_index = rand::thread_rng().gen_range(0, 100);
    let inputs = INP_DATASET[data_index];
    let label = LBLS_DATASET[data_index];
    println!("==> Input values={:?}", inputs);
    println!("==> Expected value={}", label);

    println!("\n==> Start training the model...");
    let iterations = 50;
    for pass in 0..iterations {
        // prepping
        m.zero_grad();

        // forward pass
        let preds = m.forward(&inputs);
        let loss = mse(&preds, label);
        // let loss = svm_maxmargin(&preds, label);

        // L2 regularization
        let reg = l2(&m.params(), Some(&Value::new(l2_lambda)));
        let tot_loss = &loss + &reg;

        // backward pass
        tot_loss.backward(); 
        for p in m.params().iter() {
            p.core.borrow().data.set(p.get_data() - (alpha(pass, iterations) * p.get_grad()));
        }

        println!(
            "pass={}, alpha={:.16}, prediction={:.16}, reg={:.16}, loss={:.16}, tot_loss={:.16}", 
            pass,
            alpha(pass, iterations),
            preds.get_data(),
            reg.get_data(),
            loss.get_data(),
            tot_loss.get_data(),
        );
    }
    println!("==> DONE");
}
