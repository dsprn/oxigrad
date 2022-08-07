// TODO: implement new loss functions
// TODO: implement L2 regularization
// TODO: implement cross validation
// TODO: implement model export to some kind of file

mod oxigrad;

use oxigrad::nn::Model;
use oxigrad::utils::mse;
use crate::oxigrad::nn::Base;


fn main() {
    let m = Model::new(2, &[2, 2, 1]);
    let expected = 2.314;
    let alpha = 0.03;
    let acceptable_loss = 0.000005 as f32;

    // forward pass + loss
    let mut pred;
    let mut loss_value;
    let mut loss = f32::MAX;

    let mut epoch = 0;
    println!("==> Expected value={}", expected);
    println!("==> Start training the model...");
    while loss > acceptable_loss {
        epoch += 1;
        m.zero_grad();

        // forward pass
        pred = m.forward(&[0.342, -1.876]);
        loss_value = mse(&pred[0], expected);
        loss = loss_value.core.borrow().data.get();

        // backward pass
        loss_value.backward(); 
        // update model params
        for p in m.params().iter() {
            p.core.borrow().data.set(p.core.borrow().data.get() - (alpha * p.core.borrow().grad.get()));
        }

        // print results
        println!(
            "epoch:{}, prediction:{}, loss:{}", 
            epoch,
            pred[0].core.borrow().data.get(),
            loss_value.core.borrow().data.get(),
        );
    }
    println!("==> DONE");
}
