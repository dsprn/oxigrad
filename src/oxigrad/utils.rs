use crate::oxigrad::engine::Value;
use crate::oxigrad::engine::Operation;

// dynamic learning rate function dependent on # of cycle iterations (from 0 to a maximum of 500 passes)
// mind that the hyperparameters chosen here could not work well for some NN architectures
pub fn alpha(pass: i32, iterations: i32) -> f64 {
    0.03 - 0.02 * pass as f64 / iterations as f64
}

pub fn mse(predicted: &Value, exp: f64) -> Value {
    let expected = Value::new((
        exp,
        Some(Operation::None),
    ));

    (predicted - &expected).power(2.0)
}

pub fn svm_maxmargin(predicted: &Value, exp: f64) -> Value {
    let expected = Value::new((
        exp,
        Some(Operation::None),
    ));

    (&(-expected * predicted) + 1.0).relu()
}

pub fn l2(model_params: &Vec<Value>, lambda: Option<&Value>) -> Value {
    let squared: Vec<Value> = model_params
        .iter()
        .map(|v| v.power(2.0))
        .collect();

    let l = lambda.unwrap_or(&Value::new(1e-4)).to_owned();

    let reg = l * &squared.iter()
        .fold(Value::new(0.0), |sum, el| sum+el);

    reg
}

// split data into equal sized groups
pub fn group(data: Vec<[f64; 2]>, labels: Vec<f64>, k: Option<usize>) -> (Vec<Vec<[f64; 2]>>, Vec<Vec<f64>>) {
    // if no size is given then keep the data undivided (i.e. with the whole length)
    let group_size = k.unwrap_or(data.len());
    let size = (data.len() / group_size) as usize;
    let mut data_groups = Vec::new();
    let mut labels_groups: Vec<Vec<f64>> = Vec::new();
    let mut start = 0;

    for _ in 0..group_size {
        data_groups.push(Vec::from(data[start..start+size].to_vec()));
        labels_groups.push(Vec::from_iter(labels[start..start+size].iter().cloned()));
        start += size;
    }

    (data_groups, labels_groups)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_mse() {
        let ref predicted = Value::new(1.111378);
        let expected = 2.314213;
        let rounded_mse = (mse(predicted, expected).core.borrow().data.get() * 1_000_000_f64).round() / 1_000_000_f64;

        assert_eq!(rounded_mse, 1.446812);
    }

    #[test]
    fn test_svm_maxmargin() {
        let predicted = &Value::new(1.111378);
        let mut expected = 2.314213;
        let mut rounded_svm = (svm_maxmargin(predicted, expected).get_data() * 1_000_000_f64).round() / 1_000_000_f64;
        assert_eq!(rounded_svm, 0.0);

        expected = 0.003;
        rounded_svm = (svm_maxmargin(predicted, expected).get_data() * 1_000_000_f64).round() / 1_000_000_f64;
        assert_eq!(rounded_svm, 0.996666);
    }

    #[test]
    fn test_alpha() {
        assert_eq!((alpha(314, 500) * 10_000_f64).round() / 10_000_f64, 0.4348);
    }

    #[test]
    fn test_groups() {
        // dummy data
        let dummy_dataset: [[f64; 2]; 10] = [
            [ 5.39412337e-01,  8.61363932e-01],
            [-1.03234535e+00,  5.77661126e-02],
            [-1.12251058e+00,  4.40911069e-01],
            [ 6.34512779e-01, -3.86770491e-01],
            [ 4.74812014e-01,  7.05693581e-01],
            [ 9.23972493e-01,  4.34679296e-01],
            [ 6.05938266e-01, -3.99049289e-01],
            [ 3.38158252e-01,  1.00461575e+00],
            [-9.65489273e-01,  1.44116250e-01],
            [ 1.73508562e+00, -3.03348212e-01]
        ];
        let dummy_labels: [f64; 10] = [-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0];

        // grouped data
        let (groups, labels) = group(dummy_dataset.to_vec(), dummy_labels.to_vec(), Some(5));
        
        // testing lenght and grouping on both
        assert_eq!(groups.len(), 5);
        assert_eq!(groups.iter()
            .zip([[[ 5.39412337e-01,  8.61363932e-01],
                   [-1.03234535e+00,  5.77661126e-02]],
                  [[-1.12251058e+00,  4.40911069e-01],
                   [ 6.34512779e-01, -3.86770491e-01]],
                  [[ 4.74812014e-01,  7.05693581e-01],
                   [ 9.23972493e-01,  4.34679296e-01]],
                  [[ 6.05938266e-01, -3.99049289e-01],
                   [ 3.38158252e-01,  1.00461575e+00]],
                  [[-9.65489273e-01,  1.44116250e-01],
                   [ 1.73508562e+00, -3.03348212e-01]]].to_vec().iter())
            .all(|(a, b)| a==b), true);

        assert_eq!(labels.len(), 5);
        assert_eq!(labels.iter()
            .zip([[-1.0, -1.0],
                  [-1.0, 1.0], 
                  [-1.0, -1.0], 
                  [1.0, -1.0],
                  [-1.0, 1.0]].to_vec().iter())
            .all(|(a, b)| a==b), true);
    }
}