use crate::oxigrad::engine::Value;
use crate::oxigrad::engine::Operation;

pub fn mse(predicted: &Value, exp: f32) -> Value {
    let expected = Value::new((
        exp,
        Some(Operation::None),
    ));

    (predicted - &expected).power(2.0)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_mse() {
        let ref predicted = Value::new(1.1113786);
        let expected = 2.314;

        assert_eq!(mse(predicted, expected).core.borrow().data.get(), 1.4462981);
    }
}
