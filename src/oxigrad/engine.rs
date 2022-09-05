use std::ops;
use std::rc::Rc;
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::fmt::{Debug, Display};

#[derive(Debug, Clone, Copy)]
pub enum Operation {
    Addition,
    Subtraction,
    Multiplication,
    Division,
    Power,
    ReLU,
    None,
}

pub struct Core {
    pub data: Rc<Cell<f64>>,
    pub grad: Rc<Cell<f64>>,
    op: Option<Operation>,
    pub children: Option<Vec<Value>>,
    backward: Option<Box<dyn Fn() -> ()>>,
}

impl Debug for Core {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CORE")
            .field("DATA", &self.data)
            .field("GRAD", &self.grad)
            .field("CHILDREN", &self.children)
            .finish()
    }
}

pub trait ValueConstructors {
    fn construct(self) -> Value;
}

// constructor requiring fields: data
impl ValueConstructors for f64 {
    fn construct(self) -> Value {
        Value {
            core: Rc::new(RefCell::new(Core {
                data: Rc::new(Cell::new(self)),
                grad: Rc::new(Cell::new(0.0)),
                op: None,
                children: None,
                backward: None,
            }))
        }
    }
}

// constructor requiring fields: data, grad
impl ValueConstructors for (f64, f64) {
    fn construct(self) -> Value {
        Value {
            core: Rc::new(RefCell::new(Core {
                data: Rc::new(Cell::new(self.0)),
                grad: Rc::new(Cell::new(self.1)),
                op: None,
                children: None,
                backward: None,
            }))
        }
    }
}

// constructor requiring fields: data, op
impl ValueConstructors for (f64, Option<Operation>) {
    fn construct(self) -> Value {
        Value {
            core: Rc::new(RefCell::new(Core {
                data: Rc::new(Cell::new(self.0)),
                grad: Rc::new(Cell::new(0.0)),
                op: self.1,
                children: None,
                backward: None,
            }))
        }
    }
}

// constructor requiring fields: data, grad, op
impl ValueConstructors for (f64, f64, Option<Operation>) {
    fn construct(self) -> Value {
        Value {
            core: Rc::new(RefCell::new(Core {
                data: Rc::new(Cell::new(self.0)),
                grad: Rc::new(Cell::new(self.1)),
                op: self.2,
                children: None,
                backward: None,
            }))
        }
    }
}

// constructor requiring fields: data, op, children
impl ValueConstructors for (f64, Option<Operation>, Option<Vec<Value>>) {
    fn construct(self) -> Value {
        Value {
            core: Rc::new(RefCell::new(Core {
                data: Rc::new(Cell::new(self.0)),
                grad: Rc::new(Cell::new(0.0)),
                op: self.1,
                children: self.2,
                backward: None,
            }))
        }
    }
}

// constructor requiring fields: data, grad, op, children
impl ValueConstructors for (f64, f64, Option<Operation>, Option<Vec<Value>>) {
    fn construct(self) -> Value {
        Value {
            core: Rc::new(RefCell::new(Core {
                data: Rc::new(Cell::new(self.0)),
                grad: Rc::new(Cell::new(self.1)),
                op: self.2,
                children: self.3,
                backward: None,
            }))
        }
    }
}

// constructor requiring fields: data, grad, op, children, backward
impl ValueConstructors for (f64, f64, Option<Operation>, Option<Vec<Value>>, Option<Box<dyn Fn() -> ()>>) {
    fn construct(self) -> Value {
        Value {
            core: Rc::new(RefCell::new(Core {
                data: Rc::new(Cell::new(self.0)),
                grad: Rc::new(Cell::new(self.1)),
                op: self.2,
                children: self.3,
                backward: self.4,
            }))
        }
    }
}

#[derive(Clone, Debug)]
pub struct Value {
    pub core: Rc<RefCell<Core>>,
}

impl Value {
    pub fn new<V>(args: V) -> Value 
        where V: ValueConstructors
    {
        args.construct()
    }

    pub fn backward(&self) {
        let mut tp_order: Vec<Value> = vec![];
        let mut visited = HashSet::new();

        fn topological_sort(node: &Value, visited: &mut HashSet<Value>, tp_order: &mut Vec<Value>) {
            if !visited.contains(&node) {
                visited.insert(node.clone());

                match node.core.borrow().children.as_ref() {
                    Some(v) => {
                        for c in v.iter() {
                            topological_sort(c, visited, tp_order);
                        }
                        tp_order.push(node.clone());
                    },
                    None => {}
                }
            }
        }

        // topological sort of graph's nodes
        topological_sort(self, &mut visited, &mut tp_order);
        
        // a derivative of something (i.e. the starting node for the backward pass) w.r.t itself is 1
        self.set_grad(1.0);

        // backward pass on reversed topological order
        for v in tp_order.iter().rev() {
            match v.core.borrow_mut().backward.as_mut() {
                Some(back) => back(),
                None => {
                    panic!("No backward closure to call for this node");
                }
            }
        }
    }

    pub fn power(&self, exp: f64) -> Self {
        let out = Value::new((
            self.get_data().powf(exp),
            Some(Operation::Power),
            Some(vec![self.clone()]),
        ));

        let  s_grad = self.core.borrow().grad.clone();
        let out_grad = out.core.borrow().grad.clone();
        let s_data = self.core.borrow().data.clone();

        // derivative for raise to the power operation
        let back = Box::new(move || {
            s_grad.set(s_grad.get() + (exp * (s_data.get().powf(exp - 1.0)) * out_grad.get()));
        });
        out.core.borrow_mut().backward = Some(back);

        out
    }

    pub fn relu(&self) -> Self {
        let data = if self.get_data() >= 0.0 { self.get_data() } else { 0.0 };
        let out = Value::new((
            data,
            Some(Operation::ReLU),
            Some(vec![self.clone()]),
        ));

        let s_grad = self.core.borrow().grad.clone();
        let out_grad = out.core.borrow().grad.clone();
        let s_data = self.core.borrow().data.clone();

        // derivative for ReLU operation
        let back = Box::new(move || {
            s_grad.set(s_grad.get() + (if s_data.get() < 0.0 { 0.0 } else { 1.0 * out_grad.get() }));
        });
        out.core.borrow_mut().backward = Some(back);

        out
    }

    pub fn get_data(&self) -> f64 {
        self.core.borrow().data.get()
    }

    pub fn set_data(&self, val: f64) -> () {
        self.core.borrow().data.set(val);
    }

    pub fn get_grad(&self) -> f64 {
        self.core.borrow().grad.get()
    }

    pub fn set_grad(&self, val: f64) -> () {
        self.core.borrow().grad.set(val);
    }

}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.core.as_ptr() == other.core.as_ptr()
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.core.as_ptr().hash(state);
    }
}

impl ops::Add<&Value> for &Value {
    type Output = Value;

    fn add(self, other: &Value) -> Self::Output {
        let out = Value::new((
            self.get_data() + other.get_data(),
            Some(Operation::Addition),
            Some(vec![self.clone(), other.clone()]),
        ));

        let s_grad = self.core.borrow().grad.clone();
        let oth_grad = other.core.borrow().grad.clone();
        let out_grad = out.core.borrow().grad.clone();

        // derivative for add operation
        let back = Box::new(move || {
            s_grad.set(s_grad.get() + out_grad.get());
            oth_grad.set(oth_grad.get() + out_grad.get());
        }) as Box<dyn Fn() -> ()>;
        out.core.borrow_mut().backward = Some(back);

        out
    }
}

impl ops::Add<&Value> for Value {
    type Output = Value;

    fn add(self, other: &Value) -> Self::Output {
        &self + other
    }
}

impl ops::Add<Value> for &Value {
    type Output = Value;

    fn add(self, other: Value) -> Self::Output {
        self + &other
    }
}

impl ops::Add<f64> for &Value {
    type Output = Value;

    fn add(self, other: f64) -> Self::Output {
        self + Value::new(other)
    }
}

// FIXME: should not disrupt computational graph as the returned Value'll be the basis of backprop
impl<'a> std::iter::Sum<&'a Value> for Value {
    fn sum<I: Iterator<Item = &'a Value>>(iter: I) -> Self {
        iter.fold(
            Value::new(0.0), 
            |sum, el| sum + el,
        )
    }
}

impl ops::Mul<&Value> for &Value {
    type Output = Value;

    fn mul(self, other: &Value) -> Self::Output {
        let out = Value::new((
            self.get_data() * other.get_data(),
            Some(Operation::Multiplication),
            Some(vec![self.clone(), other.clone()]),
        ));

        let s_grad = self.core.borrow().grad.clone();
        let oth_grad = other.core.borrow().grad.clone();
        let out_grad = out.core.borrow().grad.clone();

        let s_data = self.core.borrow().data.clone();
        let oth_data = other.core.borrow().data.clone();

        // derivative for mul operation
        let back = Box::new(move || {
            s_grad.set(s_grad.get() + (oth_data.get() * out_grad.get()));
            oth_grad.set(oth_grad.get() + (s_data.get() * out_grad.get()));
        }) as Box<dyn Fn() -> ()>;
        out.core.borrow_mut().backward = Some(back);

        out
    }
}

impl ops::Mul<&Value> for Value {
    type Output = Value;

    fn mul(self, other: &Value) -> Self::Output {
        &self * other
    }
}

impl ops::Mul<Value> for &Value {
    type Output = Value;

    fn mul(self, other: Value) -> Self::Output {
        self * &other
    }
}

impl ops::Neg for &Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        self * &Value::new(-1.0)
    }
}

impl ops::Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        self * &Value::new(-1.0)
    }
}

impl ops::Sub<&Value> for &Value {
    type Output = Value;

    fn sub(self, other: &Value) -> Self::Output {
        self + &(-other)
    }
}

impl ops::Sub<Value> for Value {
    type Output = Value;

    fn sub(self, other: Value) -> Self::Output {
        self + &(-other)
    }
}

impl ops::Div<f64> for Value {
    type Output = Value;

    fn div(self, other: f64) -> Self::Output {
        self * &Value::new(1.0/other)
    }
}

impl ops::Div<&Value> for &Value {
    type Output = Value;

    fn div(self, other: &Value) -> Self::Output {
        self * &other.power(-1.0)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VALUE")
            .field("DATA", &self.get_data())
            .field("GRAD", &self.get_grad())
            .field("OP", &self.core.borrow().op)
            .finish()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_add() {
        let ref a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let ref c = a + b;
        let ref d = c + b;

        // testing operation
        assert_eq!(c.get_data(), 3.0);
        assert_eq!(d.get_data(), 5.0);

        // testing derivative
        d.backward();
        assert_eq!(b.get_grad(), 2.0);
    }

    #[test]
    fn test_sub() {
        let ref a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let ref c = a - b;
        let ref d = c - b;

        // testing operation
        assert_eq!(c.get_data(), -1.0);
        assert_eq!(d.get_data(), -3.0);

        // testing derivative
        d.backward();
        assert_eq!(b.get_grad(), -2.0);
    }

    #[test]
    fn test_mul() {
        let ref a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let ref c = a + b;
        let d = c * b;

        // testing operation
        assert_eq!(d.get_data(), 6.0);

        // testing derivative
        d.backward();
        assert_eq!(b.get_grad(), 5.0);
    }

    #[test]
    fn test_mul_neg() {
        let ref a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let ref c = a - b;
        let d = c * b;

        // testing operation
        assert_eq!(d.get_data(), -2.0);

        // testing derivative
        d.backward();
        assert_eq!(b.get_grad(), -3.0);
    }

    #[test]
    fn test_power() {
        let ref a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let ref c = a + b;
        let d = c.power(2.0);

        // testing operation
        assert_eq!(d.get_data(), 9.0);

        // testing derivative
        d.backward();
        assert_eq!(b.get_grad(), 6.0);
    }

    #[test]
    fn test_relu() {
        let a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let c = a + &(b * &Value::new(2.0));
        let d = c.relu();
        let e = d * &Value::new(2.0);

        // testing operation
        assert_eq!(e.get_data(), 10.0);

        // testing derivative
        e.backward();
        assert_eq!(b.get_grad(), 4.0);
    }

    #[test]
    fn test_relu_neg() {
        let a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let c = a - (b * &Value::new(2.0));
        let d = c.relu();
        let e = d * &Value::new(2.0);

        // testing operation
        assert_eq!(e.get_data(), 0.0);

        // testing derivative
        e.backward();
        assert_eq!(b.get_grad(), 0.0);
    }

    #[test]
    fn test_div() {
        let ref a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let ref c = a + b;
        let d = c / b;

        // testing operation
        assert_eq!(d.get_data(), 1.5);

        // testing derivative
        d.backward();
        assert_eq!(b.get_grad(), -0.25);
    }

    #[test]
    fn test_constructors() {
        let v1 = Value::new((
            3.141592,
            Some(Operation::None),
            None,
        ));
        assert_eq!(v1.get_data(), 3.141592);
        assert_eq!(v1.get_grad(), 0.0);

        let v2 = Value::new((
            3.141592 + 5.0,
            Some(Operation::None),
            None,
        ));
        assert_eq!(v2.get_data(), 8.141592);
        assert_eq!(v2.get_grad(), 0.0);

        let v3 = Value::new((
            (v1.get_data() * v2.get_data() * 1_000_f64).round() / 1_000_f64,
            Some(Operation::None),
            None,
        ));
        assert_eq!(v3.get_data(), 25.578);
        assert_eq!(v3.get_grad(), 0.0);
    }

    #[test]
    fn test_get_set() {
        let v = Value::new((
            3.14159265,
            0.99999999,
        ));
        assert_eq!(v.get_data(), 3.14159265);
        assert_eq!(v.get_grad(), 0.99999999);

        v.set_data(1.2345);
        v.set_grad(6.7890);
        assert_eq!(v.get_data(), 1.2345);
        assert_eq!(v.get_grad(), 6.7890);
    }
}
