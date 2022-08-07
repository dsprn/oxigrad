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
    pub data: Rc<Cell<f32>>,
    pub grad: Rc<Cell<f32>>,
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

#[derive(Clone, Debug)]
pub struct Value {
    pub core: Rc<RefCell<Core>>,
}

pub trait ValueConstructors {
    fn construct(self) -> Value;
}

// constructor requiring fields: data
impl ValueConstructors for f32 {
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
impl ValueConstructors for (f32, f32) {
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
impl ValueConstructors for (f32, Option<Operation>) {
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
impl ValueConstructors for (f32, f32, Option<Operation>) {
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
impl ValueConstructors for (f32, Option<Operation>, Option<Vec<Value>>) {
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
impl ValueConstructors for (f32, f32, Option<Operation>, Option<Vec<Value>>) {
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
impl ValueConstructors for (f32, f32, Option<Operation>, Option<Vec<Value>>, Option<Box<dyn Fn() -> ()>>) {
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

impl Value {
    pub fn new<V>(args: V) -> Self 
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
                // for c in node.children.as_ref().unwrap().iter() {
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
        self.core.borrow().grad.set(1.0);

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

    pub fn power(&self, exp: f32) -> Self {
        let out = Value::new((
            self.core.borrow().data.get().powf(exp),
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
        let data = if self.core.borrow().data.get() >= 0.0 { self.core.borrow().data.get() } else { 0.0 };
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
            self.core.borrow().data.get() + other.core.borrow().data.get(),
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

impl ops::Mul<&Value> for &Value {
    type Output = Value;

    fn mul(self, other: &Value) -> Self::Output {
        let out = Value::new((
            self.core.borrow().data.get() * other.core.borrow().data.get(),
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
        &self * &other
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

impl ops::Div<&Value> for &Value {
    type Output = Value;

    fn div(self, other: &Value) -> Self::Output {
        // self * other.power(-1.0)
        self * &other.power(-1.0)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VALUE")
            .field("DATA", &self.core.borrow().data)
            .field("GRAD", &self.core.borrow().grad)
            .field("OP", &self.core.borrow().op)
            // .field("CHILDREN", &self.core.borrow().children) // not printing the vector as it could be pretty long, depending on the architecture of the network
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
        assert_eq!(c.core.borrow().data.get(), 3.0);
        assert_eq!(d.core.borrow().data.get(), 5.0);

        // testing derivative
        d.backward();
        assert_eq!(b.core.borrow().grad.get(), 2.0);
    }

    #[test]
    fn test_sub() {
        let ref a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let ref c = a - b;
        let ref d = c - b;

        // testing operation
        assert_eq!(c.core.borrow().data.get(), -1.0);
        assert_eq!(d.core.borrow().data.get(), -3.0);

        // testing derivative
        d.backward();
        assert_eq!(b.core.borrow().grad.get(), -2.0);
    }

    #[test]
    fn test_mul() {
        let ref a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let ref c = a + b;
        let d = c * b;

        // testing operation
        assert_eq!(d.core.borrow().data.get(), 6.0);

        // testing derivative
        d.backward();
        assert_eq!(b.core.borrow().grad.get(), 5.0);
    }

    #[test]
    fn test_mul_neg() {
        let ref a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let ref c = a - b;
        let d = c * b;

        // testing operation
        assert_eq!(d.core.borrow().data.get(), -2.0);

        // testing derivative
        d.backward();
        assert_eq!(b.core.borrow().grad.get(), -3.0);
    }

    #[test]
    fn test_power() {
        let ref a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let ref c = a + b;
        let d = c.power(2.0);

        // testing operation
        assert_eq!(d.core.borrow().data.get(), 9.0);

        // testing derivative
        d.backward();
        assert_eq!(b.core.borrow().grad.get(), 6.0);
    }

    #[test]
    fn test_relu() {
        let a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let c = a + &(b * &Value::new(2.0));
        let d = c.relu();
        let e = d * &Value::new(2.0);

        // testing operation
        assert_eq!(e.core.borrow().data.get(), 10.0);

        // testing derivative
        e.backward();
        assert_eq!(b.core.borrow().grad.get(), 4.0);
    }

    #[test]
    fn test_relu_neg() {
        let a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let c = a - (b * &Value::new(2.0));
        let d = c.relu();
        let e = d * &Value::new(2.0);

        // testing operation
        assert_eq!(e.core.borrow().data.get(), 0.0);

        // testing derivative
        e.backward();
        assert_eq!(b.core.borrow().grad.get(), 0.0);
    }

    #[test]
    fn test_div() {
        let ref a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let ref c = a + b;
        let d = c / b;

        // testing operation
        assert_eq!(d.core.borrow().data.get(), 1.5);

        // testing derivative
        d.backward();
        assert_eq!(b.core.borrow().grad.get(), -0.25);
    }
}
