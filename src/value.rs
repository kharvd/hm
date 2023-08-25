use std::{
    fmt::{Debug, Display},
    rc::Rc,
};

use crate::{ast::Expr, env::Env};

pub trait BuiltinFunc {
    fn eval(&self, arg: Value) -> Result<Value, String>;
}

#[derive(Clone)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Func {
        param: String,
        body: Rc<Expr>,
        closure: Env,
    },
    BuiltinFunc(Rc<dyn BuiltinFunc>),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int(l0), Self::Int(r0)) => l0 == r0,
            (Self::Bool(l0), Self::Bool(r0)) => l0 == r0,
            (
                Self::Func {
                    param: l_param,
                    body: l_body,
                    closure: l_closure,
                },
                Self::Func {
                    param: r_param,
                    body: r_body,
                    closure: r_closure,
                },
            ) => l_param == r_param && l_body == r_body && l_closure == r_closure,
            (Self::BuiltinFunc(_), Self::BuiltinFunc(_)) => false,
            _ => false,
        }
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int(arg0) => f.debug_tuple("Int").field(arg0).finish(),
            Self::Bool(arg0) => f.debug_tuple("Bool").field(arg0).finish(),
            Self::Func {
                param,
                body,
                closure,
            } => f
                .debug_struct("Func")
                .field("param", param)
                .field("body", body)
                .field("closure", closure)
                .finish(),
            Self::BuiltinFunc(_) => f.write_str("<builtin>"),
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Func { param, body, .. } => write!(f, "(fun {} -> {})", param, body),
            Value::BuiltinFunc(_) => write!(f, "<builtin>"),
        }
    }
}

impl Value {
    pub fn as_int(&self) -> Result<i64, String> {
        match self {
            Value::Int(i) => Ok(*i),
            _ => Err(format!("Expected int, but got {}", self)),
        }
    }

    pub fn as_bool(&self) -> Result<bool, String> {
        match self {
            Value::Bool(b) => Ok(*b),
            _ => Err(format!("Expected bool, but got {}", self)),
        }
    }

    pub fn new_builtin<T: BuiltinFunc + 'static>(f: T) -> Self {
        Value::BuiltinFunc(Rc::new(f))
    }
}