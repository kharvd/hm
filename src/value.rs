use std::{
    borrow::Borrow,
    fmt::{Debug, Display},
    rc::Rc,
};

use itertools::Itertools;

use crate::{ast::Expr, env::Env};

pub trait BuiltinFunc {
    fn eval(&self, arg: Value) -> Result<Value, String>;
}

pub enum RefValue {
    Func {
        param: String,
        body: Rc<Expr>,
        closure: Env,
    },
    BuiltinFunc(Rc<dyn BuiltinFunc>),
    Data(String, Vec<Value>),
}

impl PartialEq for RefValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                RefValue::Func {
                    param: p1,
                    body: b1,
                    closure: c1,
                },
                RefValue::Func {
                    param: p2,
                    body: b2,
                    closure: c2,
                },
            ) => p1 == p2 && b1 == b2 && c1 == c2,
            (RefValue::BuiltinFunc(f1), RefValue::BuiltinFunc(f2)) => Rc::ptr_eq(f1, f2),
            (RefValue::Data(name1, args1), RefValue::Data(name2, args2)) => {
                name1 == name2 && args1 == args2
            }
            _ => false,
        }
    }
}

impl Debug for RefValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RefValue::Func {
                param,
                body,
                closure,
            } => write!(f, "Func({:?}, {:?}, {:?})", param, body, closure),
            RefValue::BuiltinFunc(_) => write!(f, "BuiltinFunc"),
            RefValue::Data(name, args) => write!(f, "{}({:?})", name, args),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Char(char),
    Fix,
    RefValue(Rc<RefValue>),
}

impl Value {}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Int(i1), Value::Int(i2)) => i1 == i2,
            (Value::Bool(b1), Value::Bool(b2)) => b1 == b2,
            (Value::Char(c1), Value::Char(c2)) => c1 == c2,
            (Value::Fix, Value::Fix) => true,
            (Value::RefValue(r1), Value::RefValue(r2)) => r1 == r2,
            _ => false,
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(i) => write!(f, "{}", i),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Char(c) => write!(f, "{}", c),
            Value::Fix => write!(f, "fix"),
            Value::RefValue(r) => match r.borrow() {
                RefValue::Func { .. } => write!(f, "<function>"),
                RefValue::BuiltinFunc(_) => write!(f, "<built-in>"),
                RefValue::Data(name, args) => match name.as_str() {
                    "Nil" | "Cons" => write_list(f, name, args),
                    s if s.starts_with("Tuple") => write_tuple(f, args),
                    _ if args.len() == 0 => write!(f, "{}", name),
                    _ => write!(
                        f,
                        "({} {})",
                        name,
                        args.iter()
                            .map(|x| x.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    ),
                },
            },
        }
    }
}

fn extract_list(name: &String, args: &Vec<Value>) -> Result<Vec<Value>, String> {
    let mut name = name;
    let mut args = args;

    let mut res = vec![];

    loop {
        match name.as_str() {
            "Nil" => break,
            "Cons" => {
                res.push(args[0].clone());
                match &args[1] {
                    Value::RefValue(ref_value) => match ref_value.borrow() {
                        RefValue::Data(name2, args2) => {
                            name = name2;
                            args = args2;
                        }
                        _ => Err(format!("Expected list, but got {}", args[1]))?,
                    },
                    _ => Err(format!("Expected list, but got {}", args[1]))?,
                }
            }
            _ => Err(format!("Expected list, but got {}", args[1]))?,
        }
    }

    Ok(res)
}

fn write_list(
    f: &mut std::fmt::Formatter,
    name: &String,
    args: &Vec<Value>,
) -> Result<(), std::fmt::Error> {
    let name = name;
    let args = args;

    if name == "Nil" {
        return write!(f, "[]");
    }

    if let Value::Char(_) = args[0].borrow() {
        return write_string(f, name, args);
    }

    let vals = extract_list(name, args).map_err(|_| std::fmt::Error)?;

    write!(f, "[{}]", vals.into_iter().join(", "))?;

    Ok(())
}

fn write_string(
    f: &mut std::fmt::Formatter,
    name: &String,
    args: &Vec<Value>,
) -> Result<(), std::fmt::Error> {
    let vals = extract_list(name, args).map_err(|_| std::fmt::Error)?;

    write!(
        f,
        "\"{}\"",
        vals.into_iter()
            .map(|v| {
                if let Value::Char(c) = v.borrow() {
                    c.clone()
                } else {
                    panic!("Expected char, but got {}", v);
                }
            })
            .collect::<String>()
    )?;

    Ok(())
}

fn write_tuple(f: &mut std::fmt::Formatter, args: &Vec<Value>) -> Result<(), std::fmt::Error> {
    write!(f, "({})", args.into_iter().join(", "))
}

impl Value {
    pub fn int(n: i64) -> Self {
        Value::Int(n)
    }

    pub fn bool(b: bool) -> Self {
        Value::Bool(b)
    }

    pub fn char(c: char) -> Self {
        Value::Char(c)
    }

    pub fn func(param: String, body: Rc<Expr>, closure: Env) -> Self {
        // println!("allocating func");
        Value::RefValue(Rc::new(RefValue::Func {
            param,
            body,
            closure,
        }))
    }

    pub fn builtin<T: BuiltinFunc + 'static>(f: T) -> Self {
        // println!("allocating builtin");
        Value::RefValue(Rc::new(RefValue::BuiltinFunc(Rc::new(f))))
    }

    pub fn data(name: String, args: Vec<Value>) -> Self {
        // println!("allocating data");
        Value::RefValue(Rc::new(RefValue::Data(name, args)))
    }

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

    pub fn as_char(&self) -> Result<char, String> {
        match self {
            Value::Char(c) => Ok(*c),
            _ => Err(format!("Expected char, but got {}", self)),
        }
    }

    pub fn as_ref_value(&self) -> Result<Rc<RefValue>, String> {
        match self {
            Value::RefValue(r) => Ok(r.clone()),
            _ => Err(format!("Expected RefValue, but got {}", self)),
        }
    }
}
