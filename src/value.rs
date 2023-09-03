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

#[derive(Clone)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Char(char),
    Func {
        param: String,
        body: Rc<Expr>,
        closure: Env,
    },
    Fix,
    BuiltinFunc(Rc<dyn BuiltinFunc>),
    Data(String, Vec<Rc<Value>>),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int(l0), Self::Int(r0)) => l0 == r0,
            (Self::Bool(l0), Self::Bool(r0)) => l0 == r0,
            (Self::Char(l0), Self::Char(r0)) => l0 == r0,
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
            (Self::Fix, Self::Fix) => true,
            (Self::Data(l_name, l_args), Self::Data(r_name, r_args)) => {
                l_name == r_name && l_args == r_args
            }
            _ => false,
        }
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int(arg0) => f.debug_tuple("Int").field(arg0).finish(),
            Self::Bool(arg0) => f.debug_tuple("Bool").field(arg0).finish(),
            Self::Char(arg0) => f.debug_tuple("Char").field(arg0).finish(),
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
            Self::BuiltinFunc(_) => f.write_str("<built-in>"),
            Self::Fix => f.write_str("Fix"),
            Self::Data(name, args) => f.debug_tuple("Data").field(name).field(args).finish(),
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Char(c) => write!(f, "\'{}\'", c),
            Value::Func { param, body, .. } => write!(f, "(fun {} -> {})", param, body),
            Value::BuiltinFunc(_) => write!(f, "<builtin>"),
            Value::Fix => write!(f, "fix"),
            Value::Data(name, args) => {
                if name == "Cons" || name == "Nil" {
                    write_list(f, name, args)
                } else if name.starts_with("Tuple") {
                    write_tuple(f, args)
                } else if args.is_empty() {
                    write!(f, "{}", name)
                } else {
                    write!(
                        f,
                        "({} {})",
                        name,
                        args.iter()
                            .map(|x| x.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    )
                }
            }
        }
    }
}

fn extract_list(name: &String, args: &Vec<Rc<Value>>) -> Result<Vec<Rc<Value>>, String> {
    let mut name = name;
    let mut args = args;

    let mut res = vec![];

    loop {
        match name.as_str() {
            "Nil" => break,
            "Cons" => {
                res.push(args[0].clone());
                match args[1].borrow() {
                    Value::Data(name2, args2) => {
                        name = name2;
                        args = args2;
                    }
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
    args: &Vec<Rc<Value>>,
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
    args: &Vec<Rc<Value>>,
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

fn write_tuple(f: &mut std::fmt::Formatter, args: &Vec<Rc<Value>>) -> Result<(), std::fmt::Error> {
    write!(f, "({})", args.into_iter().join(", "))
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

    pub fn as_char(&self) -> Result<char, String> {
        match self {
            Value::Char(c) => Ok(*c),
            _ => Err(format!("Expected char, but got {}", self)),
        }
    }

    pub fn new_builtin<T: BuiltinFunc + 'static>(f: T) -> Self {
        Value::BuiltinFunc(Rc::new(f))
    }
}
