use std::rc::Rc;

use crate::value::{BuiltinFunc, Value};

trait BinOp {
    fn eval(&self, arg1: Value, arg2: Value) -> Result<Value, String>;
}

struct Partial {
    binop: Rc<dyn BinOp>,
    arg1: Value,
}

impl BuiltinFunc for Partial {
    fn eval(&self, arg2: Value) -> Result<Value, String> {
        self.binop.eval(self.arg1.clone(), arg2)
    }
}

impl<T: BinOp + Clone + 'static> BuiltinFunc for T {
    fn eval(&self, arg: Value) -> Result<Value, String> {
        Ok(Value::new_builtin(Partial {
            binop: Rc::new(self.clone()),
            arg1: arg,
        }))
    }
}

#[derive(Clone, Copy)]
pub enum IntUnaryOps {
    Neg,
}

impl BuiltinFunc for IntUnaryOps {
    fn eval(&self, arg: Value) -> Result<Value, String> {
        let arg = arg.as_int()?;
        Ok(match self {
            IntUnaryOps::Neg => Value::Int(-arg),
        })
    }
}

#[derive(Clone, Copy)]
pub enum IntBinOps {
    Plus,
    Minus,
    Mult,
    Div,
    Lt,
    Leq,
    Gt,
    Geq,
}

impl BinOp for IntBinOps {
    fn eval(&self, arg1: Value, arg2: Value) -> Result<Value, String> {
        let arg1 = arg1.as_int()?;
        let arg2 = arg2.as_int()?;
        Ok(match self {
            IntBinOps::Plus => Value::Int(arg1 + arg2),
            IntBinOps::Minus => Value::Int(arg1 - arg2),
            IntBinOps::Mult => Value::Int(arg1 * arg2),
            IntBinOps::Div => Value::Int(arg1 / arg2),
            IntBinOps::Lt => Value::Bool(arg1 < arg2),
            IntBinOps::Leq => Value::Bool(arg1 <= arg2),
            IntBinOps::Gt => Value::Bool(arg1 > arg2),
            IntBinOps::Geq => Value::Bool(arg1 >= arg2),
        })
    }
}

#[derive(Clone, Copy)]
pub enum BoolUnaryOps {
    Not,
}

impl BuiltinFunc for BoolUnaryOps {
    fn eval(&self, arg: Value) -> Result<Value, String> {
        let arg = arg.as_bool()?;
        Ok(match self {
            BoolUnaryOps::Not => Value::Bool(!arg),
        })
    }
}

#[derive(Clone, Copy)]
pub enum BoolBinOps {
    And,
    Or,
    Xor,
}

impl BinOp for BoolBinOps {
    fn eval(&self, arg1: Value, arg2: Value) -> Result<Value, String> {
        let arg1 = arg1.as_bool()?;
        let arg2 = arg2.as_bool()?;
        Ok(match self {
            BoolBinOps::And => Value::Bool(arg1 && arg2),
            BoolBinOps::Or => Value::Bool(arg1 || arg2),
            BoolBinOps::Xor => Value::Bool(arg1 ^ arg2),
        })
    }
}
