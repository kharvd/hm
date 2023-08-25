use crate::{
    builtins::{BoolBinOps, BoolUnaryOps, IntBinOps, IntUnaryOps},
    env::Env,
    value::Value,
};

impl Env {
    pub fn prelude() -> Env {
        let env = Env::new()
            .extend("plus", Value::new_builtin(IntBinOps::Plus))
            .extend("minus", Value::new_builtin(IntBinOps::Plus))
            .extend("mult", Value::new_builtin(IntBinOps::Mult))
            .extend("div", Value::new_builtin(IntBinOps::Div))
            .extend("lt", Value::new_builtin(IntBinOps::Lt))
            .extend("leq", Value::new_builtin(IntBinOps::Leq))
            .extend("gt", Value::new_builtin(IntBinOps::Gt))
            .extend("geq", Value::new_builtin(IntBinOps::Geq))
            .extend("neg", Value::new_builtin(IntUnaryOps::Neg))
            .extend("not", Value::new_builtin(BoolUnaryOps::Not))
            .extend("and", Value::new_builtin(BoolBinOps::And))
            .extend("or", Value::new_builtin(BoolBinOps::Or))
            .extend("xor", Value::new_builtin(BoolBinOps::Xor));

        env
    }
}
