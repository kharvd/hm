use crate::{
    builtins::{BoolBinOps, BoolUnaryOps, IntBinOps, IntUnaryOps},
    env::Env,
    parser::parse_statement,
    value::Value,
};

impl Env {
    pub fn prelude() -> Env {
        let mut env = Env::new()
            .extend("plus", Value::new_builtin(IntBinOps::Plus))
            .extend("minus", Value::new_builtin(IntBinOps::Minus))
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
            .extend("xor", Value::new_builtin(BoolBinOps::Xor))
            .extend("fix", Value::Fix);

        let prelude_source = "
            val plus : int -> int -> int
            val minus : int -> int -> int
            val mult : int -> int -> int
            val div : int -> int -> int
            val lt : int -> int -> bool
            val leq : int -> int -> bool
            val gt : int -> int -> bool
            val geq : int -> int -> bool
            val neg : int -> int
            val not : bool -> bool
            val and : bool -> bool -> bool
            val or : bool -> bool -> bool
            val xor : bool -> bool -> bool

            val fix : ('a -> 'a) -> 'a
            let eq = fun x -> fun y -> and (leq x y) (geq x y)

            let rec fact = fun n -> if eq n 0 then 1 else mult n (fact (minus n 1))
            let rec fib = fun n -> if or (eq n 0) (eq n 1) then 1 else plus (fib (minus n 1)) (fib (minus n 2))
        ";

        for line in prelude_source.lines() {
            if line.trim().len() > 0 {
                let stmt_eval = env.eval_statement(&parse_statement(line).unwrap()).unwrap();
                env = stmt_eval.new_env
            }
        }

        env
    }
}
