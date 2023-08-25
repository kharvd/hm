use crate::{
    ast::{Expr, Statement},
    env::Env,
    value::Value,
};

impl Env {
    pub fn eval_statement(&self, stmt: Statement) -> Result<Self, String> {
        stmt.eval(self)
    }

    pub fn eval_expr(&self, expr: Expr) -> Result<Value, String> {
        expr.eval(self)
    }
}

impl Statement {
    fn eval(self, env: &Env) -> Result<Env, String> {
        match self {
            Statement::Let(name, expr) => {
                let value = expr.eval(env)?;
                Ok(env.extend(&name, value))
            }
            Statement::Val(_, _) => Ok(env.clone()),
        }
    }
}

impl Expr {
    fn eval(&self, env: &Env) -> Result<Value, String> {
        match self {
            Expr::Int(i) => Ok(Value::Int(*i)),
            Expr::Bool(b) => Ok(Value::Bool(*b)),
            Expr::Ident(name) => {
                let value = env.resolve(&name)?;
                Ok(value.clone())
            }
            Expr::If(cond, if_true, if_false) => {
                let cond = cond.eval(env)?;
                if cond.as_bool()? {
                    if_true.eval(env).clone()
                } else {
                    if_false.eval(env).clone()
                }
            }
            Expr::Lambda(param_name, body) => Ok(Value::Func {
                param: param_name.clone(),
                body: body.clone(),
                closure: env.clone(),
            }),
            Expr::Ap(func, arg) => {
                let func_eval = func.eval(env)?;
                match func_eval {
                    Value::Func {
                        param,
                        body,
                        closure,
                    } => {
                        let arg_eval = arg.eval(env)?;
                        let inner_env = closure.extend(&param, arg_eval);
                        body.eval(&inner_env)
                    }
                    Value::BuiltinFunc(f) => {
                        let arg_eval = arg.eval(env)?;
                        f.eval(arg_eval)
                    }
                    _ => Err(format!("Expected function, got {}", func_eval)),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use crate::{
        ast::{Expr, Statement},
        eval::{Env, Value},
    };

    fn parse_expr(s: &str) -> Expr {
        crate::parser::parse_expression(s).unwrap()
    }

    fn parse_statement(s: &str) -> Statement {
        crate::parser::parse_statement(s).unwrap()
    }

    fn eval(s: &str) -> Value {
        eval_env(Env::new(), s)
    }

    fn eval_env(env: Env, s: &str) -> Value {
        env.eval_expr(parse_expr(s)).unwrap()
    }

    fn eval_statements(env: Env, statements: Vec<&str>) -> Env {
        statements.iter().fold(env, |env, s| {
            env.eval_statement(parse_statement(s)).unwrap()
        })
    }

    #[test]
    fn eval_expr_simple() {
        assert_eq!(eval("123"), Value::Int(123));
        assert_eq!(eval("true"), Value::Bool(true));
        assert_eq!(
            eval("fun x -> x"),
            Value::Func {
                param: "x".to_string(),
                body: Rc::new(Expr::Ident("x".to_string())),
                closure: Env::new(),
            }
        );
    }

    #[test]
    fn eval_apply() {
        let env = eval_statements(Env::new(), vec!["let f = fun x -> if x then 5 else 10"]);
        assert_eq!(eval_env(env.clone(), "f true"), Value::Int(5));
        assert_eq!(eval_env(env.clone(), "f false"), Value::Int(10));
    }

    #[test]
    fn eval_pairs() {
        let env = eval_statements(
            Env::new(),
            vec![
                "let pair = fun x -> fun y -> fun f -> f x y",
                "let fst = fun p -> p (fun x -> fun y -> x)",
                "let snd = fun p -> p (fun x -> fun y -> y)",
                "let p = pair 123 false",
            ],
        );

        assert_eq!(eval_env(env.clone(), "fst p"), Value::Int(123));
        assert_eq!(eval_env(env.clone(), "snd p"), Value::Bool(false));
    }

    #[test]
    fn eval_int() {
        let env = eval_statements(
            Env::prelude(),
            vec!["let x = 1", "let y = 42", "let z = plus x y"],
        );
        assert_eq!(eval_env(env.clone(), "z"), Value::Int(43))
    }
}
