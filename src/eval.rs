use std::rc::Rc;

use crate::{
    ast::{Expr, Statement, TypeExpr},
    env::Env,
    typing::infer,
    value::Value,
};

pub struct StatementEval {
    pub new_env: Env,
    pub var_name: String,
    pub var_type: Rc<TypeExpr>,
    pub value: Option<Value>,
}

impl Env {
    pub fn eval_statement(&self, stmt: &Statement) -> Result<StatementEval, String> {
        stmt.eval(self)
    }

    pub fn eval_expr(&self, expr: &Expr) -> Result<Value, String> {
        expr.eval(self)
    }
}

impl Statement {
    fn eval(&self, env: &Env) -> Result<StatementEval, String> {
        Ok(match self {
            Statement::Let(name, expr) => {
                let var_type = infer(env, &expr)?;
                let value = expr.eval(env)?;
                StatementEval {
                    new_env: env
                        .extend_type(&name, var_type.clone())
                        .extend(&name, value.clone()),
                    var_name: name.clone(),
                    var_type,
                    value: Some(value),
                }
            }
            Statement::LetRec(name, expr) => Statement::Let(
                name.clone(),
                Rc::new(Expr::Ap(
                    Rc::new(Expr::Ident("fix".to_string())),
                    Rc::new(Expr::Lambda(name.clone(), expr.clone())),
                )),
            )
            .eval(env)?,
            Statement::Val(name, type_expr) => StatementEval {
                new_env: env.extend_type(&name, type_expr.clone()),
                var_name: name.clone(),
                var_type: type_expr.clone(),
                value: None,
            },
        })
    }
}

impl Expr {
    fn eval(&self, env: &Env) -> Result<Value, String> {
        match self {
            Expr::Int(i) => Ok(Value::Int(*i)),
            Expr::Bool(b) => Ok(Value::Bool(*b)),
            Expr::Ident(name) => {
                let value = env.resolve_value(&name)?;
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
                    Value::Fix => {
                        let arg_eval = arg.eval(env)?;
                        match arg_eval {
                            Value::Func {
                                param,
                                body,
                                closure,
                            } => Ok(Value::RecFunc {
                                name: param.clone(),
                                body: body.clone(),
                                closure: closure.clone(),
                            }),
                            _ => Err(format!("Unexpected argument to fix: {}", arg)),
                        }
                    }
                    Value::RecFunc {
                        name,
                        body,
                        closure,
                    } => {
                        let body_env = env.extend(
                            &name,
                            Value::RecFunc {
                                name: name.clone(),
                                body: body.clone(),
                                closure: closure.clone(),
                            },
                        );
                        let ap_expr = Expr::Ap(body, arg.clone());
                        ap_expr.eval(&body_env)
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
        parser,
    };

    fn parse_expr(s: &str) -> Expr {
        match parser::parse(s).unwrap() {
            parser::ParseResult::Statement(_) => panic!("not an expression"),
            parser::ParseResult::Expression(expr) => expr,
        }
    }

    fn parse_statement(s: &str) -> Statement {
        parser::parse_statement(s).unwrap()
    }

    fn eval(s: &str) -> Value {
        eval_env(Env::new(), s)
    }

    fn eval_env(env: Env, s: &str) -> Value {
        env.eval_expr(&parse_expr(s)).unwrap()
    }

    fn eval_statements(env: Env, statements: Vec<&str>) -> Env {
        statements.iter().fold(env, |env, s| {
            env.eval_statement(&parse_statement(s)).unwrap().new_env
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
    fn eval_int() {
        let env = eval_statements(
            Env::prelude(),
            vec!["let x = 1", "let y = 42", "let z = plus x y"],
        );
        assert_eq!(eval_env(env.clone(), "z"), Value::Int(43))
    }

    #[test]
    fn eval_rec() {
        let env = eval_statements(Env::prelude(), vec![
            "let rec fact = fun n -> if eq n 0 then 1 else mult n (fact (minus n 1))",
            "let rec fib = fun n -> if or (eq n 0) (eq n 1) then 1 else plus (fib (minus n 1)) (fib (minus n 2))",
        ]);

        assert_eq!(eval_env(env.clone(), "fact 5"), Value::Int(120));
        assert_eq!(eval_env(env.clone(), "fib 8"), Value::Int(34));
    }
}
