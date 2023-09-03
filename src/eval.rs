use std::{borrow::Borrow, rc::Rc};

use crate::{
    ast::{Expr, Statement, TypeExpr},
    env::Env,
    pattern::try_pattern_match,
    typing::infer,
    value::Value,
};

pub struct StatementEval {
    pub new_env: Env,
    pub statement: Statement,
}

impl Env {
    pub fn eval_statement(&self, stmt: &Statement) -> Result<StatementEval, String> {
        Ok(StatementEval {
            new_env: stmt.eval(self)?,
            statement: stmt.clone(),
        })
    }

    pub fn eval_expr(&self, expr: &Expr) -> Result<Value, String> {
        expr.eval(self)
    }

    pub fn eval_file(&self, file: &str) -> Result<Env, String> {
        let statements = crate::parser::parse_file(file)?;
        let mut env = self.clone();
        for statement in statements {
            env = env.eval_statement(&statement)?.new_env;
        }
        Ok(env)
    }
}

impl Statement {
    fn eval(&self, env: &Env) -> Result<Env, String> {
        Ok(match self {
            Statement::Let(name, expr) => {
                let var_type = infer(env, &expr)?;
                let value = expr.eval(env)?;
                env.extend_type(&name, var_type.clone())
                    .extend(&name, value.clone())
            }
            Statement::LetRec(name, expr) => {
                let binding = Expr::Ap(
                    Rc::new(Expr::Ident(String::from("fix"))),
                    Rc::new(Expr::Lambda(name.clone(), expr.clone())),
                );
                Statement::Let(name.clone(), Rc::new(binding)).eval(env)?
            }
            Statement::Val(name, type_expr) => {
                let generalized_type_expr = match type_expr.borrow() {
                    TypeExpr::Forall(vars, expr) => {
                        if expr.free_variables() != *vars {
                            return Err(format!(
                                "Type variables in forall do not match free variables: {}",
                                type_expr
                            ));
                        }

                        (**type_expr).clone()
                    }
                    _ => {
                        let free_vars = type_expr.free_variables();
                        TypeExpr::Forall(free_vars, type_expr.clone())
                    }
                };

                env.extend_type(&name, generalized_type_expr.clone())
            }
            Statement::Data(name, params, variants) => {
                let mut new_env = env.clone();

                for variant in variants {
                    let (constructor_name, constructor_type) =
                        constructor_for_variant(variant, name, params)?;
                    new_env = new_env
                        .extend(
                            &constructor_name,
                            Value::Data(constructor_name.clone(), vec![]),
                        )
                        .extend_type(&constructor_name, constructor_type);
                }

                new_env
            }
        })
    }
}

fn constructor_for_variant(
    variant: &TypeExpr,
    name: &String,
    params: &Vec<String>,
) -> Result<(String, TypeExpr), String> {
    let (constructor_name, constructor_type) = match variant {
        TypeExpr::Constructor(constructor_name, args) => {
            let mut constructor_type = TypeExpr::constructor(
                name,
                params.into_iter().map(|p| TypeExpr::type_var(p)).collect(),
            );
            for arg in args.iter().rev() {
                constructor_type = TypeExpr::Fun(arg.clone(), Rc::new(constructor_type));
            }
            constructor_type = TypeExpr::Forall(
                params.into_iter().map(|p| p.clone()).collect(),
                Rc::new(constructor_type),
            );
            constructor_type = constructor_type.normalize();

            Ok((constructor_name, constructor_type))
        }
        v => Err(format!("Variant should be a constructor, but got {}", v)),
    }?;
    Ok((constructor_name.clone(), constructor_type))
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
            Expr::Let(name, bound_expr, expr) => {
                let bound_value = bound_expr.eval(env)?;
                let inner_env = env.extend(&name, bound_value);
                expr.eval(&inner_env)
            }
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
                    Value::Data(name, args) => {
                        let arg_eval = arg.eval(env)?;
                        let mut new_args = args;
                        new_args.push(arg_eval);
                        Ok(Value::Data(name, new_args))
                    }
                    _ => Err(format!("Expected function, got {}", func_eval)),
                }
            }
            Expr::Match(expr, patterns) => {
                let eval_expr = expr.eval(env)?;
                for (choice_pattern, choice_expr) in patterns {
                    if let Some(bound_env) = try_pattern_match(env, &eval_expr, choice_pattern) {
                        let choice_expr_eval = choice_expr.eval(&bound_env)?;
                        return Ok(choice_expr_eval);
                    }
                }
                Err(format!("No match for {}", eval_expr))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use crate::{
        ast::Expr,
        e_ident,
        eval::{Env, Value},
        parser,
    };

    fn parse_expr(s: &str) -> Expr {
        match parser::parse(s).unwrap() {
            parser::ParseResult::Statement(_) => panic!("not an expression"),
            parser::ParseResult::Expression(expr) => expr,
        }
    }

    fn basic_env() -> Env {
        Env::prelude()
    }

    fn eval_file(file: &str) -> Env {
        basic_env().eval_file(file).unwrap()
    }

    fn eval_env(env: &Env, s: &str) -> Value {
        env.eval_expr(&parse_expr(s)).unwrap()
    }

    macro_rules! assert_same_value {
        ($env:expr, $left:expr, $right:expr) => {
            assert_eq!(eval_env(&$env, $left), eval_env(&$env, $right))
        };
    }

    #[test]
    fn eval_expr_simple() {
        let env = eval_file(
            "data Sign = Negative | Zero | Positive
            let a = 1",
        );
        assert_eq!(eval_env(&env, "123"), Value::Int(123));
        assert_eq!(eval_env(&env, "true"), Value::Bool(true));
        assert_eq!(
            eval_env(&env, "fun x -> x"),
            Value::Func {
                param: "x".to_string(),
                body: Rc::new(e_ident!("x")),
                closure: env.clone(),
            }
        );
        assert_eq!(eval_env(&env, "a"), Value::Int(1));
        assert_eq!(
            eval_env(&env, "Negative"),
            Value::Data("Negative".to_string(), vec![])
        );
    }

    #[test]
    fn eval_apply() {
        let env = eval_file("let f = fun x -> if x then 5 else 10");
        assert_same_value!(env, "f true", "5");
        assert_same_value!(env, "f false", "10");
    }

    #[test]
    fn eval_int() {
        let env = eval_file(
            "let x = 1
            let y = 42
            let z = x + y",
        );
        assert_same_value!(env, "z", "43");
    }

    #[test]
    fn eval_rec() {
        let env = eval_file(
            "let rec fact = fun n -> 
                match n with
                | 0 -> 1
                | _ -> n * fact (n - 1)

            let rec fib = fun n -> 
                match n with
                | 0 -> 1
                | 1 -> 1
                | _ -> fib (n - 1) + fib (n - 2)",
        );

        assert_same_value!(env, "fact 5", "120");
        assert_same_value!(env, "fib 8", "34");
    }

    #[test]
    fn eval_let_expr() {
        let env = eval_file(
            "let x = 1
            let y = let x = x + 1 in x",
        );
        assert_same_value!(env, "y", "2");
    }

    #[test]
    fn eval_data_simple() {
        let env = eval_file(
            "data Sign = Negative | Zero | Positive
            let f = fun x -> if x < 0 then Negative else if x > 0 then Positive else Zero",
        );

        assert_eq!(
            eval_env(&env, "f 5"),
            Value::Data("Positive".to_string(), vec![])
        );
        assert_eq!(
            eval_env(&env, "f 0"),
            Value::Data("Zero".to_string(), vec![])
        );
        assert_eq!(
            eval_env(&env, "f (neg 5)"),
            Value::Data("Negative".to_string(), vec![])
        );
    }

    #[test]
    fn eval_parameterized_data() {
        let env = eval_file(
            "data List 'a = Nil | Cons 'a (List 'a)
            let l = Cons 1 (Cons 2 Nil)",
        );

        assert_eq!(
            eval_env(&env, "l"),
            Value::Data(
                "Cons".to_string(),
                vec![
                    Value::Int(1),
                    Value::Data(
                        "Cons".to_string(),
                        vec![Value::Int(2), Value::Data("Nil".to_string(), vec![])]
                    )
                ]
            )
        );
    }

    #[test]
    fn match_constant() {
        let env = eval_file(
            "data Sign = Negative | Zero | Positive
            let f = fun x ->
                match x with
                | Negative -> 0
                | Zero -> 1
                | Positive -> 2",
        );

        assert_same_value!(env, "f Negative", "0");
        assert_same_value!(env, "f Zero", "1");
        assert_same_value!(env, "f Positive", "2");
    }

    #[test]
    fn match_pair() {
        let env = eval_file(
            "data Pair 'a 'b = Pair 'a 'b
            let f = fun x ->
                match x with
                | Pair a b -> a + b",
        );

        assert_same_value!(env, "f (Pair 1 2)", "3");
    }

    #[test]
    fn match_list() {
        let env = eval_file(
            "data List 'a = Nil | Cons 'a (List 'a)
            let rec len = fun l ->
                match l with
                | Nil -> 0
                | Cons _ xs -> 1 + (len xs)",
        );

        assert_same_value!(env, "len (Cons 1 (Cons 2 (Cons 3 Nil)))", "3");
        assert_same_value!(env, "len Nil", "0");
    }
}
