use std::{borrow::Borrow, cell::RefCell, rc::Rc};

use crate::{
    ast::{Expr, Statement, TypeExpr},
    env::Env,
    pattern::try_pattern_match,
    typing::infer,
    value::{RefValue, Value},
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

    pub fn eval_expr(&self, expr: Expr) -> Result<Value, String> {
        eval_expr(Rc::new(expr), &self)
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
                let value = eval_expr(expr.clone(), env)?;
                env.extend_type(&name, var_type.clone())
                    .extend_value(&name, value)
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
                        .extend_value(
                            &constructor_name,
                            Value::data(constructor_name.clone(), vec![]),
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

fn eval_expr(expr: Rc<Expr>, env: &Env) -> Result<Value, String> {
    let mut expr = expr;
    let mut env = env.clone();

    'outer: loop {
        match expr.borrow() {
            Expr::Int(i) => return Ok(Value::int(*i)),
            Expr::Bool(b) => return Ok(Value::bool(*b)),
            Expr::Char(c) => return Ok(Value::char(*c)),
            Expr::Ident(name) => {
                let value = env.resolve_value(&name)?;
                let Some(value) = value else {
                    return Err(format!("Infinite loop: {}", name));
                };
                return Ok(value);
            }
            Expr::If(cond, if_true, if_false) => {
                let cond = eval_expr(cond.clone(), &env)?;

                // tail call optimization
                if cond.as_bool()? {
                    expr = if_true.clone();
                } else {
                    expr = if_false.clone();
                }
                continue 'outer;
            }
            Expr::Lambda(param_name, body) => {
                return Ok(Value::func(param_name.clone(), body.clone(), env.clone()))
            }
            Expr::Let(name, bound_expr, in_expr) => {
                let bound_value = eval_expr(bound_expr.clone(), &env)?;
                let inner_env = env.extend_value(&name, bound_value);

                // tail call optimization
                env = inner_env;
                expr = in_expr.clone();
                continue 'outer;
            }
            Expr::Ap(func, arg) => {
                let func_eval = eval_expr(func.clone(), &env)?;
                match &func_eval {
                    Value::Fix => return apply_fix(arg, &env),
                    Value::RefValue(ref_value) => match ref_value.borrow() {
                        RefValue::Func {
                            param,
                            body,
                            closure,
                        } => {
                            let arg_eval = eval_expr(arg.clone(), &env)?;
                            let inner_env = closure.extend_value(&param, arg_eval);

                            // tail call optimization
                            env = inner_env;
                            expr = body.clone();
                            continue 'outer;
                        }
                        RefValue::BuiltinFunc(f) => {
                            let arg_eval = eval_expr(arg.clone(), &env)?;
                            return f.eval(arg_eval);
                        }
                        RefValue::Data(name, args) => {
                            let arg_eval = eval_expr(arg.clone(), &env)?;
                            let mut new_args = args.clone();
                            new_args.push(arg_eval);
                            return Ok(Value::data(name.clone(), new_args));
                        }
                    },
                    _ => return Err(format!("Expected function, got {}", func_eval)),
                }
            }
            Expr::Match(match_expr, patterns) => {
                let expr_eval = eval_expr(match_expr.clone(), &env)?;
                for (choice_pattern, choice_expr) in patterns {
                    if let Some(bound_env) = try_pattern_match(&env, &expr_eval, choice_pattern) {
                        // tail call optimization
                        expr = choice_expr.clone();
                        env = bound_env;
                        continue 'outer;
                    }
                }
                return Err(format!("No match for {}", expr_eval));
            }
        }
    }
}

fn apply_fix(arg: &Rc<Expr>, env: &Env) -> Result<Value, String> {
    let arg_eval = eval_expr(arg.clone(), env)?.as_ref_value()?;
    match arg_eval.borrow() {
        RefValue::Func {
            param,
            body,
            closure,
        } => {
            let fun = Rc::new(RefCell::new(None));
            let body_eval = eval_expr(body.clone(), &closure.extend_thunk(&param, fun.clone()))?;
            fun.replace(Some(body_eval.clone()));
            Ok(body_eval)
        }
        _ => Err(format!("Unexpected argument to fix: {}", arg)),
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
        env.eval_expr(parse_expr(s)).unwrap()
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
            Value::func("x".to_string(), Rc::new(e_ident!("x")), env.clone(),)
        );
        assert_eq!(eval_env(&env, "a"), Value::Int(1));
        assert_eq!(
            eval_env(&env, "Negative"),
            Value::data("Negative".to_string(), vec![])
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
            Value::data("Positive".to_string(), vec![])
        );
        assert_eq!(
            eval_env(&env, "f 0"),
            Value::data("Zero".to_string(), vec![])
        );
        assert_eq!(
            eval_env(&env, "f (neg 5)"),
            Value::data("Negative".to_string(), vec![])
        );
    }

    #[test]
    fn eval_parameterized_data() {
        let env = eval_file(
            "data List a = Nil | Cons a (List a)
            let l = Cons 1 (Cons 2 Nil)",
        );

        assert_eq!(
            eval_env(&env, "l"),
            Value::data(
                "Cons".to_string(),
                vec![
                    Value::Int(1),
                    Value::data(
                        "Cons".to_string(),
                        vec![Value::Int(2), Value::data("Nil".to_string(), vec![])]
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
            "data Pair a b = Pair a b
            let f = fun x ->
                match x with
                | Pair a b -> a + b",
        );

        assert_same_value!(env, "f (Pair 1 2)", "3");
    }

    #[test]
    fn match_list() {
        let env = eval_file(
            "data List a = Nil | Cons a (List a)
            let rec len = fun l ->
                match l with
                | Nil -> 0
                | Cons _ xs -> 1 + (len xs)",
        );

        assert_same_value!(env, "len (Cons 1 (Cons 2 (Cons 3 Nil)))", "3");
        assert_same_value!(env, "len Nil", "0");
    }

    #[test]
    fn mutual_recursion() {
        let env = eval_file(
            "let is_even_is_odd = fix (fun f -> 
                (fun n -> if n == 0 then true else (snd f) (n - 1), 
                 fun n -> if n == 0 then false else (fst f) (n - 1)))

            let is_even = fst is_even_is_odd
            let is_odd = snd is_even_is_odd
            ",
        );

        assert_same_value!(env, "is_even 10", "true");
        assert_same_value!(env, "is_even 11", "false");
        assert_same_value!(env, "is_odd 10", "false");
        assert_same_value!(env, "is_odd 11", "true");
    }
}
