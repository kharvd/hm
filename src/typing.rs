use std::fmt::Display;

use rpds::RedBlackTreeSet;

use crate::{
    ast::{Expr, ExprPattern, TypeExpr},
    env::Env,
};

#[derive(Clone)]
struct Constraint {
    lhs: TypeExpr,
    rhs: TypeExpr,
}

impl Constraint {
    fn new(lhs: TypeExpr, rhs: TypeExpr) -> Self {
        Constraint { lhs, rhs }
    }
}

impl Display for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} = {}", self.lhs, self.rhs)
    }
}

pub struct Inference {
    inferred_type: TypeExpr,
    constraints: Vec<Constraint>,
}

impl Display for Inference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, ": {} -|\n", self.inferred_type,)?;
        for constraint in &self.constraints {
            write!(f, "\t{}\n", constraint)?;
        }
        Ok(())
    }
}

pub fn infer(env: &Env, expr: &Expr) -> Result<TypeExpr, String> {
    let inference = infer_constraints(env, expr)?;
    let (inferred, _) = unify(inference)?;
    let (generalized, _) = generalize(
        env,
        Inference {
            inferred_type: inferred.clone(),
            constraints: Vec::new(),
        },
    )?;

    assert!(generalized.is_scheme());
    Ok(generalized.normalize())
}

fn infer_constraints(env: &Env, expr: &Expr) -> Result<Inference, String> {
    infer_constraints_inner(env, expr, &mut 0)
}

fn allocate_type_var(counter: &mut u64) -> TypeExpr {
    let type_var = TypeExpr::type_var(format!("t{}", counter).as_str());
    *counter += 1;
    type_var
}

fn infer_constraints_inner(
    env: &Env,
    expr: &Expr,
    type_var_counter: &mut u64,
) -> Result<Inference, String> {
    Ok(match expr {
        Expr::Int(_) => Inference {
            inferred_type: TypeExpr::Int,
            constraints: Vec::new(),
        },
        Expr::Bool(_) => Inference {
            inferred_type: TypeExpr::Bool,
            constraints: Vec::new(),
        },
        Expr::Char(_) => Inference {
            inferred_type: TypeExpr::Char,
            constraints: Vec::new(),
        },
        Expr::Ident(name) => {
            let type_var = allocate_type_var(type_var_counter);
            let ident_type = env.resolve_type(name)?;
            let instantiated_type = instantiate(&ident_type, type_var_counter);
            Inference {
                inferred_type: type_var.clone(),
                constraints: vec![Constraint {
                    lhs: type_var,
                    rhs: instantiated_type,
                }],
            }
        }
        Expr::If(cond, if_true, if_false) => {
            let type_var = allocate_type_var(type_var_counter);
            let mut infer_cond = infer_constraints_inner(env, cond, type_var_counter)?;
            let mut infer_if_true = infer_constraints_inner(env, if_true, type_var_counter)?;
            let mut infer_if_false = infer_constraints_inner(env, if_false, type_var_counter)?;

            let mut new_constraints = Vec::new();
            new_constraints.append(&mut infer_cond.constraints);
            new_constraints.append(&mut infer_if_true.constraints);
            new_constraints.append(&mut infer_if_false.constraints);
            new_constraints.push(Constraint::new(infer_cond.inferred_type, TypeExpr::Bool));
            new_constraints.push(Constraint::new(
                type_var.clone(),
                infer_if_true.inferred_type,
            ));
            new_constraints.push(Constraint::new(
                type_var.clone(),
                infer_if_false.inferred_type,
            ));

            Inference {
                inferred_type: type_var,
                constraints: new_constraints,
            }
        }
        Expr::Lambda(param, body) => {
            let type_var = allocate_type_var(type_var_counter);
            let inner_env = env.extend_type(param, type_var.clone());
            let infer_body = infer_constraints_inner(&inner_env, body, type_var_counter)?;

            Inference {
                inferred_type: TypeExpr::fun(type_var, infer_body.inferred_type),
                constraints: infer_body.constraints,
            }
        }
        Expr::Ap(func, arg) => {
            let type_var = allocate_type_var(type_var_counter);
            let mut infer_func = infer_constraints_inner(env, func, type_var_counter)?;
            let mut infer_arg = infer_constraints_inner(env, arg, type_var_counter)?;

            let mut new_constraints = Vec::new();
            new_constraints.append(&mut infer_func.constraints);
            new_constraints.append(&mut infer_arg.constraints);
            new_constraints.push(Constraint::new(
                infer_func.inferred_type,
                TypeExpr::fun(infer_arg.inferred_type, type_var.clone()),
            ));

            Inference {
                inferred_type: type_var,
                constraints: new_constraints,
            }
        }
        Expr::Let(name, bound_expr, expr) => {
            let infer_bound = infer_constraints_inner(env, bound_expr, type_var_counter)?;
            let mut bound_constraints = infer_bound.constraints.clone();
            let (generalized_bound, env2) = generalize(env, infer_bound)?;

            let mut infer_expr = infer_constraints_inner(
                &env2.extend_type(name, generalized_bound),
                expr,
                type_var_counter,
            )?;

            let mut new_constraints = Vec::new();
            new_constraints.append(&mut bound_constraints);
            new_constraints.append(&mut infer_expr.constraints);

            Inference {
                inferred_type: infer_expr.inferred_type,
                constraints: new_constraints,
            }
        }
        Expr::Match(expr, patterns) => {
            let type_var = allocate_type_var(type_var_counter);
            let mut infer_expr = infer_constraints_inner(env, expr, type_var_counter)?;

            let mut new_constraints = Vec::new();
            new_constraints.append(&mut infer_expr.constraints);

            for (pattern, body) in patterns.iter() {
                let mut infer_pattern =
                    infer_constraints_for_pattern(env, pattern, type_var_counter)?;
                let env_with_bindings = infer_pattern
                    .bindings
                    .iter()
                    .fold(env.clone(), |acc, (name, ty)| {
                        acc.extend_type(name, ty.clone())
                    });
                let mut infer_body =
                    infer_constraints_inner(&env_with_bindings, body, type_var_counter)?;

                new_constraints.append(&mut infer_pattern.inference.constraints);
                new_constraints.append(&mut infer_body.constraints);
                new_constraints.push(Constraint::new(
                    infer_expr.inferred_type.clone(),
                    infer_pattern.inference.inferred_type.clone(),
                ));
                new_constraints.push(Constraint::new(
                    infer_body.inferred_type.clone(),
                    type_var.clone(),
                ));
            }

            Inference {
                inferred_type: type_var,
                constraints: new_constraints,
            }
        }
    })
}

struct PatternBindings {
    inference: Inference,
    bindings: Vec<(String, TypeExpr)>,
}

fn infer_constraints_for_pattern(
    env: &Env,
    pattern: &ExprPattern,
    type_var_counter: &mut u64,
) -> Result<PatternBindings, String> {
    Ok(match pattern {
        ExprPattern::Int(_) => PatternBindings {
            inference: Inference {
                inferred_type: TypeExpr::Int,
                constraints: Vec::new(),
            },
            bindings: Vec::new(),
        },
        ExprPattern::Bool(_) => PatternBindings {
            inference: Inference {
                inferred_type: TypeExpr::Bool,
                constraints: Vec::new(),
            },
            bindings: Vec::new(),
        },
        ExprPattern::Char(_) => PatternBindings {
            inference: Inference {
                inferred_type: TypeExpr::Char,
                constraints: Vec::new(),
            },
            bindings: Vec::new(),
        },
        ExprPattern::Wildcard => {
            let type_var = allocate_type_var(type_var_counter);
            PatternBindings {
                inference: Inference {
                    inferred_type: type_var.clone(),
                    constraints: Vec::new(),
                },
                bindings: Vec::new(),
            }
        }
        ExprPattern::Variable(name) => {
            let type_var = allocate_type_var(type_var_counter);
            PatternBindings {
                inference: Inference {
                    inferred_type: type_var.clone(),
                    constraints: Vec::new(),
                },
                bindings: vec![(name.clone(), type_var)],
            }
        }
        ExprPattern::Constructor(name, args) => {
            let constructor_type = env.resolve_type(name)?;
            let instantiated_constructor_type = instantiate(&constructor_type, type_var_counter);
            let (constructor_args, constructor_return_type) =
                instantiated_constructor_type.as_function_type();

            let mut new_constraints = Vec::new();
            let mut bindings = Vec::new();
            for (arg, pattern_arg) in constructor_args.iter().zip(args.iter()) {
                let mut infer_arg =
                    infer_constraints_for_pattern(env, pattern_arg, type_var_counter)?;
                new_constraints.append(&mut infer_arg.inference.constraints);
                new_constraints.push(Constraint::new(
                    (**arg).clone(),
                    infer_arg.inference.inferred_type,
                ));
                bindings.append(&mut infer_arg.bindings);
            }

            PatternBindings {
                inference: Inference {
                    inferred_type: constructor_return_type,
                    constraints: new_constraints,
                },
                bindings,
            }
        }
    })
}

fn generalize(env: &Env, infer_bound: Inference) -> Result<(TypeExpr, Env), String> {
    let (unified_bound, substitutions) = unify(infer_bound)?;
    let new_env = env.substitute(&substitutions);

    let bound_free_vars = unified_bound.free_variables();
    let new_env_free_vars = new_env.free_type_vars();
    let mut new_type_vars = bound_free_vars.clone();

    for var in new_env_free_vars.iter() {
        new_type_vars = new_type_vars.remove(var);
    }

    let generalized_type = TypeExpr::forall(new_type_vars, unified_bound);

    Ok((generalized_type, new_env))
}

fn instantiate(type_expr: &TypeExpr, type_var_counter: &mut u64) -> TypeExpr {
    match type_expr {
        TypeExpr::Forall(vars, ty) => {
            let mut substitutions = Vec::new();
            for var in vars.iter() {
                substitutions.push((var.clone(), allocate_type_var(type_var_counter)));
            }
            substitute(&ty, &substitutions)
        }
        _ => type_expr.clone(),
    }
}

type Substitution = (String, TypeExpr);

fn unify(inference: Inference) -> Result<(TypeExpr, Vec<Substitution>), String> {
    let mut constraints = inference.constraints;
    let mut substitutions: Vec<Substitution> = Vec::new();
    while let Some(constraint) = constraints.pop() {
        let constraint = Constraint::new(
            substitute(&constraint.lhs, &substitutions),
            substitute(&constraint.rhs, &substitutions),
        );

        let Reduction {
            new_constraints,
            substitution,
        } = reduce(constraint)?;

        for new_constraint in new_constraints {
            constraints.push(new_constraint);
        }

        if let Some(substitution) = substitution {
            substitutions.push(substitution);
        }
    }

    Ok((
        substitute(&inference.inferred_type, &substitutions),
        substitutions,
    ))
}

fn substitute(t: &TypeExpr, substitutions: &Vec<Substitution>) -> TypeExpr {
    substitutions
        .into_iter()
        .fold(t.clone(), |acc, sub| acc.substitute(sub))
}

impl TypeExpr {
    fn is_free(&self, name: &str) -> bool {
        match self {
            TypeExpr::Int | TypeExpr::Bool | TypeExpr::Char => true,
            TypeExpr::Constructor(_, args) => args.iter().all(|arg| arg.is_free(name)),
            TypeExpr::Fun(t1, t2) => t1.is_free(name) && t2.is_free(name),
            TypeExpr::TypeVar(other_name) => name != other_name,
            TypeExpr::Forall(vars, ty) => {
                if vars.contains(name) {
                    true
                } else {
                    ty.is_free(name)
                }
            }
        }
    }

    fn substitute(&self, sub: &Substitution) -> Self {
        match self {
            TypeExpr::Int | TypeExpr::Bool | TypeExpr::Char => self.clone(),
            TypeExpr::Constructor(name, args) => {
                TypeExpr::constructor(name, args.iter().map(|arg| arg.substitute(sub)).collect())
            }
            TypeExpr::Fun(t1, t2) => TypeExpr::fun(t1.substitute(sub), t2.substitute(sub)),
            TypeExpr::TypeVar(name) => {
                if name == &sub.0 {
                    sub.1.clone()
                } else {
                    self.clone()
                }
            }
            TypeExpr::Forall(vars, ty) => {
                if vars.contains(&sub.0) {
                    self.clone()
                } else {
                    TypeExpr::forall(vars.clone(), ty.substitute(sub))
                }
            }
        }
    }

    pub fn free_variables(&self) -> RedBlackTreeSet<String> {
        match self {
            TypeExpr::Int | TypeExpr::Bool | TypeExpr::Char => RedBlackTreeSet::new(),
            TypeExpr::Constructor(_, args) => {
                let mut vars = RedBlackTreeSet::new();
                for arg in args.iter() {
                    for var in arg.free_variables().iter() {
                        vars = vars.insert(var.clone());
                    }
                }
                vars
            }
            TypeExpr::Fun(t1, t2) => {
                let vars1 = t1.free_variables();
                let vars2 = t2.free_variables();
                let mut vars = vars1.clone();
                for var in vars2.iter() {
                    vars = vars.insert(var.clone());
                }
                vars
            }
            TypeExpr::TypeVar(name) => RedBlackTreeSet::new().insert(name.clone()),
            TypeExpr::Forall(vars, ty) => {
                let mut free_vars = ty.free_variables();
                for var in vars {
                    free_vars = free_vars.remove(var);
                }
                free_vars
            }
        }
    }

    pub fn normalize(self) -> Self {
        match self {
            TypeExpr::Forall(vars, ty) => {
                if vars.is_empty() {
                    (*ty).clone()
                } else {
                    let mut counter = 0;
                    let mut substitutions = Vec::new();
                    let mut new_vars = RedBlackTreeSet::new();

                    for var in vars.iter() {
                        let new_name = if counter < 26 {
                            format!("{}", ('a' as u8 + counter as u8) as char)
                        } else {
                            format!("t{}", counter - 26)
                        };
                        let type_var = TypeExpr::type_var(new_name.as_str());
                        counter += 1;
                        substitutions.push((var.clone(), type_var));
                        new_vars = new_vars.insert(new_name);
                    }

                    TypeExpr::forall(new_vars, substitute(&ty, &substitutions))
                }
            }
            s => s,
        }
    }
}

impl Env {
    fn substitute(&self, substitutions: &Vec<Substitution>) -> Self {
        let mut new_env = self.clone();
        for (name, ty) in self.typings.iter() {
            new_env = new_env.extend_type(name, substitute(ty, substitutions));
        }
        new_env
    }
}

struct Reduction {
    new_constraints: Vec<Constraint>,
    substitution: Option<Substitution>,
}

impl Reduction {
    fn new(new_constraints: Vec<Constraint>, substitution: Option<Substitution>) -> Self {
        Reduction {
            new_constraints,
            substitution,
        }
    }

    fn empty() -> Self {
        Reduction::new(Vec::new(), None)
    }

    fn substitution(substitution: Substitution) -> Self {
        Reduction::new(Vec::new(), Some(substitution))
    }

    fn constraint(new_constraints: Vec<Constraint>) -> Self {
        Reduction::new(new_constraints, None)
    }
}

fn reduce(constraint: Constraint) -> Result<Reduction, String> {
    match (constraint.lhs, constraint.rhs) {
        (TypeExpr::Int, TypeExpr::Int)
        | (TypeExpr::Bool, TypeExpr::Bool)
        | (TypeExpr::Char, TypeExpr::Char) => Ok(Reduction::empty()),
        (TypeExpr::TypeVar(x), TypeExpr::TypeVar(y)) => {
            if x == y {
                Ok(Reduction::empty())
            } else {
                Ok(Reduction::substitution((x.clone(), TypeExpr::TypeVar(y))))
            }
        }
        (TypeExpr::Fun(t1, t2), TypeExpr::Fun(t3, t4)) => Ok(Reduction::constraint(vec![
            Constraint::new((*t1).clone(), (*t3).clone()),
            Constraint::new((*t2).clone(), (*t4).clone()),
        ])),
        (TypeExpr::Constructor(name1, args1), TypeExpr::Constructor(name2, args2))
            if name1 == name2 && args1.len() == args2.len() =>
        {
            let mut new_constraints = Vec::new();
            for (arg1, arg2) in args1.into_iter().zip(args2.into_iter()) {
                new_constraints.push(Constraint::new((*arg1).clone(), (*arg2).clone()));
            }
            Ok(Reduction::constraint(new_constraints))
        }
        (TypeExpr::TypeVar(x), rhs) if rhs.is_free(&x) => Ok(Reduction::substitution((x, rhs))),
        (lhs, TypeExpr::TypeVar(x)) if lhs.is_free(&x) => Ok(Reduction::substitution((x, lhs))),
        (lhs, rhs) => Err(format!("Failed to unify constraint {} = {}", lhs, rhs)),
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::{Expr, TypeExpr},
        env::Env,
        lexer::tokenize,
        parser,
    };

    use super::infer;

    fn parse_expr(s: &str) -> Expr {
        match parser::parse(s).unwrap() {
            parser::ParseResult::Statement(_) => panic!("not an expression"),
            parser::ParseResult::Expression(expr) => expr,
        }
    }

    fn parse_type_expr(s: &str) -> Result<TypeExpr, String> {
        let tokens = tokenize(s).unwrap();
        let mut iter = tokens.into_iter().peekable();
        crate::parser::parse_type_expr(&mut iter)
    }

    fn infer_type(env: &Env, s: &str) -> Result<TypeExpr, String> {
        infer(env, &parse_expr(s))
    }

    macro_rules! assert_type {
        ($env:expr, $s:expr, $expected:expr) => {
            assert_eq!(
                infer_type($env, $s).unwrap(),
                parse_type_expr($expected).unwrap()
            )
        };
    }

    macro_rules! assert_type_error {
        ($env:expr, $s:expr, $expected:expr) => {
            assert_eq!(infer_type($env, $s), Err($expected.to_string()))
        };
    }

    #[test]
    fn test_simple() {
        let env = Env::prelude();
        assert_type!(&env, "5", "int");
        assert_type!(&env, "true", "bool");
        assert_type!(&env, "fun x -> x + 1", "int -> int");
        assert_type!(&env, "fun x -> (+ x)", "int -> int -> int");
    }

    #[test]
    fn test_polymorphic() {
        let mut env = Env::prelude();
        env = env.eval_file("let id = fun x -> x").unwrap();

        assert_type!(&env, "id", "forall a. a -> a");
        assert_type!(&env, "id 5", "int");
        assert_type!(&env, "id neg", "int -> int");
        assert_type!(&env, "fun x -> id x", "forall a. a -> a");
        assert_type!(&env, "fun x -> id (id x)", "forall a. a -> a");
    }

    #[test]
    fn test_if() {
        let env = Env::prelude();
        assert_type!(&env, "fun x -> if x then 1 else 2", "bool -> int");
        assert_type_error!(
            &env,
            "fun x -> if x then 1 else true",
            "Failed to unify constraint bool = int"
        );
        assert_type_error!(
            &env,
            "fun x -> if x then 1 else neg",
            "Failed to unify constraint int = (int -> int)"
        );
    }

    #[test]
    fn test_let() {
        let env = Env::prelude();
        assert_type!(&env, "(let f = fun x -> x in f)", "forall a. a -> a");
        assert_type!(&env, "(let f = fun x -> x in f 5)", "int");
        assert_type!(&env, "(let f = fun x -> x in f neg)", "int -> int");
        assert_type!(
            &env,
            "(let id = fun x -> x in (let a = id 0 in id true))",
            "bool"
        );
    }

    #[test]
    fn test_fun() {
        let env = Env::prelude();

        assert_type!(&env, "fun x -> fun y -> x", "forall a b. a -> b -> a");
        assert_type!(&env, "fun x -> fun y -> y", "forall a b. a -> b -> b");
        assert_type!(
            &env,
            "(let g = fun x -> fun y -> fun z -> y in g 1)",
            "forall a b. a -> b -> a"
        );
    }

    #[test]
    fn test_data() {
        let mut env = Env::prelude();
        env = env
            .eval_file("data Sign = Negative | Zero | Positive")
            .unwrap();
        assert_type!(&env, "Negative", "Sign");
        assert_type!(&env, "Zero", "Sign");
        assert_type!(&env, "Positive", "Sign");
        assert_type!(
            &env,
            "fun x -> if x < 0 then Negative else if x > 0 then Positive else Zero",
            "int -> Sign"
        );
    }

    #[test]
    fn test_parameterized_data() {
        let mut env = Env::prelude();
        env = env.eval_file("data Maybe a = Nothing | Just a").unwrap();

        assert_type!(&env, "Nothing", "forall a. Maybe a");
        assert_type!(&env, "Just 5", "Maybe int");
        assert_type!(&env, "Just true", "Maybe bool");
        assert_type!(&env, "fun x -> Just x", "forall a. a -> Maybe a");
        assert_type!(
            &env,
            "fun x -> Just (Just x)",
            "forall a. a -> Maybe (Maybe a)"
        );
        assert_type!(
            &env,
            "fun x -> Just (Just (Just x))",
            "forall a. a -> Maybe (Maybe (Maybe a))"
        );
    }

    #[test]
    fn test_pair() {
        let mut env = Env::prelude();
        env = env.eval_file("data Pair a b = Pair a b").unwrap();

        assert_type!(&env, "Pair 5 true", "Pair int bool");
        assert_type!(&env, "fun x -> Pair x 5", "forall a. a -> Pair a int");
        assert_type!(&env, "fun x -> Pair x true", "forall a. a -> Pair a bool");
        assert_type!(
            &env,
            "fun x -> fun y -> Pair y (Pair x true)",
            "forall a b. a -> b -> Pair b (Pair a bool)"
        );
    }

    #[test]
    fn test_match_constant() {
        let mut env = Env::prelude();
        env = env
            .eval_file(
                "
            data Sign = Negative | Zero | Positive
            data Maybe a = Nothing | Just a
            ",
            )
            .unwrap();

        assert_type!(
            &env,
            "fun x -> match x with | Negative -> 0 | Zero -> 1 | Positive -> 2",
            "Sign -> int"
        );
        assert_type_error!(
            &env,
            "fun x -> match x with | Negative -> 0 | Zero -> true | Positive -> 2",
            "Failed to unify constraint bool = int"
        );
        assert_type_error!(
            &env,
            "fun x -> match x with | Negative -> 0 | Nothing -> 1",
            "Failed to unify constraint (Maybe t3) = Sign"
        )
    }

    #[test]
    fn test_match_args() {
        let mut env = Env::prelude();
        env = env.eval_file("data Maybe a = Nothing | Just a").unwrap();

        assert_type!(
            &env,
            "fun x -> match x with | Nothing -> 0 | Just y -> 1",
            "forall a. Maybe a -> int"
        );
        assert_type!(
            &env,
            "fun x -> match x with | Nothing -> 0 | Just 1 -> 1",
            "Maybe int -> int"
        );
        assert_type!(
            &env,
            "fun f -> fun x -> match x with | Nothing -> Nothing | Just y -> Just (f y)",
            "forall a b. (a -> b) -> Maybe a -> Maybe b"
        );
    }

    #[test]
    fn test_match_nested() {
        let mut env = Env::prelude();
        env = env.eval_file("data Maybe a = Nothing | Just a").unwrap();

        assert_type!(
            &env,
            "fun x -> match x with | Nothing -> 0 | Just Nothing -> 1 | Just (Just y) -> 2",
            "forall a. Maybe (Maybe a) -> int"
        );
        assert_type!(
            &env,
            "fun f -> fun x -> match x with | Nothing -> Nothing | Just Nothing -> Just Nothing | Just (Just y) -> Just (Just (f y))",
            "forall a b. (b -> a) -> Maybe (Maybe b) -> Maybe (Maybe a)"
        );
    }
}
