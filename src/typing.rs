use std::{borrow::Borrow, fmt::Display, rc::Rc};

use crate::{
    ast::{Expr, TypeExpr},
    env::Env,
};

struct Constraint {
    lhs: Rc<TypeExpr>,
    rhs: Rc<TypeExpr>,
}

impl Constraint {
    fn new(lhs: Rc<TypeExpr>, rhs: Rc<TypeExpr>) -> Self {
        Constraint { lhs, rhs }
    }
}

impl Display for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} = {}", self.lhs, self.rhs)
    }
}

pub struct Inference {
    inferred_type: Rc<TypeExpr>,
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

pub fn infer(env: &Env, expr: &Expr) -> Result<Rc<TypeExpr>, String> {
    let inference = infer_constraints(env, expr)?;
    // print!("{} {}\n", expr, inference);
    unify(inference)
}

fn infer_constraints(env: &Env, expr: &Expr) -> Result<Inference, String> {
    let mut type_var_counter = 0;
    infer_constraints_inner(env, expr, &mut type_var_counter)
}

fn allocate_type_var(counter: &mut u64) -> Rc<TypeExpr> {
    let type_var = Rc::new(TypeExpr::TypeVar(format!("t{}", counter)));
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
            inferred_type: Rc::new(TypeExpr::Int),
            constraints: Vec::new(),
        },
        Expr::Bool(_) => Inference {
            inferred_type: Rc::new(TypeExpr::Bool),
            constraints: Vec::new(),
        },
        Expr::Ident(name) => {
            let type_var = allocate_type_var(type_var_counter);
            let ident_type = env.resolve_type(name)?;
            Inference {
                inferred_type: type_var.clone(),
                constraints: vec![Constraint {
                    lhs: type_var,
                    rhs: ident_type,
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
            new_constraints.push(Constraint::new(
                infer_cond.inferred_type,
                Rc::new(TypeExpr::Bool),
            ));
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
                inferred_type: Rc::new(TypeExpr::Fun(type_var, infer_body.inferred_type)),
                constraints: infer_body.constraints,
            }
        }
        Expr::Ap(func, arg) => {
            let type_var = allocate_type_var(type_var_counter);
            let mut infer_func = infer_constraints_inner(env, func, type_var_counter)?;
            // println!("infer_func {}", infer_func);
            let mut infer_arg = infer_constraints_inner(env, arg, type_var_counter)?;

            let mut new_constraints = Vec::new();
            new_constraints.append(&mut infer_func.constraints);
            new_constraints.append(&mut infer_arg.constraints);
            new_constraints.push(Constraint::new(
                infer_func.inferred_type,
                Rc::new(TypeExpr::Fun(infer_arg.inferred_type, type_var.clone())),
            ));

            Inference {
                inferred_type: type_var,
                constraints: new_constraints,
            }
        }
    })
}

type Substitution = (String, Rc<TypeExpr>);

fn unify(inference: Inference) -> Result<Rc<TypeExpr>, String> {
    let mut constraints = inference.constraints;
    let mut substitutions: Vec<Substitution> = Vec::new();
    while let Some(constraint) = constraints.pop() {
        let constraint = Constraint::new(
            substitute(constraint.lhs, &substitutions),
            substitute(constraint.rhs, &substitutions),
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

    Ok(substitute(inference.inferred_type, &substitutions))
}

fn substitute(t: Rc<TypeExpr>, substitutions: &Vec<Substitution>) -> Rc<TypeExpr> {
    substitutions
        .into_iter()
        .fold(t, |acc, sub| substitute_one(acc, sub))
}

fn substitute_one(t: Rc<TypeExpr>, sub: &Substitution) -> Rc<TypeExpr> {
    match t.borrow() {
        TypeExpr::Int => t.clone(),
        TypeExpr::Bool => t.clone(),
        TypeExpr::Fun(t1, t2) => Rc::new(TypeExpr::Fun(
            substitute_one(t1.clone(), sub),
            substitute_one(t2.clone(), sub),
        )),
        TypeExpr::TypeVar(name) => {
            if name == &sub.0 {
                sub.1.clone()
            } else {
                t.clone()
            }
        }
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

fn is_free(name: &String, type_expr: Rc<TypeExpr>) -> bool {
    match type_expr.borrow() {
        TypeExpr::Int => true,
        TypeExpr::Bool => true,
        TypeExpr::Fun(t1, t2) => is_free(name, t1.clone()) && is_free(name, t2.clone()),
        TypeExpr::TypeVar(other_name) => name != other_name,
    }
}

fn reduce(constraint: Constraint) -> Result<Reduction, String> {
    match (constraint.lhs.borrow(), constraint.rhs.borrow()) {
        (TypeExpr::Int, TypeExpr::Int) | (TypeExpr::Bool, TypeExpr::Bool) => Ok(Reduction::empty()),
        (TypeExpr::TypeVar(x), TypeExpr::TypeVar(y)) => {
            if x == y {
                Ok(Reduction::empty())
            } else {
                Ok(Reduction::substitution((x.clone(), constraint.rhs.clone())))
            }
        }
        (TypeExpr::Fun(t1, t2), TypeExpr::Fun(t3, t4)) => Ok(Reduction::constraint(vec![
            Constraint::new(t1.clone(), t3.clone()),
            Constraint::new(t2.clone(), t4.clone()),
        ])),
        (TypeExpr::TypeVar(x), _) => {
            if is_free(x, constraint.rhs.clone()) {
                Ok(Reduction::substitution((x.clone(), constraint.rhs.clone())))
            } else {
                Err(format!("Failed to unify constraint {}", constraint))
            }
        }
        (_, TypeExpr::TypeVar(x)) => {
            if is_free(x, constraint.lhs.clone()) {
                Ok(Reduction::substitution((x.clone(), constraint.lhs.clone())))
            } else {
                Err(format!("Failed to unify constraint {}", constraint))
            }
        }
        _ => Err(format!("Failed to unify constraint {}", constraint)),
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

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

    fn parse_type_expr(s: &str) -> TypeExpr {
        let tokens = tokenize(s).unwrap();
        let mut iter = tokens.into_iter().peekable();
        crate::parser::parse_type_expr(&mut iter).unwrap()
    }

    fn infer_type(env: &Env, s: &str) -> Rc<TypeExpr> {
        infer(env, &parse_expr(s)).unwrap().to_owned()
    }

    #[test]
    fn test_simple() {
        let env = Env::prelude();
        assert_eq!(infer_type(&env, "5"), Rc::new(TypeExpr::Int));
        assert_eq!(infer_type(&env, "true"), Rc::new(TypeExpr::Bool));
        assert_eq!(
            infer_type(&env, "fun x -> plus x 1"),
            Rc::new(TypeExpr::Fun(
                Rc::new(TypeExpr::Int),
                Rc::new(TypeExpr::Int)
            ))
        );
    }

    #[test]
    fn test_polymorphic() {
        let env = Env::prelude().extend_type("id", Rc::new(parse_type_expr("'a -> 'a")));
        assert_eq!(infer_type(&env, "id 5"), Rc::new(TypeExpr::Int));
        assert_eq!(
            infer_type(&env, "id neg"),
            Rc::new(TypeExpr::Fun(
                Rc::new(TypeExpr::Int),
                Rc::new(TypeExpr::Int)
            ))
        );
    }
}
