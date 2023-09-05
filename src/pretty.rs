use std::fmt::Display;

use crate::ast::{Expr, ExprPattern, MatchCase, Statement, TypeExpr};
use itertools::Itertools;

impl ExprPattern {
    fn precedence(&self) -> usize {
        match self {
            ExprPattern::Int(_) => 10,
            ExprPattern::Bool(_) => 10,
            ExprPattern::Char(_) => 10,
            ExprPattern::Variable(_) => 10,
            ExprPattern::Wildcard => 10,
            ExprPattern::Constructor(_, _) => 5,
        }
    }

    pub fn pretty_print(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        precedence: usize,
        indent: usize,
    ) -> std::fmt::Result {
        if precedence > self.precedence() {
            write!(f, "(")?;
        }

        match self {
            ExprPattern::Int(n) => write!(f, "{}", n)?,
            ExprPattern::Bool(b) => write!(f, "{}", b)?,
            ExprPattern::Char(c) => write!(f, "\'{}\'", c)?,
            ExprPattern::Variable(s) => write!(f, "{}", s)?,
            ExprPattern::Constructor(name, args) => {
                if args.is_empty() {
                    write!(f, "{}", name)?;
                } else {
                    write!(f, "{}", name)?;

                    for arg in args {
                        write!(f, " ")?;
                        arg.pretty_print(f, self.precedence() + 1, indent)?;
                    }
                }
            }
            ExprPattern::Wildcard => write!(f, "_")?,
        }

        if precedence > self.precedence() {
            write!(f, ")")?;
        }

        Ok(())
    }
}

impl Display for ExprPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.pretty_print(f, 0, 0)
    }
}

impl Expr {
    fn precedence(&self) -> usize {
        match self {
            Expr::Int(_) => 10,
            Expr::Bool(_) => 10,
            Expr::Char(_) => 10,
            Expr::Ident(_) => 10,
            Expr::Ap(_, _) => 5,
            Expr::If(_, _, _) => 1,
            Expr::Let(_, _, _) => 1,
            Expr::Lambda(_, _) => 1,
            Expr::Match(_, _) => 1,
        }
    }

    pub fn pretty_print(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        precedence: usize,
        indent: usize,
    ) -> std::fmt::Result {
        let parenthesized = precedence > self.precedence();

        if parenthesized {
            write!(f, "(")?;
        }

        let indentation = " ".repeat(indent);
        let indentation_next = " ".repeat(indent + 2);

        match self {
            Expr::Int(n) => write!(f, "{}", n)?,
            Expr::Bool(b) => write!(f, "{}", b)?,
            Expr::Char(c) => {
                write!(f, "'")?;
                match c {
                    '\n' => write!(f, "\\n")?,
                    '\r' => write!(f, "\\r")?,
                    '\t' => write!(f, "\\t")?,
                    '\\' => write!(f, "\\\\")?,
                    '\'' => write!(f, "\\'")?,
                    c => write!(f, "{}", c)?,
                };
                write!(f, "'")?;
            }
            Expr::Ident(s) => write!(f, "{}", s)?,
            Expr::If(cond, then, else_) => {
                write!(f, "\n{}", indentation)?;
                write!(f, "if ")?;
                cond.pretty_print(f, self.precedence(), indent)?;
                write!(f, "\n{}", indentation)?;
                write!(f, "then ")?;
                then.pretty_print(f, self.precedence(), indent + 2)?;
                write!(f, "\n{}", indentation)?;
                write!(f, "else ")?;
                else_.pretty_print(f, self.precedence(), indent + 2)?;
            }
            Expr::Let(name, expr, body) => {
                write!(f, "\n{}", indentation)?;
                write!(f, "let")?;
                write!(f, "\n{}", indentation_next)?;
                write!(f, "{} = ", name)?;
                expr.pretty_print(f, self.precedence(), indent + 2)?;
                write!(f, "\n{}", indentation)?;
                write!(f, "in ")?;
                body.pretty_print(f, self.precedence(), indent + 2)?;
            }
            Expr::Lambda(param, body) => {
                write!(f, "fun {} -> ", param)?;
                body.pretty_print(f, self.precedence(), indent + 2)?;
            }
            Expr::Ap(fun, arg) => {
                fun.pretty_print(f, self.precedence(), indent + 2)?;
                write!(f, " ")?;
                arg.pretty_print(f, self.precedence() + 1, indent + 2)?;
            }
            Expr::Match(expr, cases) => {
                write!(f, "\n{}", indentation)?;
                write!(f, "match ")?;
                expr.pretty_print(f, self.precedence(), indent + 2)?;
                write!(f, " with")?;
                for case in cases {
                    let MatchCase {
                        pattern,
                        guard,
                        body,
                    } = case;
                    write!(f, "\n{}", indentation)?;
                    write!(f, "| ")?;

                    pattern.pretty_print(f, 0, indent)?;

                    write!(f, " ")?;

                    if let Some(guard) = guard {
                        write!(f, "if ")?;
                        guard.pretty_print(f, 0, indent + 2)?;
                        write!(f, " ")?;
                    }

                    write!(f, "-> ")?;
                    body.pretty_print(f, 0, indent + 2)?;
                }
            }
        }

        if parenthesized {
            write!(f, ")")?;
        }

        Ok(())
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.pretty_print(f, 0, 0)
    }
}

impl TypeExpr {
    fn precedence(&self) -> usize {
        match self {
            TypeExpr::Int => 10,
            TypeExpr::Bool => 10,
            TypeExpr::Char => 10,
            TypeExpr::TypeVar(_) => 10,
            TypeExpr::Forall(_, _) => 10,
            TypeExpr::Constructor(_, _) => 5,
            TypeExpr::Fun(_, _) => 1,
        }
    }

    pub fn pretty_print(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        precedence: usize,
    ) -> std::fmt::Result {
        let parenthesized = precedence > self.precedence();

        if parenthesized {
            write!(f, "(")?;
        }

        match self {
            TypeExpr::Int => write!(f, "int")?,
            TypeExpr::Bool => write!(f, "bool")?,
            TypeExpr::Char => write!(f, "char")?,
            TypeExpr::Fun(from_type, to_type) => {
                from_type.pretty_print(f, self.precedence() + 1)?;
                write!(f, " -> ")?;
                to_type.pretty_print(f, self.precedence())?;
            }
            TypeExpr::TypeVar(s) => write!(f, "{}", s)?,
            TypeExpr::Forall(vars, ty) => {
                if vars.is_empty() {
                    write!(f, "{}", ty)?;
                } else {
                    write!(
                        f,
                        "forall {} . {}",
                        vars.into_iter().map(|v| format!("{}", v)).join(" "),
                        ty
                    )?;
                }
            }
            TypeExpr::Constructor(name, args) => {
                if args.is_empty() {
                    write!(f, "{}", name)?;
                } else {
                    write!(f, "{}", name)?;

                    for arg in args {
                        write!(f, " ")?;
                        arg.pretty_print(f, self.precedence() + 1)?;
                    }
                }
            }
        }

        if parenthesized {
            write!(f, ")")?;
        }

        Ok(())
    }
}

impl Display for TypeExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.pretty_print(f, 0)
    }
}

impl Display for Statement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Statement::Let(name, expr) => {
                write!(f, "let {} = ", name)?;
                expr.pretty_print(f, 0, 2)
            }
            Statement::Val(name, ty) => write!(f, "val {} : {}", name, ty),
            Statement::Data(name, args, variants) => {
                write!(
                    f,
                    "data {} {} = {}",
                    name,
                    args.iter().map(|arg| format!("{}", arg)).join(" "),
                    variants.iter().join(" | ")
                )
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use crate::{e_ap, e_ident, e_int, e_lambda, t_bool, t_constructor, t_fun, t_int, t_type_var};

    use super::*;

    #[test]
    fn display_expr() {
        let expr = e_ap!(
            e_lambda!(
                "x",
                e_ap!(
                    e_lambda!("y", e_ap!(e_ident!("x"), e_ident!("y"))),
                    e_int!(1)
                )
            ),
            e_int!(2)
        );
        assert_eq!(format!("{}", expr), "(fun x -> (fun y -> x y) 1) 2");
    }

    #[test]
    fn display_type_expr() {
        let ty = t_fun!(t_fun!(t_int!(), t_type_var!("a")), t_bool!());
        assert_eq!(format!("{}", ty), "(int -> a) -> bool");

        let ty = t_constructor!(
            "Constructor",
            t_int!(),
            t_bool!(),
            t_fun!(t_int!(), t_int!())
        );
        assert_eq!(format!("{}", ty), "Constructor int bool (int -> int)");

        let ty = t_fun!(
            t_constructor!("Tuple2", t_type_var!("a"), t_type_var!("b")),
            t_type_var!("a")
        );
        assert_eq!(format!("{}", ty), "Tuple2 a b -> a");

        let ty = t_fun!(
            t_fun!(
                t_constructor!("Maybe", t_type_var!("a")),
                t_constructor!("List", t_constructor!("Maybe", t_type_var!("b")))
            ),
            t_fun!(
                t_constructor!("List", t_constructor!("Maybe", t_type_var!("a"))),
                t_constructor!("List", t_constructor!("Maybe", t_type_var!("b")))
            )
        );
        assert_eq!(
            format!("{}", ty),
            "(Maybe a -> List (Maybe b)) -> List (Maybe a) -> List (Maybe b)"
        );
    }

    #[test]
    fn display_let_statement() {
        let stmt = Statement::Let(
            "f".to_string(),
            Rc::new(e_lambda!("x", e_ap!(e_ident!("f"), e_ident!("x")))),
        );
        assert_eq!(format!("{}", stmt), "let f = fun x -> f x");
    }

    #[test]
    fn display_val_statement() {
        let stmt = Statement::Val(
            "f".to_string(),
            Rc::new(t_fun!(t_type_var!("a"), t_type_var!("a"))),
        );
        assert_eq!(format!("{}", stmt), "val f : a -> a");
    }
}
