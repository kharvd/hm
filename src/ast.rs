use std::{fmt::Display, rc::Rc};

use itertools::Itertools;
use rpds::RedBlackTreeSet;

#[derive(Debug, PartialEq, Clone)]
pub enum ExprPattern {
    Int(i64),
    Bool(bool),
    Variable(String),
    Wildcard,
    Constructor(String, Vec<Rc<ExprPattern>>),
}

impl Display for ExprPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExprPattern::Int(n) => write!(f, "{}", n),
            ExprPattern::Bool(b) => write!(f, "{}", b),
            ExprPattern::Variable(s) => write!(f, "{}", s),
            ExprPattern::Wildcard => write!(f, "_"),
            ExprPattern::Constructor(name, args) => {
                if args.is_empty() {
                    write!(f, "{}", name)
                } else {
                    write!(
                        f,
                        "({} {})",
                        name,
                        args.iter()
                            .map(|x| x.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    )
                }
            }
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Int(i64),
    Bool(bool),
    Ident(String),
    If(Rc<Expr>, Rc<Expr>, Rc<Expr>),
    Let(String, Rc<Expr>, Rc<Expr>),
    Lambda(String, Rc<Expr>),
    Ap(Rc<Expr>, Rc<Expr>),
    Match(Rc<Expr>, Vec<(Rc<ExprPattern>, Rc<Expr>)>),
}

impl Expr {
    #[cfg(test)]
    pub fn int(n: i64) -> Self {
        Self::Int(n)
    }

    #[cfg(test)]
    pub fn bool(b: bool) -> Self {
        Self::Bool(b)
    }

    pub fn ident(name: &str) -> Self {
        Self::Ident(name.to_string())
    }

    #[cfg(test)]
    pub fn if_(cond: Expr, then: Expr, else_: Expr) -> Self {
        Self::If(Rc::new(cond), Rc::new(then), Rc::new(else_))
    }

    #[cfg(test)]
    pub fn let_(name: &str, expr: Expr, body: Expr) -> Self {
        Self::Let(name.to_string(), Rc::new(expr), Rc::new(body))
    }

    pub fn lambda(param: &str, body: Expr) -> Self {
        Self::Lambda(param.to_string(), Rc::new(body))
    }

    pub fn ap(fun: Expr, arg: Expr) -> Self {
        Self::Ap(Rc::new(fun), Rc::new(arg))
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Int(n) => write!(f, "{}", n),
            Expr::Bool(b) => write!(f, "{}", b),
            Expr::Ident(s) => write!(f, "{}", s),
            Expr::If(cond, then, else_) => {
                write!(f, "(if {} then {} else {})", cond, then, else_)
            }
            Expr::Lambda(param, body) => write!(f, "(fun {} -> {})", param, body),
            Expr::Ap(fun, arg) => write!(f, "({} {})", fun, arg),
            Expr::Let(name, expr, body) => write!(f, "(let {} = {} in {})", name, expr, body),
            Expr::Match(expr, patterns) => write!(
                f,
                "(match {} with {})",
                expr,
                patterns
                    .iter()
                    .map(|(pat, body)| format!("| {} -> {}", pat, body))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypeExpr {
    Int,
    Bool,
    Fun(Rc<TypeExpr>, Rc<TypeExpr>),
    TypeVar(String),
    Forall(RedBlackTreeSet<String>, Rc<TypeExpr>),
    Constructor(String, Vec<Rc<TypeExpr>>),
}

impl TypeExpr {
    pub fn fun(param: TypeExpr, body: TypeExpr) -> Self {
        assert!(!param.is_scheme() && !body.is_scheme());
        Self::Fun(Rc::new(param), Rc::new(body))
    }

    pub fn type_var(name: &str) -> Self {
        Self::TypeVar(name.to_string())
    }

    pub fn constructor(name: &str, arguments: Vec<TypeExpr>) -> Self {
        Self::Constructor(
            name.to_string(),
            arguments.into_iter().map(Rc::new).collect(),
        )
    }

    pub fn forall(vars: RedBlackTreeSet<String>, ty: TypeExpr) -> Self {
        assert!(!ty.is_scheme());
        Self::Forall(vars, Rc::new(ty))
    }

    pub fn is_scheme(&self) -> bool {
        match self {
            TypeExpr::Forall(_, _) => true,
            _ => false,
        }
    }
}

impl Display for TypeExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeExpr::Int => write!(f, "int"),
            TypeExpr::Bool => write!(f, "bool"),
            TypeExpr::Fun(param, body) => write!(f, "({} -> {})", param, body),
            TypeExpr::TypeVar(s) => write!(f, "'{}", s),
            TypeExpr::Forall(vars, ty) => {
                if vars.is_empty() {
                    write!(f, "{}", ty)
                } else {
                    write!(
                        f,
                        "forall {} . {}",
                        vars.into_iter().map(|v| format!("'{}", v)).join(" "),
                        ty
                    )
                }
            }
            TypeExpr::Constructor(name, args) => {
                if args.is_empty() {
                    write!(f, "{}", name)
                } else {
                    write!(
                        f,
                        "({} {})",
                        name,
                        args.iter()
                            .map(|x| x.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    )
                }
            }
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    Let(String, Rc<Expr>),
    LetRec(String, Rc<Expr>),
    Val(String, Rc<TypeExpr>),
    Data(String, Vec<String>, Vec<TypeExpr>),
}

impl Statement {
    pub fn let_(name: &str, expr: Expr) -> Self {
        Self::Let(name.to_string(), Rc::new(expr))
    }

    #[cfg(test)]
    pub fn val(name: &str, ty: TypeExpr) -> Self {
        Self::Val(name.to_string(), Rc::new(ty))
    }
}

impl Display for Statement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Statement::Let(name, expr) => write!(f, "let {} = {}", name, expr),
            Statement::LetRec(name, expr) => write!(f, "let rec {} = {}", name, expr),
            Statement::Val(name, ty) => write!(f, "val {} : {}", name, ty),
            Statement::Data(name, args, variants) => {
                write!(
                    f,
                    "data {} {} = {}",
                    name,
                    args.iter().map(|arg| format!("'{}", arg)).join(" "),
                    variants.iter().join(" | ")
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_expr() {
        let expr = Expr::ap(
            Expr::lambda(
                "x",
                Expr::ap(
                    Expr::lambda("y", Expr::ap(Expr::ident("x"), Expr::ident("y"))),
                    Expr::int(1),
                ),
            ),
            Expr::int(2),
        );
        assert_eq!(format!("{}", expr), "((fun x -> ((fun y -> (x y)) 1)) 2)");
    }

    #[test]
    fn display_expr_if() {
        let expr = Expr::if_(Expr::bool(true), Expr::int(1), Expr::int(2));
        assert_eq!(format!("{}", expr), "(if true then 1 else 2)");
    }

    #[test]
    fn display_type_expr() {
        let ty = TypeExpr::fun(
            TypeExpr::fun(TypeExpr::Int, TypeExpr::type_var("a")),
            TypeExpr::Bool,
        );
        assert_eq!(format!("{}", ty), "((int -> 'a) -> bool)");
    }

    #[test]
    fn display_let_statement() {
        let stmt = Statement::let_(
            "f",
            Expr::lambda("x", Expr::ap(Expr::ident("f"), Expr::ident("x"))),
        );
        assert_eq!(format!("{}", stmt), "let f = (fun x -> (f x))");
    }

    #[test]
    fn display_val_statement() {
        let stmt = Statement::val(
            "f",
            TypeExpr::fun(TypeExpr::type_var("a"), TypeExpr::type_var("a")),
        );
        assert_eq!(format!("{}", stmt), "val f : ('a -> 'a)");
    }

    #[test]
    fn free_variables() {
        let ty = TypeExpr::fun(
            TypeExpr::fun(TypeExpr::Int, TypeExpr::type_var("a")),
            TypeExpr::type_var("b"),
        );
        assert_eq!(
            ty.free_variables(),
            RedBlackTreeSet::new()
                .insert("a".to_string())
                .insert("b".to_string())
        );

        let ty = TypeExpr::forall(
            RedBlackTreeSet::new().insert("a".to_string()),
            TypeExpr::fun(TypeExpr::type_var("a"), TypeExpr::type_var("b")),
        );

        assert_eq!(
            ty.free_variables(),
            RedBlackTreeSet::new().insert("b".to_string())
        );
    }
}
