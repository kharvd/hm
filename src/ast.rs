use std::{fmt::Display, rc::Rc};

#[derive(Debug, PartialEq)]
pub enum Expr {
    Int(i64),
    Bool(bool),
    Ident(String),
    If(Rc<Expr>, Rc<Expr>, Rc<Expr>),
    Lambda(String, Rc<Expr>),
    Ap(Rc<Expr>, Rc<Expr>),
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
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum TypeExpr {
    Int,
    Bool,
    Fun(Rc<TypeExpr>, Rc<TypeExpr>),
    TypeVar(String),
}

impl Display for TypeExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeExpr::Int => write!(f, "int"),
            TypeExpr::Bool => write!(f, "bool"),
            TypeExpr::Fun(param, body) => write!(f, "({} -> {})", param, body),
            TypeExpr::TypeVar(s) => write!(f, "'{}", s),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Statement {
    Let(String, Expr),
    Val(String, TypeExpr),
}

impl Display for Statement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Statement::Let(name, expr) => write!(f, "let {} = {}", name, expr),
            Statement::Val(name, ty) => write!(f, "val {} : {}", name, ty),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_expr() {
        let expr = Expr::Ap(
            Rc::new(Expr::Lambda(
                "x".to_string(),
                Rc::new(Expr::Ap(
                    Rc::new(Expr::Lambda(
                        "y".to_string(),
                        Rc::new(Expr::Ap(
                            Rc::new(Expr::Ident("x".to_string())),
                            Rc::new(Expr::Ident("y".to_string())),
                        )),
                    )),
                    Rc::new(Expr::Int(1)),
                )),
            )),
            Rc::new(Expr::Int(2)),
        );
        assert_eq!(format!("{}", expr), "((fun x -> ((fun y -> (x y)) 1)) 2)");
    }

    #[test]
    fn display_expr_if() {
        let expr = Expr::If(
            Rc::new(Expr::Bool(true)),
            Rc::new(Expr::Int(1)),
            Rc::new(Expr::Int(2)),
        );
        assert_eq!(format!("{}", expr), "(if true then 1 else 2)");
    }

    #[test]
    fn display_type_expr() {
        let ty = TypeExpr::Fun(
            Rc::new(TypeExpr::Fun(
                Rc::new(TypeExpr::Int),
                Rc::new(TypeExpr::TypeVar("a".to_string())),
            )),
            Rc::new(TypeExpr::Bool),
        );
        assert_eq!(format!("{}", ty), "((int -> 'a) -> bool)");
    }

    #[test]
    fn display_let_statement() {
        let stmt = Statement::Let(
            "f".to_string(),
            Expr::Lambda(
                "x".to_string(),
                Rc::new(Expr::Ap(
                    Rc::new(Expr::Ident("f".to_string())),
                    Rc::new(Expr::Ident("x".to_string())),
                )),
            ),
        );
        assert_eq!(format!("{}", stmt), "let f = (fun x -> (f x))");
    }

    #[test]
    fn display_val_statement() {
        let stmt = Statement::Val(
            "f".to_string(),
            TypeExpr::Fun(
                Rc::new(TypeExpr::TypeVar("a".to_string())),
                Rc::new(TypeExpr::TypeVar("a".to_string())),
            ),
        );
        assert_eq!(format!("{}", stmt), "val f : ('a -> 'a)");
    }
}