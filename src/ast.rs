use std::rc::Rc;

use rpds::{List, RedBlackTreeSet};

#[derive(Debug, PartialEq, Clone)]
pub enum ExprPattern {
    Int(i64),
    Bool(bool),
    Char(char),
    Variable(String),
    Wildcard,
    Constructor(String, Vec<Rc<ExprPattern>>),
}

#[derive(Debug, PartialEq, Clone)]
pub struct MatchCase {
    pub pattern: Rc<ExprPattern>,
    pub guard: Option<Rc<Expr>>,
    pub body: Rc<Expr>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Int(i64),
    Bool(bool),
    Char(char),
    Ident(String),
    If(Rc<Expr>, Rc<Expr>, Rc<Expr>),
    Let(String, Rc<Expr>, Rc<Expr>),
    Lambda(String, Rc<Expr>),
    Ap(Rc<Expr>, Rc<Expr>),
    Match(Rc<Expr>, Vec<MatchCase>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypeExpr {
    Int,
    Bool,
    Char,
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

    pub fn as_function_type(&self) -> (List<Rc<TypeExpr>>, TypeExpr) {
        match self {
            TypeExpr::Fun(param, body) => {
                let (params, return_type) = body.as_function_type();
                (params.push_front(param.clone()), return_type)
            }
            _ => (List::new(), self.clone()),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    Let(String, Rc<Expr>),
    Val(String, Rc<TypeExpr>),
    Data(String, Vec<String>, Vec<TypeExpr>),
}

#[cfg(test)]
mod tests {
    use crate::{t_forall, t_fun, t_int, t_type_var};

    use super::*;

    #[test]
    fn free_variables() {
        let ty = t_fun!(t_fun!(t_int!(), t_type_var!("a")), t_type_var!("b"));
        assert_eq!(
            ty.free_variables(),
            RedBlackTreeSet::new()
                .insert("a".to_string())
                .insert("b".to_string())
        );

        let ty = t_forall!(["a"], t_fun!(t_type_var!("a"), t_type_var!("b")));

        assert_eq!(
            ty.free_variables(),
            RedBlackTreeSet::new().insert("b".to_string())
        );
    }
}
