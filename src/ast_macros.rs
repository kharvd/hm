#[macro_export]
macro_rules! e_ident {
    ($x:expr) => {
        Expr::Ident($x.to_string())
    };
}

#[macro_export]
macro_rules! e_int {
    ($x:expr) => {
        Expr::Int($x)
    };
}

#[macro_export]
macro_rules! e_bool {
    ($x:expr) => {
        Expr::Bool($x)
    };
}

#[macro_export]
macro_rules! e_char {
    ($x:expr) => {
        Expr::Char($x)
    };
}

#[macro_export]
macro_rules! e_ap {
    ($f:expr, $x:expr) => {
        Expr::Ap(Rc::new($f), Rc::new($x))
    };
}

#[macro_export]
macro_rules! e_lambda {
    ($x:expr, $body:expr) => {
        Expr::Lambda($x.to_string(), Rc::new($body))
    };
}

#[macro_export]
macro_rules! e_if {
    ($cond:expr, $then:expr, $else_:expr) => {
        Expr::If(Rc::new($cond), Rc::new($then), Rc::new($else_))
    };
}

#[macro_export]
macro_rules! e_let {
    ($name:expr, $expr:expr, $body:expr) => {
        Expr::Let($name.to_string(), Rc::new($expr), Rc::new($body))
    };
}

#[macro_export]
macro_rules! e_match_case {
    ($pat:expr, $guard:expr, $body:expr) => {
        MatchCase {
            pattern: Rc::new($pat),
            guard: $guard.map(Rc::new),
            body: Rc::new($body),
        }
    };
}

#[macro_export]
macro_rules! e_match {
    ($expr:expr, $cases:expr) => {
        Expr::Match(Rc::new($expr), $cases.into_iter().collect())
    };
}

#[macro_export]
macro_rules! t_int {
    () => {
        TypeExpr::Int
    };
}

#[macro_export]
macro_rules! t_bool {
    () => {
        TypeExpr::Bool
    };
}

#[macro_export]
macro_rules! t_fun {
    ($param:expr, $body:expr) => {
        TypeExpr::Fun(Rc::new($param), Rc::new($body))
    };
}

#[macro_export]
macro_rules! t_type_var {
    ($x:expr) => {
        TypeExpr::TypeVar($x.to_string())
    };
}

#[macro_export]
macro_rules! t_forall {
    ($vars:expr, $body:expr) => {
        TypeExpr::Forall(
            RedBlackTreeSet::from_iter($vars.iter().map(|x| x.to_string())),
            Rc::new($body),
        )
    };
}

#[macro_export]
macro_rules! t_constructor {
    ($name:expr, $($args:expr),*) => {
        TypeExpr::Constructor($name.to_string(), vec![$(Rc::new($args)),*])
    };

    ($name:expr) => {
        TypeExpr::Constructor($name.to_string(), vec![])
    };
}

#[macro_export]
macro_rules! p_int {
    ($x:expr) => {
        ExprPattern::Int($x)
    };
}

#[macro_export]
macro_rules! p_bool {
    ($x:expr) => {
        ExprPattern::Bool($x)
    };
}

#[macro_export]
macro_rules! p_var {
    ($x:expr) => {
        ExprPattern::Variable($x.to_string())
    };
}

#[macro_export]
macro_rules! p_wildcard {
    () => {
        ExprPattern::Wildcard
    };
}

#[macro_export]
macro_rules! p_constructor {
    ($name:expr, $($args:expr),*) => {
        ExprPattern::Constructor($name.to_string(), vec![$(Rc::new($args)),*])
    };

    ($name:expr) => {
        ExprPattern::Constructor($name.to_string(), vec![])
    };
}
