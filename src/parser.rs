use std::iter::Peekable;
use std::rc::Rc;

use rpds::RedBlackTreeSet;

use crate::ast::{Expr, Statement, TypeExpr};
use crate::lexer::{tokenize, Keyword, Token};

#[derive(Debug)]
pub enum ParseResult {
    Statement(Statement),
    Expression(Expr),
}

pub fn parse(s: &str) -> Result<ParseResult, String> {
    let tokens = tokenize(s)?;
    let mut iter = tokens.into_iter().peekable();
    match iter.peek() {
        Some(token) => Ok(match token {
            Token::Keyword(Keyword::Let)
            | Token::Keyword(Keyword::Val)
            | Token::Keyword(Keyword::Data) => ParseResult::Statement(parse_stmt(&mut iter)?),
            _ => ParseResult::Expression(parse_expr(&mut iter)?),
        }),
        None => Err("Empty input".to_string()),
    }
}

pub fn parse_statement(s: &str) -> Result<Statement, String> {
    let tokens = tokenize(s)?;
    let mut iter = tokens.into_iter().peekable();
    parse_stmt(&mut iter)
}

pub fn parse_stmt(tokens: &mut Peekable<impl Iterator<Item = Token>>) -> Result<Statement, String> {
    let statement = match tokens.peek() {
        Some(Token::Keyword(Keyword::Let)) => parse_let_statement(tokens)?,
        Some(Token::Keyword(Keyword::Val)) => parse_val_statement(tokens)?,
        Some(Token::Keyword(Keyword::Data)) => parse_data_statement(tokens)?,
        _ => return Err("Expected statement".to_string()),
    };

    Ok(statement)
}

fn parse_val_statement(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<Statement, String> {
    tokens.next();
    let name = match tokens.next() {
        Some(Token::Ident(name)) => name,
        _ => return Err("Expected identifier after 'val'".to_string()),
    };
    match tokens.next() {
        Some(Token::Colon) => (),
        _ => return Err("Expected ':' after identifier".to_string()),
    }
    let ty = parse_type_expr(tokens)?;
    Ok(Statement::Val(name, Rc::new(ty)))
}

fn parse_let_statement(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<Statement, String> {
    tokens.next();

    match tokens.peek() {
        Some(Token::Keyword(Keyword::Rec)) => parse_let_rec_statement(tokens),
        Some(Token::Ident(_)) => {
            let (name, expr) = parse_let_name_binding(tokens)?;
            Ok(Statement::Let(name, expr))
        }
        _ => return Err("Expected identifier or 'rec' after 'let'".to_string()),
    }
}

fn parse_type_var(tokens: &mut Peekable<impl Iterator<Item = Token>>) -> Result<String, String> {
    match tokens.next() {
        Some(Token::Ident(name)) => Ok(name),
        _ => Err("Expected identifier after apostrophe".to_string()),
    }
}

fn parse_data_statement(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<Statement, String> {
    tokens.next();
    let name = match tokens.next() {
        Some(Token::Ident(name)) => name,
        _ => return Err("Expected identifier after 'data'".to_string()),
    };

    let mut args = Vec::new();
    loop {
        match tokens.peek() {
            Some(Token::Apostrophe) => {
                tokens.next();
                let var_name = parse_type_var(tokens)?;
                args.push(var_name);
            }
            _ => break,
        }
    }

    match tokens.next() {
        Some(Token::Equals) => (),
        _ => return Err("Expected '=' after identifier".to_string()),
    }
    let variants = parse_data_variants(tokens)?;
    Ok(Statement::Data(name, args, variants))
}

fn parse_data_variants(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<Vec<TypeExpr>, String> {
    let mut variants = Vec::new();
    loop {
        let variant = match tokens.next() {
            Some(Token::Ident(name)) => parse_type_constructor(name, tokens)?,
            _ => return Err("Expected identifier".to_string()),
        };
        variants.push(variant);
        match tokens.peek() {
            Some(Token::Pipe) => {
                tokens.next();
            }
            _ => break,
        }
    }
    Ok(variants)
}

fn parse_let_rec_statement(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<Statement, String> {
    tokens.next();

    let (name, expr) = parse_let_name_binding(tokens)?;
    Ok(Statement::LetRec(name, expr))
}

fn parse_let_name_binding(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<(String, Rc<Expr>), String> {
    let name = match tokens.next() {
        Some(Token::Ident(name)) => name,
        _ => return Err("Expected identifier after 'let'".to_string()),
    };

    match tokens.next() {
        Some(Token::Equals) => (),
        _ => return Err("Expected '=' after identifier".to_string()),
    }
    let expr = parse_expr(tokens)?;
    Ok((name, Rc::new(expr)))
}

pub fn parse_type_expr(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<TypeExpr, String> {
    match tokens.peek() {
        Some(Token::Keyword(Keyword::Forall)) => parse_forall_type_expr(tokens),
        _ => parse_arrow_type_expr(tokens),
    }
}

fn parse_forall_type_expr(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<TypeExpr, String> {
    tokens.next();

    let mut vars = RedBlackTreeSet::new();
    loop {
        match tokens.next() {
            Some(Token::Apostrophe) => {
                let var_name = parse_type_var(tokens)?;
                vars = vars.insert(var_name);
            }
            Some(Token::Dot) => break,
            t => return Err(format!("Expected apostrophe or dot, but got {:?}", t)),
        }
    }

    let ty = parse_arrow_type_expr(tokens)?;

    Ok(TypeExpr::Forall(vars, Rc::new(ty)))
}

pub fn parse_arrow_type_expr(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<TypeExpr, String> {
    let mut ty = parse_non_arrow_expr(tokens)?;

    if let Some(Token::Arrow) = tokens.peek() {
        tokens.next();
        ty = TypeExpr::fun(ty, parse_arrow_type_expr(tokens)?);
    }

    Ok(ty)
}

fn parse_non_arrow_expr(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<TypeExpr, String> {
    let ty = match tokens.next() {
        Some(Token::Apostrophe) => TypeExpr::TypeVar(parse_type_var(tokens)?),
        Some(Token::Ident(name)) => parse_type_constructor(name, tokens)?,
        Some(Token::Keyword(Keyword::Int)) => TypeExpr::Int,
        Some(Token::Keyword(Keyword::Bool)) => TypeExpr::Bool,
        Some(Token::LParen) => {
            let ty = parse_arrow_type_expr(tokens)?;
            match tokens.next() {
                Some(Token::RParen) => ty,
                _ => return Err("Expected closing parenthesis".to_string()),
            }
        }
        _ => return Err("Expected type expression".to_string()),
    };

    Ok(ty)
}

fn parse_type_constructor(
    name: String,
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<TypeExpr, String> {
    let mut args = Vec::new();
    loop {
        match tokens.peek() {
            Some(Token::Apostrophe)
            | Some(Token::LParen)
            | Some(Token::Ident(_))
            | Some(Token::Keyword(Keyword::Bool))
            | Some(Token::Keyword(Keyword::Int)) => {
                let arg = Rc::new(parse_non_arrow_expr(tokens)?);
                args.push(arg);
            }
            _ => break,
        }
    }

    Ok(TypeExpr::Constructor(name, args))
}

fn parse_expr(tokens: &mut Peekable<impl Iterator<Item = Token>>) -> Result<Expr, String> {
    let mut expr = parse_non_ap_expr(tokens)?;

    while let Some(next) = tokens.peek() {
        match next {
            Token::LParen
            | Token::Ident(_)
            | Token::Int(_)
            | Token::Underscore
            | Token::Keyword(Keyword::True)
            | Token::Keyword(Keyword::False) => {
                expr = Expr::Ap(Rc::new(expr), Rc::new(parse_non_ap_expr(tokens)?));
            }
            _ => break,
        }
    }

    Ok(expr)
}

fn parse_non_ap_expr(tokens: &mut Peekable<impl Iterator<Item = Token>>) -> Result<Expr, String> {
    let expr = match tokens.next() {
        Some(Token::Ident(name)) => Expr::Ident(name),
        Some(Token::Int(i)) => Expr::Int(i),
        Some(Token::Keyword(Keyword::True)) => Expr::Bool(true),
        Some(Token::Keyword(Keyword::False)) => Expr::Bool(false),
        Some(Token::Underscore) => Expr::Placeholder,
        Some(Token::Keyword(Keyword::If)) => {
            let cond = Rc::new(parse_expr(tokens)?);
            match tokens.next() {
                Some(Token::Keyword(Keyword::Then)) => (),
                _ => return Err("Expected 'then' after 'if'".to_string()),
            }
            let then = Rc::new(parse_expr(tokens)?);
            match tokens.next() {
                Some(Token::Keyword(Keyword::Else)) => (),
                _ => return Err("Expected 'else' after 'then'".to_string()),
            }
            let else_ = Rc::new(parse_expr(tokens)?);
            Expr::If(cond, then, else_)
        }
        Some(Token::Keyword(Keyword::Fun)) => {
            let param = match tokens.next() {
                Some(Token::Ident(name)) => name,
                _ => return Err("Expected identifier after 'fun'".to_string()),
            };
            match tokens.next() {
                Some(Token::Arrow) => (),
                _ => return Err("Expected '->' after parameter".to_string()),
            }
            let body = Rc::new(parse_expr(tokens)?);
            Expr::Lambda(param, body)
        }
        Some(Token::Keyword(Keyword::Let)) => {
            let (name, bound_expr, expr) = parse_let_expr(tokens)?;
            Expr::Let(name, Rc::new(bound_expr), Rc::new(expr))
        }
        Some(Token::LParen) => {
            let expr = parse_expr(tokens)?;
            match tokens.next() {
                Some(Token::RParen) => expr,
                _ => return Err("Expected closing parenthesis".to_string()),
            }
        }
        _ => return Err("Expected expression".to_string()),
    };

    Ok(expr)
}

fn parse_let_expr(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<(String, Expr, Expr), String> {
    let name = match tokens.next() {
        Some(Token::Ident(name)) => name,
        _ => return Err("Expected identifier after 'let'".to_string()),
    };

    match tokens.next() {
        Some(Token::Equals) => (),
        _ => return Err("Expected '=' after identifier".to_string()),
    }

    let bound_expr = parse_expr(tokens)?;

    match tokens.next() {
        Some(Token::Keyword(Keyword::In)) => (),
        _ => return Err("Expected 'in' after bound expression".to_string()),
    }

    let expr = parse_expr(tokens)?;

    Ok((name, bound_expr, expr))
}

#[cfg(test)]
mod tests {
    use crate::lexer;

    use super::*;

    #[test]
    fn test_simple_type() {
        let tokens = lexer::tokenize("int -> bool").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_type_expr(&mut iter),
            Ok(TypeExpr::fun(TypeExpr::Int, TypeExpr::Bool))
        );
    }

    #[test]
    fn test_type_constructor_simple() {
        let tokens = lexer::tokenize("MyType -> bool").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_type_expr(&mut iter),
            Ok(TypeExpr::fun(
                TypeExpr::Constructor("MyType".to_string(), vec![]),
                TypeExpr::Bool
            ))
        );
    }

    #[test]
    fn test_type_constructor_args() {
        let tokens = lexer::tokenize("int -> MyType 'a (MyOtherType -> 'b) -> bool").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_type_expr(&mut iter),
            Ok(TypeExpr::fun(
                TypeExpr::Int,
                TypeExpr::fun(
                    TypeExpr::constructor(
                        "MyType",
                        vec![
                            TypeExpr::TypeVar("a".to_string()),
                            TypeExpr::fun(
                                TypeExpr::constructor("MyOtherType", vec![]),
                                TypeExpr::type_var("b")
                            )
                        ]
                    ),
                    TypeExpr::Bool
                )
            ))
        );
    }

    #[test]
    fn test_arrow_left_assoc() {
        let tokens = lexer::tokenize("int -> bool -> int").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_arrow_type_expr(&mut iter),
            Ok(TypeExpr::fun(
                TypeExpr::Int,
                TypeExpr::fun(TypeExpr::Bool, TypeExpr::Int)
            ))
        );
    }

    #[test]
    fn test_type_var() {
        let tokens = lexer::tokenize("('a -> 'b) -> 'a").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_arrow_type_expr(&mut iter),
            Ok(TypeExpr::fun(
                TypeExpr::fun(TypeExpr::type_var("a"), TypeExpr::type_var("b"),),
                TypeExpr::type_var("a")
            ))
        );
    }

    #[test]
    fn test_simple_expr() {
        let tokens = lexer::tokenize("fun x -> (fun y -> plus x y)").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_expr(&mut iter),
            Ok(Expr::lambda(
                "x",
                Expr::lambda(
                    "y",
                    Expr::ap(
                        Expr::ap(Expr::ident("plus"), Expr::ident("x")),
                        Expr::ident("y")
                    )
                )
            ))
        );
    }

    #[test]
    fn test_if_expr() {
        let tokens = lexer::tokenize("if true then mul x y else (fun x -> x) x").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_expr(&mut iter),
            Ok(Expr::if_(
                Expr::bool(true),
                Expr::ap(
                    Expr::ap(Expr::ident("mul"), Expr::ident("x"),),
                    Expr::ident("y"),
                ),
                Expr::ap(Expr::lambda("x", Expr::ident("x"),), Expr::ident("x"))
            ))
        );
    }

    #[test]
    fn test_let_statement() {
        let tokens = lexer::tokenize("let f = fun x -> x").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_stmt(&mut iter),
            Ok(Statement::let_("f", Expr::lambda("x", Expr::ident("x"))))
        );
    }

    #[test]
    fn test_let_expression() {
        let tokens = lexer::tokenize("let x = f y in x 5").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_expr(&mut iter),
            Ok(Expr::let_(
                "x",
                Expr::ap(Expr::ident("f"), Expr::ident("y")),
                Expr::ap(Expr::ident("x"), Expr::int(5))
            ))
        );
    }

    #[test]
    fn test_type_scheme() {
        let tokens = lexer::tokenize("forall 'a 'b . 'a -> 'b").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_type_expr(&mut iter),
            Ok(TypeExpr::forall(
                RedBlackTreeSet::new()
                    .insert("a".to_string())
                    .insert("b".to_string()),
                TypeExpr::fun(TypeExpr::type_var("a"), TypeExpr::type_var("b"))
            ))
        );
    }

    #[test]
    fn test_data_simple() {
        let tokens = lexer::tokenize("data MyType = A | BB | Ccc").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_stmt(&mut iter),
            Ok(Statement::Data(
                "MyType".to_string(),
                vec![],
                vec![
                    TypeExpr::constructor("A", vec![]),
                    TypeExpr::constructor("BB", vec![]),
                    TypeExpr::constructor("Ccc", vec![]),
                ]
            ))
        );
    }

    #[test]
    fn test_data_with_args() {
        let tokens = lexer::tokenize("data MyType 'a 'b = A | BB 'a bool | Ccc 'b").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_stmt(&mut iter),
            Ok(Statement::Data(
                "MyType".to_string(),
                vec!["a".to_string(), "b".to_string()],
                vec![
                    TypeExpr::constructor("A", vec![]),
                    TypeExpr::constructor("BB", vec![TypeExpr::type_var("a"), TypeExpr::Bool]),
                    TypeExpr::constructor("Ccc", vec![TypeExpr::type_var("b")]),
                ]
            ))
        );
    }
}

/*
   match_expr ::= match expr_0 with | pat_1 -> expr_1 | pat_2 -> expr_2 | ...  | pat_n -> expr_n
*/
