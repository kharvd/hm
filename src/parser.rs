use std::iter::Peekable;
use std::rc::Rc;

use rpds::RedBlackTreeSet;

use crate::ast::{Expr, ExprPattern, MatchCase, Statement, TypeExpr};
use crate::lexer::{tokenize, InfixOp, Keyword, Token};

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
            | Token::Keyword(Keyword::Data) => ParseResult::Statement(parse_statement(&mut iter)?),
            _ => ParseResult::Expression(parse_expr(&mut iter)?),
        }),
        None => Err("Empty input".to_string()),
    }
}

pub fn parse_file(file: &str) -> Result<Vec<Statement>, String> {
    let tokens = tokenize(file)?;
    let mut iter = tokens.into_iter().peekable();
    let mut statements = Vec::new();
    while iter.peek().is_some() {
        statements.push(parse_statement(&mut iter)?);
    }
    Ok(statements)
}

pub fn parse_statement(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<Statement, String> {
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
    let name = parse_binding_name(tokens)?;
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
        Some(Token::Variable(_)) | Some(Token::LParen) => {
            let (name, expr) = parse_let_name_binding(tokens)?;
            Ok(Statement::Let(name, expr))
        }
        _ => return Err("Expected identifier or 'rec' after 'let'".to_string()),
    }
}

fn parse_data_statement(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<Statement, String> {
    tokens.next();
    let name = match tokens.next() {
        Some(Token::Constructor(name)) => name,
        _ => return Err("Expected capitalized identifier after 'data'".to_string()),
    };

    let mut args = Vec::new();
    loop {
        match tokens.peek() {
            Some(Token::Variable(_)) => {
                let Some(Token::Variable(var_name)) = tokens.next() else {
                    panic!("Expected variable name");
                };
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
            Some(Token::Constructor(name)) => parse_type_constructor(name, tokens)?,
            t => return Err(format!("Expected constructor name, but got {:?}", t)),
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
    let name = parse_binding_name(tokens)?;

    match tokens.next() {
        Some(Token::Equals) => (),
        _ => return Err("Expected '=' after identifier".to_string()),
    }
    let expr = parse_expr(tokens)?;
    Ok((name, Rc::new(expr)))
}

fn parse_binding_name(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<String, String> {
    match tokens.next() {
        Some(Token::Variable(name)) => Ok(name),
        Some(Token::LParen) => {
            let name = match tokens.next() {
                Some(Token::InfixOp(op)) => op.to_string().to_string(),
                _ => return Err("Expected identifier after 'let' or 'val'".to_string()),
            };
            match tokens.next() {
                Some(Token::RParen) => (),
                _ => return Err("Expected closing parenthesis".to_string()),
            }
            Ok(name)
        }
        _ => return Err("Expected identifier after 'let' or 'val'".to_string()),
    }
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
            Some(Token::Variable(var_name)) => {
                vars = vars.insert(var_name);
            }
            Some(Token::Dot) => break,
            t => return Err(format!("Expected variable or dot, but got {:?}", t)),
        }
    }

    let ty = parse_arrow_type_expr(tokens)?;

    Ok(TypeExpr::Forall(vars, Rc::new(ty)))
}

pub fn parse_arrow_type_expr(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<TypeExpr, String> {
    let mut ty = parse_non_arrow_type_expr(tokens)?;

    if let Some(Token::Arrow) = tokens.peek() {
        tokens.next();
        ty = TypeExpr::fun(ty, parse_arrow_type_expr(tokens)?);
    }

    Ok(ty)
}

fn parse_non_arrow_type_expr(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<TypeExpr, String> {
    let ty = match tokens.next() {
        Some(Token::Variable(name)) => TypeExpr::TypeVar(name),
        Some(Token::Constructor(name)) => parse_type_constructor(name, tokens)?,
        Some(Token::Keyword(Keyword::Int)) => TypeExpr::Int,
        Some(Token::Keyword(Keyword::Bool)) => TypeExpr::Bool,
        Some(Token::Keyword(Keyword::Char)) => TypeExpr::Char,
        Some(Token::LParen) => {
            let ty = parse_arrow_type_expr(tokens)?;
            match tokens.next() {
                Some(Token::RParen) => ty,
                _ => return Err("Expected closing parenthesis".to_string()),
            }
        }
        t => return Err(format!("Expected type expression, got {:?}", t)),
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
            Some(Token::LParen)
            | Some(Token::Variable(_))
            | Some(Token::Constructor(_))
            | Some(Token::Keyword(Keyword::Bool))
            | Some(Token::Keyword(Keyword::Int))
            | Some(Token::Keyword(Keyword::Char)) => {
                let arg = Rc::new(parse_non_arrow_type_expr(tokens)?);
                args.push(arg);
            }
            _ => break,
        }
    }

    Ok(TypeExpr::Constructor(name, args))
}

impl InfixOp {
    fn precedence(&self) -> i32 {
        match self {
            InfixOp::Mult | InfixOp::Div | InfixOp::Mod => 20,
            InfixOp::Plus | InfixOp::Minus => 10,
            InfixOp::LessThan
            | InfixOp::GreaterThan
            | InfixOp::Equals
            | InfixOp::NotEquals
            | InfixOp::LessEquals
            | InfixOp::GreaterEquals => 5,
            InfixOp::And => 3,
            InfixOp::Or => 2,
        }
    }
}

fn parse_expr(tokens: &mut Peekable<impl Iterator<Item = Token>>) -> Result<Expr, String> {
    parse_expr_prec(tokens, 0)
}

fn parse_expr_prec(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
    precedence: i32,
) -> Result<Expr, String> {
    let mut lhs = parse_primary(tokens)?;

    while let Some(op) = tokens.peek() {
        let op = (*op).clone();

        match op {
            Token::InfixOp(op) => {
                let op_precedence = op.precedence();
                if op_precedence < precedence {
                    break;
                }
                tokens.next();

                let rhs = parse_expr_prec(tokens, op_precedence + 1)?;

                lhs = make_infix_op(op, lhs, rhs);
            }
            Token::RParen
            | Token::RBracket
            | Token::Keyword(Keyword::Then)
            | Token::Keyword(Keyword::Else)
            | Token::Keyword(Keyword::In)
            | Token::Keyword(Keyword::With)
            | Token::Keyword(Keyword::Let)
            | Token::Keyword(Keyword::Data)
            | Token::Keyword(Keyword::Val)
            | Token::Comma
            | Token::Arrow
            | Token::Pipe => break,
            _ => {
                lhs = Expr::Ap(Rc::new(lhs), Rc::new(parse_primary(tokens)?));
            }
        }
    }

    Ok(lhs)
}

fn make_infix_op(op: InfixOp, lhs: Expr, rhs: Expr) -> Expr {
    Expr::Ap(
        Rc::new(Expr::Ap(
            Rc::new(Expr::Ident(op.to_string().to_string())),
            Rc::new(lhs),
        )),
        Rc::new(rhs),
    )
}

fn parse_primary(tokens: &mut Peekable<impl Iterator<Item = Token>>) -> Result<Expr, String> {
    Ok(match tokens.next() {
        Some(Token::Variable(name)) => Expr::Ident(name),
        Some(Token::InfixOp(op)) => Expr::Ident(op.to_string().to_string()),
        Some(Token::Constructor(name)) => Expr::Ident(name),
        Some(Token::Int(i)) => Expr::Int(i),
        Some(Token::Char(c)) => Expr::Char(c),
        Some(Token::String(s)) => make_char_list(s),
        Some(Token::Keyword(Keyword::True)) => Expr::Bool(true),
        Some(Token::Keyword(Keyword::False)) => Expr::Bool(false),
        Some(Token::Keyword(Keyword::If)) => parse_if_expr(tokens)?,
        Some(Token::Keyword(Keyword::Fun)) => parse_lambda_expr(tokens)?,
        Some(Token::Keyword(Keyword::Let)) => parse_let_expr(tokens)?,
        Some(Token::Keyword(Keyword::Match)) => parse_match_expr(tokens)?,
        Some(Token::LParen) => parse_parenthesized_expr(tokens)?,
        Some(Token::LBracket) => parse_list(tokens)?,
        t => return Err(format!("Expected primary expression, got {:?}", t)),
    })
}

fn make_char_list(s: String) -> Expr {
    let mut expr = Expr::Ident("Nil".to_string());
    for c in s.chars().rev() {
        expr = Expr::Ap(
            Rc::new(Expr::Ap(
                Rc::new(Expr::Ident("Cons".to_string())),
                Rc::new(Expr::Char(c)),
            )),
            Rc::new(expr),
        );
    }
    expr
}

fn parse_parenthesized_expr(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<Expr, String> {
    match tokens.peek() {
        Some(Token::RParen) => {
            tokens.next();
            Ok(Expr::Ident("Unit".to_string()))
        }
        _ => {
            let expr = parse_expr(tokens)?;
            Ok(match tokens.next() {
                Some(Token::RParen) => expr,
                Some(Token::Comma) => parse_tuple(tokens, expr)?,
                _ => return Err("Expected closing parenthesis".to_string()),
            })
        }
    }
}

fn parse_tuple(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
    first: Expr,
) -> Result<Expr, String> {
    let mut exprs = vec![first];
    exprs.append(&mut parse_comma_exprs(tokens)?);

    let tuple = Expr::Ident(format!("Tuple{}", exprs.len()));
    let tuple = exprs
        .into_iter()
        .fold(tuple, |acc, expr| Expr::Ap(Rc::new(acc), Rc::new(expr)));

    Ok(tuple)
}

fn parse_list(tokens: &mut Peekable<impl Iterator<Item = Token>>) -> Result<Expr, String> {
    let exprs = parse_comma_exprs(tokens)?;
    let mut list = Expr::Ident("Nil".to_string());
    for expr in exprs.into_iter().rev() {
        list = Expr::Ap(
            Rc::new(Expr::Ap(
                Rc::new(Expr::Ident("Cons".to_string())),
                Rc::new(expr),
            )),
            Rc::new(list),
        );
    }
    Ok(list)
}

fn parse_comma_exprs(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<Vec<Expr>, String> {
    let mut exprs = Vec::new();
    loop {
        match tokens.peek() {
            Some(Token::RParen) | Some(Token::RBracket) => {
                tokens.next();
                break;
            }
            Some(Token::Comma) => {
                tokens.next();
            }
            _ => {
                let expr = parse_expr(tokens)?;
                exprs.push(expr);
            }
        }
    }
    Ok(exprs)
}

fn parse_lambda_expr(tokens: &mut Peekable<impl Iterator<Item = Token>>) -> Result<Expr, String> {
    let param = match tokens.next() {
        Some(Token::Variable(name)) => name,
        _ => return Err("Expected identifier after 'fun'".to_string()),
    };
    match tokens.next() {
        Some(Token::Arrow) => (),
        _ => return Err("Expected '->' after parameter".to_string()),
    }
    let body = Rc::new(parse_expr(tokens)?);
    Ok(Expr::Lambda(param, body))
}

fn parse_if_expr(tokens: &mut Peekable<impl Iterator<Item = Token>>) -> Result<Expr, String> {
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
    Ok(Expr::If(cond, then, else_))
}

fn parse_let_expr(tokens: &mut Peekable<impl Iterator<Item = Token>>) -> Result<Expr, String> {
    let is_rec = match tokens.peek() {
        Some(Token::Keyword(Keyword::Rec)) => {
            tokens.next();
            true
        }
        _ => false,
    };

    let name = match tokens.next() {
        Some(Token::Variable(name)) => name,
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

    if is_rec {
        Ok(Expr::Let(
            name.clone(),
            Rc::new(Expr::Ap(
                Rc::new(Expr::Ident("fix".to_string())),
                Rc::new(Expr::Lambda(name, Rc::new(bound_expr))),
            )),
            Rc::new(expr),
        ))
    } else {
        Ok(Expr::Let(name, Rc::new(bound_expr), Rc::new(expr)))
    }
}

fn parse_match_expr(tokens: &mut Peekable<impl Iterator<Item = Token>>) -> Result<Expr, String> {
    let expr = parse_expr(tokens)?;

    match tokens.next() {
        Some(Token::Keyword(Keyword::With)) => (),
        _ => return Err("Expected 'with' after match expression".to_string()),
    }

    let cases = parse_match_cases(tokens)?;

    Ok(Expr::Match(Rc::new(expr), cases))
}

fn parse_match_cases(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<Vec<MatchCase>, String> {
    let mut cases = Vec::new();
    while let Some(Token::Pipe) = tokens.peek() {
        tokens.next();

        let pat = parse_pattern(tokens)?;

        let guard = match tokens.peek() {
            Some(Token::Keyword(Keyword::If)) => {
                tokens.next();
                Some(Rc::new(parse_expr(tokens)?))
            }
            _ => None,
        };

        match tokens.next() {
            Some(Token::Arrow) => (),
            _ => return Err("Expected '->' after pattern".to_string()),
        }
        let case_expr = parse_expr(tokens)?;
        cases.push(MatchCase {
            pattern: Rc::new(pat),
            guard,
            body: Rc::new(case_expr),
        });
    }
    Ok(cases)
}

fn parse_pattern(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<ExprPattern, String> {
    let pat = match tokens.peek() {
        Some(Token::Constructor(_)) => parse_constructor_pattern(tokens)?,
        _ => parse_non_constructor_pattern(tokens)?,
    };

    Ok(pat)
}

fn parse_constructor_pattern(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<ExprPattern, String> {
    let name = match tokens.next() {
        Some(Token::Constructor(name)) => name,
        _ => return Err("Expected constructor name".to_string()),
    };

    let mut args = Vec::new();
    loop {
        match tokens.peek() {
            Some(Token::LParen)
            | Some(Token::Variable(_))
            | Some(Token::Constructor(_))
            | Some(Token::Int(_))
            | Some(Token::Underscore)
            | Some(Token::Keyword(Keyword::True))
            | Some(Token::Keyword(Keyword::False)) => {
                let arg = Rc::new(parse_non_constructor_pattern(tokens)?);
                args.push(arg);
            }
            _ => break,
        }
    }

    Ok(ExprPattern::Constructor(name, args))
}

fn parse_non_constructor_pattern(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<ExprPattern, String> {
    let pat = match tokens.next() {
        Some(Token::Underscore) => ExprPattern::Wildcard,
        Some(Token::Variable(name)) => ExprPattern::Variable(name),
        Some(Token::Constructor(name)) => ExprPattern::Constructor(name, vec![]),
        Some(Token::Int(i)) => ExprPattern::Int(i),
        Some(Token::Char(c)) => ExprPattern::Char(c),
        Some(Token::Keyword(Keyword::True)) => ExprPattern::Bool(true),
        Some(Token::Keyword(Keyword::False)) => ExprPattern::Bool(false),
        Some(Token::LParen) => parse_parenthesized_pattern(tokens)?,
        t => return Err(format!("Expected pattern, got {:?}", t)),
    };

    Ok(pat)
}

fn parse_parenthesized_pattern(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<ExprPattern, String> {
    match tokens.peek() {
        Some(Token::RParen) => {
            tokens.next();
            Ok(ExprPattern::Constructor("Unit".to_string(), vec![]))
        }
        _ => {
            let pat = parse_pattern(tokens)?;
            Ok(match tokens.next() {
                Some(Token::RParen) => pat,
                Some(Token::Comma) => parse_tuple_pattern(tokens, pat)?,
                _ => return Err("Expected closing parenthesis".to_string()),
            })
        }
    }
}

fn parse_tuple_pattern(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
    first: ExprPattern,
) -> Result<ExprPattern, String> {
    let mut pats = vec![first];
    pats.append(&mut parse_comma_patterns(tokens)?);

    let tuple = ExprPattern::Constructor(
        format!("Tuple{}", pats.len()),
        pats.into_iter().map(Rc::new).collect(),
    );

    Ok(tuple)
}

fn parse_comma_patterns(
    tokens: &mut Peekable<impl Iterator<Item = Token>>,
) -> Result<Vec<ExprPattern>, String> {
    let mut pats = Vec::new();
    loop {
        match tokens.peek() {
            Some(Token::RParen) => {
                tokens.next();
                break;
            }
            Some(Token::Comma) => {
                tokens.next();
            }
            _ => {
                let pat = parse_pattern(tokens)?;
                pats.push(pat);
            }
        }
    }
    Ok(pats)
}

#[cfg(test)]
mod tests {
    use crate::{
        e_ap, e_bool, e_ident, e_if, e_int, e_lambda, e_let, e_match, e_match_case, p_bool,
        p_constructor, p_int, p_var, p_wildcard, t_bool, t_constructor, t_forall, t_fun, t_int,
        t_type_var,
    };
    use crate::{e_char, lexer};

    use super::*;

    #[test]
    fn test_simple_type() {
        let tokens = lexer::tokenize("int -> bool").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(parse_type_expr(&mut iter), Ok(t_fun!(t_int!(), t_bool!())));
    }

    #[test]
    fn test_type_constructor_simple() {
        let tokens = lexer::tokenize("MyType -> bool").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_type_expr(&mut iter),
            Ok(t_fun!(t_constructor!("MyType"), t_bool!()))
        );
    }

    #[test]
    fn test_type_constructor_args() {
        let tokens = lexer::tokenize("int -> MyType a (MyOtherType -> b) -> bool").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_type_expr(&mut iter),
            Ok(t_fun!(
                t_int!(),
                t_fun!(
                    t_constructor!(
                        "MyType",
                        t_type_var!("a"),
                        t_fun!(t_constructor!("MyOtherType"), t_type_var!("b"))
                    ),
                    t_bool!()
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
            Ok(t_fun!(t_int!(), t_fun!(t_bool!(), t_int!())))
        );
    }

    #[test]
    fn test_type_var() {
        let tokens = lexer::tokenize("(a -> b) -> a").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_arrow_type_expr(&mut iter),
            Ok(t_fun!(
                t_fun!(t_type_var!("a"), t_type_var!("b")),
                t_type_var!("a")
            ))
        );
    }

    #[test]
    fn test_simple_expr() {
        let tokens = lexer::tokenize("fun x -> (fun y -> plus x y)").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_expr(&mut iter),
            Ok(e_lambda!(
                "x",
                e_lambda!(
                    "y",
                    e_ap!(e_ap!(e_ident!("plus"), e_ident!("x")), e_ident!("y"))
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
            Ok(e_if!(
                e_bool!(true),
                e_ap!(e_ap!(e_ident!("mul"), e_ident!("x")), e_ident!("y")),
                e_ap!(e_lambda!("x", e_ident!("x")), e_ident!("x"))
            ))
        );
    }

    #[test]
    fn test_let_statement() {
        let tokens = lexer::tokenize("let f = fun x -> x").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_statement(&mut iter),
            Ok(Statement::Let(
                "f".to_string(),
                Rc::new(e_lambda!("x", e_ident!("x")))
            ))
        );
    }

    #[test]
    fn test_let_expression() {
        let tokens = lexer::tokenize("let x = f y in x 5").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_expr(&mut iter),
            Ok(e_let!(
                "x",
                e_ap!(e_ident!("f"), e_ident!("y")),
                e_ap!(e_ident!("x"), e_int!(5))
            ))
        );
    }

    #[test]
    fn test_let_rec_expression() {
        let tokens = lexer::tokenize("let rec f = fun n -> f n in f").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_expr(&mut iter),
            Ok(e_let!(
                "f",
                e_ap!(
                    e_ident!("fix"),
                    e_lambda!("f", e_lambda!("n", e_ap!(e_ident!("f"), e_ident!("n"))))
                ),
                e_ident!("f")
            ))
        );
    }

    #[test]
    fn test_type_scheme() {
        let tokens = lexer::tokenize("forall a b . a -> b").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_type_expr(&mut iter),
            Ok(t_forall!(
                ["a", "b"],
                t_fun!(t_type_var!("a"), t_type_var!("b"))
            ))
        );
    }

    #[test]
    fn test_data_simple() {
        let tokens = lexer::tokenize("data MyType = A | BB | Ccc").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_statement(&mut iter),
            Ok(Statement::Data(
                "MyType".to_string(),
                vec![],
                vec![
                    t_constructor!("A"),
                    t_constructor!("BB"),
                    t_constructor!("Ccc"),
                ]
            ))
        );
    }

    #[test]
    fn test_data_with_args() {
        let tokens = lexer::tokenize("data MyType a b = A | BB a bool | Ccc b").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_statement(&mut iter),
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

    #[test]
    fn test_match() {
        let tokens = lexer::tokenize("match x with | A -> 1 | B -> 2").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_expr(&mut iter),
            Ok(e_match!(
                e_ident!("x"),
                [
                    e_match_case!(p_constructor!("A"), None, e_int!(1)),
                    e_match_case!(p_constructor!("B"), None, e_int!(2))
                ]
            ))
        );
    }

    #[test]
    fn test_match_with_args() {
        let tokens =
            lexer::tokenize("match f x with | A 2 -> 1 | B (C _) true x if x > 0 -> 2").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_expr(&mut iter),
            Ok(e_match!(
                e_ap!(e_ident!("f"), e_ident!("x")),
                [
                    e_match_case!(p_constructor!("A", p_int!(2)), None, e_int!(1)),
                    e_match_case!(
                        p_constructor!(
                            "B",
                            p_constructor!("C", p_wildcard!()),
                            p_bool!(true),
                            p_var!("x")
                        ),
                        Some(e_ap!(e_ap!(e_ident!(">"), e_ident!("x")), e_int!(0))),
                        e_int!(2)
                    )
                ]
            ))
        );
    }

    #[test]
    fn test_parse_list() {
        let tokens = lexer::tokenize("[1, 2, 3]").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_expr(&mut iter),
            Ok(e_ap!(
                e_ap!(e_ident!("Cons"), e_int!(1)),
                e_ap!(
                    e_ap!(e_ident!("Cons"), e_int!(2)),
                    e_ap!(e_ap!(e_ident!("Cons"), e_int!(3)), e_ident!("Nil"))
                )
            ))
        );
    }

    #[test]
    fn test_parse_tuple() {
        let tokens = lexer::tokenize("(1, 2, 3)").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_expr(&mut iter),
            Ok(e_ap!(
                e_ap!(e_ap!(e_ident!("Tuple3"), e_int!(1)), e_int!(2)),
                e_int!(3)
            ))
        );
    }

    #[test]
    fn test_pattern_tuple() {
        let tokens = lexer::tokenize("(1, 2, 3)").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_pattern(&mut iter),
            Ok(p_constructor!("Tuple3", p_int!(1), p_int!(2), p_int!(3)))
        );
    }

    #[test]
    fn test_parse_string() {
        let tokens = lexer::tokenize("\"abc\"").unwrap();
        let mut iter = tokens.into_iter().peekable();

        assert_eq!(
            parse_expr(&mut iter),
            Ok(e_ap!(
                e_ap!(e_ident!("Cons"), e_char!('a')),
                e_ap!(
                    e_ap!(e_ident!("Cons"), e_char!('b')),
                    e_ap!(e_ap!(e_ident!("Cons"), e_char!('c')), e_ident!("Nil"))
                )
            ))
        );
    }
}
