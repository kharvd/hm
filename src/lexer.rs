use itertools::Itertools;

#[derive(Debug, PartialEq, Clone)]
pub enum Keyword {
    Fun,
    Let,
    Rec,
    Match,
    With,
    Data,
    In,
    Val,
    Forall,
    If,
    Then,
    Else,
    True,
    False,
    Bool,
    Int,
}

#[derive(Debug, PartialEq, Clone)]
pub enum InfixOp {
    Plus,
    Minus,
    Mult,
    Div,
    Equals,
    NotEquals,
    LessThan,
    GreaterThan,
    LessEquals,
    GreaterEquals,
    And,
    Or,
}

impl InfixOp {
    pub fn to_string(&self) -> &'static str {
        match self {
            InfixOp::Plus => "+",
            InfixOp::Minus => "-",
            InfixOp::Mult => "*",
            InfixOp::Div => "/",
            InfixOp::Equals => "==",
            InfixOp::NotEquals => "!=",
            InfixOp::LessThan => "<",
            InfixOp::GreaterThan => ">",
            InfixOp::LessEquals => "<=",
            InfixOp::GreaterEquals => ">=",
            InfixOp::And => "&&",
            InfixOp::Or => "||",
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Variable(String),
    Constructor(String),
    Int(i64),
    Keyword(Keyword),
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Arrow,
    Colon,
    Apostrophe,
    Equals,
    Dot,
    Pipe,
    Underscore,
    InfixOp(InfixOp),
}

pub fn is_valid_identifier_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

pub fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&c) = chars.peek() {
        match c {
            'a'..='z' | 'A'..='Z' => {
                let ident: String = chars
                    .by_ref()
                    .peeking_take_while(|&ch| is_valid_identifier_char(ch))
                    .collect();

                match ident.as_str() {
                    "fun" => tokens.push(Token::Keyword(Keyword::Fun)),
                    "let" => tokens.push(Token::Keyword(Keyword::Let)),
                    "rec" => tokens.push(Token::Keyword(Keyword::Rec)),
                    "match" => tokens.push(Token::Keyword(Keyword::Match)),
                    "with" => tokens.push(Token::Keyword(Keyword::With)),
                    "data" => tokens.push(Token::Keyword(Keyword::Data)),
                    "in" => tokens.push(Token::Keyword(Keyword::In)),
                    "val" => tokens.push(Token::Keyword(Keyword::Val)),
                    "forall" => tokens.push(Token::Keyword(Keyword::Forall)),
                    "if" => tokens.push(Token::Keyword(Keyword::If)),
                    "then" => tokens.push(Token::Keyword(Keyword::Then)),
                    "else" => tokens.push(Token::Keyword(Keyword::Else)),
                    "true" => tokens.push(Token::Keyword(Keyword::True)),
                    "false" => tokens.push(Token::Keyword(Keyword::False)),
                    "bool" => tokens.push(Token::Keyword(Keyword::Bool)),
                    "int" => tokens.push(Token::Keyword(Keyword::Int)),
                    s => {
                        if s.starts_with(char::is_uppercase) {
                            tokens.push(Token::Constructor(ident))
                        } else {
                            tokens.push(Token::Variable(ident))
                        }
                    }
                }
            }
            '0'..='9' => {
                let num: String = chars
                    .by_ref()
                    .peeking_take_while(|ch| ch.is_digit(10))
                    .collect();
                if let Ok(i) = num.parse::<i64>() {
                    tokens.push(Token::Int(i));
                } else {
                    return Err(format!("Failed to parse {} as i64", num));
                }
            }
            '(' => {
                tokens.push(Token::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(Token::RParen);
                chars.next();
            }
            '[' => {
                tokens.push(Token::LBracket);
                chars.next();
            }
            ']' => {
                tokens.push(Token::RBracket);
                chars.next();
            }
            ',' => {
                tokens.push(Token::Comma);
                chars.next();
            }
            ':' => {
                tokens.push(Token::Colon);
                chars.next();
            }
            '\'' => {
                tokens.push(Token::Apostrophe);
                chars.next();
            }
            '-' => {
                chars.next();
                if let Some('>') = chars.peek() {
                    tokens.push(Token::Arrow);
                    chars.next();
                } else {
                    tokens.push(Token::InfixOp(InfixOp::Minus));
                }
            }
            '=' => {
                chars.next();
                if let Some('=') = chars.peek() {
                    tokens.push(Token::InfixOp(InfixOp::Equals));
                    chars.next();
                } else {
                    tokens.push(Token::Equals);
                }
            }
            '<' => {
                chars.next();
                if let Some('=') = chars.peek() {
                    tokens.push(Token::InfixOp(InfixOp::LessEquals));
                    chars.next();
                } else {
                    tokens.push(Token::InfixOp(InfixOp::LessThan));
                }
            }
            '>' => {
                chars.next();
                if let Some('=') = chars.peek() {
                    tokens.push(Token::InfixOp(InfixOp::GreaterEquals));
                    chars.next();
                } else {
                    tokens.push(Token::InfixOp(InfixOp::GreaterThan));
                }
            }
            '!' => {
                chars.next();
                if let Some('=') = chars.peek() {
                    tokens.push(Token::InfixOp(InfixOp::NotEquals));
                    chars.next();
                } else {
                    return Err("Unexpected sequence after '!'".to_string());
                }
            }
            '&' => {
                chars.next();
                if let Some('&') = chars.peek() {
                    tokens.push(Token::InfixOp(InfixOp::And));
                    chars.next();
                } else {
                    return Err("Unexpected sequence after '&'".to_string());
                }
            }
            '|' => {
                chars.next();
                if let Some('|') = chars.peek() {
                    tokens.push(Token::InfixOp(InfixOp::Or));
                    chars.next();
                } else {
                    tokens.push(Token::Pipe);
                }
            }
            '.' => {
                tokens.push(Token::Dot);
                chars.next();
            }
            '_' => {
                tokens.push(Token::Underscore);
                chars.next();
            }
            '+' => {
                tokens.push(Token::InfixOp(InfixOp::Plus));
                chars.next();
            }
            '*' => {
                tokens.push(Token::InfixOp(InfixOp::Mult));
                chars.next();
            }
            '/' => {
                tokens.push(Token::InfixOp(InfixOp::Div));
                chars.next();
            }
            ' ' | '\t' | '\n' | '\r' => {
                chars.next(); // Just skip whitespace
            }
            _ => {
                return Err(format!("Unexpected character '{}'", c));
            }
        }
    }

    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_simple() {
        assert_eq!(
            tokenize("fun x -> (fun y -> plus x y)"),
            Ok(vec![
                Token::Keyword(Keyword::Fun),
                Token::Variable("x".to_string()),
                Token::Arrow,
                Token::LParen,
                Token::Keyword(Keyword::Fun),
                Token::Variable("y".to_string()),
                Token::Arrow,
                Token::Variable("plus".to_string()),
                Token::Variable("x".to_string()),
                Token::Variable("y".to_string()),
                Token::RParen,
            ])
        );
        assert_eq!(
            tokenize("'x : int"),
            Ok(vec![
                Token::Apostrophe,
                Token::Variable("x".to_string()),
                Token::Colon,
                Token::Keyword(Keyword::Int),
            ])
        );
        assert_eq!(
            tokenize("if true then (mul x y) else ((fun x -> x) x)"),
            Ok(vec![
                Token::Keyword(Keyword::If),
                Token::Keyword(Keyword::True),
                Token::Keyword(Keyword::Then),
                Token::LParen,
                Token::Variable("mul".to_string()),
                Token::Variable("x".to_string()),
                Token::Variable("y".to_string()),
                Token::RParen,
                Token::Keyword(Keyword::Else),
                Token::LParen,
                Token::LParen,
                Token::Keyword(Keyword::Fun),
                Token::Variable("x".to_string()),
                Token::Arrow,
                Token::Variable("x".to_string()),
                Token::RParen,
                Token::Variable("x".to_string()),
                Token::RParen,
            ])
        );

        assert_eq!(
            tokenize("let f = fun x -> x"),
            Ok(vec![
                Token::Keyword(Keyword::Let),
                Token::Variable("f".to_string()),
                Token::Equals,
                Token::Keyword(Keyword::Fun),
                Token::Variable("x".to_string()),
                Token::Arrow,
                Token::Variable("x".to_string()),
            ])
        );

        assert_eq!(
            tokenize("(neg 1)"),
            Ok(vec![
                Token::LParen,
                Token::Variable("neg".to_string()),
                Token::Int(1),
                Token::RParen,
            ])
        );

        assert_eq!(
            tokenize("let f = fun x -> x in f 5"),
            Ok(vec![
                Token::Keyword(Keyword::Let),
                Token::Variable("f".to_string()),
                Token::Equals,
                Token::Keyword(Keyword::Fun),
                Token::Variable("x".to_string()),
                Token::Arrow,
                Token::Variable("x".to_string()),
                Token::Keyword(Keyword::In),
                Token::Variable("f".to_string()),
                Token::Int(5),
            ])
        );

        assert_eq!(
            tokenize("(x < 5 * 2 + 1) || (y - 5 != 1)"),
            Ok(vec![
                Token::LParen,
                Token::Variable("x".to_string()),
                Token::InfixOp(InfixOp::LessThan),
                Token::Int(5),
                Token::InfixOp(InfixOp::Mult),
                Token::Int(2),
                Token::InfixOp(InfixOp::Plus),
                Token::Int(1),
                Token::RParen,
                Token::InfixOp(InfixOp::Or),
                Token::LParen,
                Token::Variable("y".to_string()),
                Token::InfixOp(InfixOp::Minus),
                Token::Int(5),
                Token::InfixOp(InfixOp::NotEquals),
                Token::Int(1),
                Token::RParen,
            ])
        );
    }
}
