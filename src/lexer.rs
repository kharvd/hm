use itertools::Itertools;

#[derive(Debug, PartialEq)]
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

#[derive(Debug, PartialEq)]
pub enum Token {
    Variable(String),
    Constructor(String),
    Int(i64),
    Keyword(Keyword),
    LParen,
    RParen,
    Arrow,
    Colon,
    Apostrophe,
    Equals,
    Dot,
    Pipe,
    Underscore,
}

pub fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&c) = chars.peek() {
        match c {
            'a'..='z' | 'A'..='Z' => {
                let ident: String = chars
                    .by_ref()
                    .peeking_take_while(|&ch| ch.is_ascii_alphanumeric())
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
            ':' => {
                tokens.push(Token::Colon);
                chars.next();
            }
            '\'' => {
                tokens.push(Token::Apostrophe);
                chars.next();
            }
            '-' => {
                chars.next(); // consume '-'
                if chars.next() == Some('>') {
                    // consume the next character, which should be '>'
                    tokens.push(Token::Arrow);
                } else {
                    return Err("Unexpected sequence after '-'".to_string());
                }
            }
            '=' => {
                tokens.push(Token::Equals);
                chars.next();
            }
            '.' => {
                tokens.push(Token::Dot);
                chars.next();
            }
            '|' => {
                tokens.push(Token::Pipe);
                chars.next();
            }
            '_' => {
                tokens.push(Token::Underscore);
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
    }
}
