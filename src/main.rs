use std::io::{stdin, stdout, Write};

use crate::{env::Env, parser::parse};

mod ast;
mod builtins;
mod env;
mod eval;
mod lexer;
mod parser;
mod prelude;
mod value;

fn main() {
    let mut env = Env::prelude();

    print!("> ");
    stdout().flush().unwrap();

    for res in stdin().lines() {
        let line = res.unwrap();

        match eval_line(&env, &line) {
            Ok((response, new_env)) => {
                println!("{}", response);
                env = new_env;
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }

        print!("> ");
        stdout().flush().unwrap();
    }

    println!("done")
}

fn eval_line(env: &Env, line: &str) -> Result<(String, Env), String> {
    let parse_result = parse(line)?;

    Ok(match parse_result {
        parser::ParseResult::Statement(stmt) => {
            let new_env = env.eval_statement(stmt)?;
            ("ok".to_string(), new_env)
        }
        parser::ParseResult::Expression(expr) => {
            let res = env.eval_expr(expr)?;
            (format!("{}", res), env.clone())
        }
    })
}

// Tests
