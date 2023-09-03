use std::io::{stdin, stdout, Write};

use eval::StatementEval;
use typing::infer;

use crate::{env::Env, parser::parse};

mod ast;
mod ast_macros;
mod builtins;
mod env;
mod eval;
mod lexer;
mod mini_parser;
mod parser;
mod pattern;
mod prelude;
mod typing;
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
            let StatementEval { new_env, statement } = env.eval_statement(&stmt)?;
            (format!("{}", statement), new_env)
        }
        parser::ParseResult::Expression(expr) => {
            let type_expr = infer(env, &expr)?;
            let res = env.eval_expr(&expr)?;
            (format!("{} : {}", res, type_expr), env.clone())
        }
    })
}

// Tests
