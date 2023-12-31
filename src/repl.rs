use std::borrow::Borrow;

use crate::ast::Statement;
use crate::eval::StatementEval;
use crate::memory;
use crate::parser;
use crate::typing::infer;
use crate::value::RefValue;
use crate::value::Value;
use rustyline::error::ReadlineError;
use rustyline::{DefaultEditor, Result};

use crate::{env::Env, parser::parse};

const HISTORY_FILE: &str = "hmhistory";

pub fn repl(profile: bool) -> Result<()> {
    let mut evaluator = Evaluator::new(profile);

    let mut rl = DefaultEditor::new()?;
    let _ = rl.load_history(HISTORY_FILE);

    loop {
        let readline = rl.readline("🤔> ");
        match readline {
            Ok(line) => {
                rl.add_history_entry(line.as_str())?;

                let line = line.trim();
                let colon_command = ColonCommand::parse(line);

                let eval_result = match colon_command {
                    Some(command) => evaluator.eval_colon_command(command),
                    None => evaluator.eval_line(line),
                };

                match eval_result {
                    Ok(response) => {
                        println!("{}", response);
                    }
                    Err(e) => {
                        println!("Error: {}", e);
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("bye~");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }

    rl.save_history(HISTORY_FILE)?;

    Ok(())
}

enum ColonCommand<'a> {
    Type(&'a str),
    Debug(&'a str),
    DebugEval(&'a str),
    Source(&'a str),
    Help,
}

impl<'a> ColonCommand<'a> {
    fn parse(line: &'a str) -> Option<Self> {
        if line.starts_with(":t") {
            Some(ColonCommand::Type(&line[3..]))
        } else if line.starts_with(":h") {
            Some(ColonCommand::Help)
        } else if line.starts_with(":e") {
            Some(ColonCommand::DebugEval(&line[3..]))
        } else if line.starts_with(":d") {
            Some(ColonCommand::Debug(&line[3..]))
        } else if line.starts_with(":s") {
            Some(ColonCommand::Source(&line[3..]))
        } else {
            None
        }
    }
}

struct Evaluator {
    env: Env,
    profile: bool,
}

impl Evaluator {
    fn new(profile: bool) -> Self {
        Self {
            env: Env::prelude(),
            profile,
        }
    }

    fn eval_line(&mut self, line: &str) -> std::result::Result<String, String> {
        let parse_result = parse(line)?;

        Ok(match parse_result {
            parser::ParseResult::Statement(stmt) => {
                let StatementEval { new_env, statement } = self.env.eval_statement(&stmt)?;
                self.env = new_env;
                format!("{}", statement)
            }
            parser::ParseResult::Expression(expr) => {
                let start = std::time::Instant::now();
                let type_expr = infer(&self.env, &expr)?;
                let type_checking_time = start.elapsed();

                let start = std::time::Instant::now();
                let res = self.env.eval_expr(expr)?;
                let evaluation_time = start.elapsed();

                let eval_result = format!("{} : {}", res, type_expr);

                if self.profile {
                    format!(
                        "{}\ntc: {}ms\neval: {}ms\nmemory: {} bytes",
                        eval_result,
                        type_checking_time.as_millis(),
                        evaluation_time.as_millis(),
                        memory::allocated()
                    )
                } else {
                    eval_result
                }
            }
        })
    }

    fn eval_colon_command(&mut self, command: ColonCommand) -> std::result::Result<String, String> {
        match command {
            ColonCommand::Type(line) => {
                let expr = parse(line)?;
                match expr {
                    parser::ParseResult::Statement(_) => Err("Expected expression".to_string()),
                    parser::ParseResult::Expression(expr) => {
                        let type_expr = infer(&self.env, &expr)?;
                        Ok(format!("{} : {}", expr, type_expr))
                    }
                }
            }
            ColonCommand::Debug(line) => {
                let expr = parse(line)?;
                Ok(format!("{:#?}", expr))
            }
            ColonCommand::DebugEval(line) => {
                let result = parse(line)?;
                match result {
                    parser::ParseResult::Statement(_) => Err("Expected expression".to_string()),
                    parser::ParseResult::Expression(expr) => {
                        let res = self.env.eval_expr(expr)?;
                        Ok(format!("{:#?}", res))
                    }
                }
            }
            ColonCommand::Source(line) => {
                let result = self.env.resolve_value(line)?;
                match &result {
                    Some(value) => match value {
                        Value::RefValue(ref_value) => match ref_value.borrow() {
                            RefValue::Func { body, .. } => {
                                let let_stmt = Statement::Let(line.to_string(), body.clone());
                                Ok(format!("{}", let_stmt))
                            }
                            _ => Ok(format!("{}", value)),
                        },
                        _ => Ok(format!("{}", value)),
                    },
                    None => Err(format!("Unknown identifier {}", line)),
                }
            }
            ColonCommand::Help => Ok("Available commands:\n\
                    :t <expr> - typecheck expression\n\
                    :d <expr> - print expression / statement AST\n\
                    :e <expr> - evaluate expression without typechecking and print result\n\
                    :s <name> - print source code of function or data type\n\
                    :h - show this help message"
                .to_string()),
        }
    }
}
