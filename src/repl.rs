use crate::eval::StatementEval;
use crate::memory;
use crate::parser;
use crate::typing::infer;
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
                match evaluator.eval_line(&line) {
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
}
