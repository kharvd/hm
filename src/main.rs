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
mod parser;
mod pattern;
mod prelude;
mod typing;
mod value;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Counter;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = System.alloc(layout);
        if !ret.is_null() {
            ALLOCATED.fetch_add(layout.size(), Relaxed);
        }
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED.fetch_sub(layout.size(), Relaxed);
    }
}

#[global_allocator]
static A: Counter = Counter;

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
            let start = std::time::Instant::now();
            let type_expr = infer(env, &expr)?;
            let type_checking_time = start.elapsed();

            let start = std::time::Instant::now();
            let res = env.eval_expr(expr)?;
            let evaluation_time = start.elapsed();

            (
                format!(
                    "{} : {}\ntc: {}ms\neval: {}ms\nmemory: {} bytes",
                    res,
                    type_expr,
                    type_checking_time.as_millis(),
                    evaluation_time.as_millis(),
                    ALLOCATED.load(Relaxed)
                ),
                env.clone(),
            )
        }
    })
}

// Tests
