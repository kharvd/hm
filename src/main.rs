mod ast;
mod ast_macros;
mod builtins;
mod env;
mod eval;
mod lexer;
mod memory;
mod parser;
mod pattern;
mod prelude;
mod repl;
mod typing;
mod value;

use rustyline::Result;

fn main() -> Result<()> {
    repl::repl()
}
