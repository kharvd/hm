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
mod pretty;
mod repl;
mod typing;
mod value;

use rustyline::Result;

use clap::Parser;

/// REPL for the 🤔 language
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Show timing and memory usage information
    #[arg(short, long)]
    profile: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    repl::repl(args.profile)
}
