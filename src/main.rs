mod ast;
mod builtins;
mod env;
mod eval;
mod lexer;
mod parser;
mod prelude;
mod value;

fn main() {
    println!("{:?}", lexer::tokenize("fun x -> (fun y -> plus x y)"));
    println!("{:?}", lexer::tokenize("'x : int"));
    println!(
        "{:?}",
        lexer::tokenize("if true then (mul x y) else ((fun x -> x) x)")
    );
    println!("{:?}", lexer::tokenize("f: ('a -> 'a) -> 'b"));
}

// Tests
