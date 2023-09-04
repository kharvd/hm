# 🤔

🤔 (pronounced "hmm") is an ML-like functional programming language.

## Usage

Run REPL:
```bash
$ cargo run
```

## Features

- Hindley-Milner type inference
- Fixed-point combinator `fix`
- Pattern matching
- Algebraic data types
- Lists and tuples
- First-class functions
- Tail call optimization
- Whitespace insensitive
- REPL

## Examples

### Recursion and pattern matching

```ocaml
let rec fib = fun n ->
  match n with
  | 0 -> 0
  | 1 -> 1
  | n -> fib (n - 1) + fib (n - 2)
```

### FizzBuzz

`putStrLn` is a built-in impure function that prints a string to stdout.

```ocaml
let fizzbuzz = fun n ->
    let fb = fun n ->
        match (n % 3, n % 5) with
        | (0, 0) -> putStrLn "FizzBuzz"
        | (0, _) -> putStrLn "Fizz"
        | (_, 0) -> putStrLn "Buzz"
        | _ -> putStrLn (intToString n)
    in
        discard (map (fun n -> fb (n + 1)) (range n))
```

### Algebraic data types

```ocaml
data Maybe a = Just a | Nothing

let map = fun f -> fun xs ->
    match xs with
    | Just x -> Just (f x)
    | Nothing -> Nothing
```

### Lists

```ocaml
let rec sum = fun xs ->
    match xs with
    | Nil -> 0
    | Cons x xs -> x + sum xs

let s = sum [1, 2, 3, 4, 5]
```

## TODO

- [ ] Standalone scripts
- [ ] Explicit type annotations
- [ ] `let rec .. in ..` syntax sugar
- [ ] `match` guards
- [ ] Mutually recursive functions (currently supported with explicit `fix`)
- [ ] Multi-argument lambdas
- [ ] `let` bindings with arguments
- [ ] Better error messages
- [ ] Exhaustiveness checking for pattern matching
- [ ] Compilation (native and/or to JS)
- [ ] Typeclasses
- ...
