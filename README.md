# 🤔

🤔 (pronounced "hmm") is a simplistic functional programming language inspired by OCaml and Haskell.

**Disclaimer: this is a toy project and not intended for production use.**

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

See [prelude](src/prelude.rs) for more examples.

### Recursion and pattern matching

```ocaml
let rec fib n =
  match n with
  | 0 -> 0
  | 1 -> 1
  | n -> fib (n - 1) + fib (n - 2)
```

### FizzBuzz

`putStrLn` is a built-in impure function that prints a string to stdout.

```ocaml
let fizzbuzz n =
    let fz = "Fizz"
    in let bz = "Buzz"
    in let fzbz = "FizzBuzz"
    in let fb n =
        match (n % 3, n % 5) with
        | (0, 0) -> putStrLn fzbz
        | (0, _) -> putStrLn fz
        | (_, 0) -> putStrLn bz
        | _ -> putStrLn (intToString n)
    in
        discard (map (fun n -> fb (n + 1)) (range n))
```

### Algebraic data types

```ocaml
data Maybe a = Just a | Nothing

let map f xs =
    match xs with
    | Just x -> Just (f x)
    | Nothing -> Nothing
```

### Lists

```ocaml
let rec sum xs =
    match xs with
    | Nil -> 0
    | Cons x xs -> x + sum xs

let s = sum [1, 2, 3, 4, 5]
```

## TODO

- [ ] Standalone scripts
- [ ] Proper prelude
- [ ] Explicit type annotations
- [x] `let rec .. in ..` syntax sugar
- [x] `match` guards
- [ ] Mutually recursive functions (currently supported with explicit `fix`)
- [x] Multi-argument lambdas
- [x] `let` bindings with arguments
- [ ] Better error messages
- [ ] Prettier pretty-printer
- [ ] Exhaustiveness checking for pattern matching
- [ ] String interning for identifiers
- [ ] Compilation (native and/or to JS)
- [ ] Typeclasses
- [ ] ...
