use crate::{
    builtins::{BoolBinOps, BoolUnaryOps, CharUnaryOps, IntBinOps, IntUnaryOps, IO},
    env::Env,
    value::Value,
};

impl Env {
    pub fn prelude() -> Env {
        let env = Env::new()
            .extend_value("+", Value::builtin(IntBinOps::Plus))
            .extend_value("-", Value::builtin(IntBinOps::Minus))
            .extend_value("*", Value::builtin(IntBinOps::Mult))
            .extend_value("/", Value::builtin(IntBinOps::Div))
            .extend_value("%", Value::builtin(IntBinOps::Mod))
            .extend_value("<", Value::builtin(IntBinOps::Lt))
            .extend_value("<=", Value::builtin(IntBinOps::Leq))
            .extend_value(">", Value::builtin(IntBinOps::Gt))
            .extend_value(">=", Value::builtin(IntBinOps::Geq))
            .extend_value("neg", Value::builtin(IntUnaryOps::Neg))
            .extend_value("chr", Value::builtin(IntUnaryOps::Chr))
            .extend_value("ord", Value::builtin(CharUnaryOps::Ord))
            .extend_value("not", Value::builtin(BoolUnaryOps::Not))
            .extend_value("&&", Value::builtin(BoolBinOps::And))
            .extend_value("||", Value::builtin(BoolBinOps::Or))
            .extend_value("xor", Value::builtin(BoolBinOps::Xor))
            .extend_value("putc", Value::builtin(IO::Putc))
            .extend_value("fix", Value::Fix);

        let prelude_source = "
            val (+) : int -> int -> int
            val (-) : int -> int -> int
            val (*) : int -> int -> int
            val (/) : int -> int -> int
            val (%) : int -> int -> int
            val (<) : int -> int -> bool
            val (<=) : int -> int -> bool
            val (>) : int -> int -> bool
            val (>=) : int -> int -> bool
            val neg : int -> int
            val chr : int -> char
            val ord : char -> int

            val not : bool -> bool
            val (&&) : bool -> bool -> bool
            val (||) : bool -> bool -> bool
            val xor : bool -> bool -> bool

            let (==) x y = (x <= y) && (x >= y)

            val fix : (a -> a) -> a

            data Unit = Unit
            data Tuple1 a = Tuple1 a
            data Tuple2 a b = Tuple2 a b
            data Tuple3 a b c = Tuple3 a b c
            data Tuple4 a b c d = Tuple4 a b c d
            data Tuple5 a b c d e = Tuple5 a b c d e
            data Tuple6 a b c d e f = Tuple6 a b c d e f
            data Tuple7 a b c d e f g = Tuple7 a b c d e f g
            data Tuple8 a b c d e f g h = Tuple8 a b c d e f g h
            data Tuple9 a b c d e f g h i = Tuple9 a b c d e f g h i
            data Tuple10 a b c d e f g h i j = Tuple10 a b c d e f g h i j

            let fst p =
                match p with 
                | (x, _) -> x

            let snd p =
                match p with
                | (_, y) -> y
            
            data List a = Nil | Cons a (List a)
            let nil = Nil
            let cons = Cons
            let rec len xs =
                match xs with 
                | Nil -> 0 
                | Cons x xs -> 1 + (len xs)

            let rec map f xs =
                match xs with
                | Nil -> Nil
                | Cons x xs -> Cons (f x) (map f xs)

            let rec foldl f acc xs =
                match xs with
                | Nil -> acc
                | Cons x xs -> foldl f (f acc x) xs

            let rec foldr f acc xs =
                match xs with
                | Nil -> acc
                | Cons x xs -> f x (foldr f acc xs)
            
            let rec sum = foldl (+) 0
            let rec product = foldl (*) 1
            let rec all = foldl (&&) true
            let rec any = foldl (||) false

            let rec take n xs = 
                if n == 0 then Nil else
                match xs with
                | Nil -> Nil
                | Cons x xs -> Cons x (take (n - 1) xs)
            
            let rec drop n xs =
                if n == 0 then xs else
                match xs with
                | Nil -> Nil
                | Cons x xs -> drop (n - 1) xs
            
            let rec append xs ys =
                match xs with
                | Nil -> ys
                | Cons x xs -> Cons x (append xs ys)
            
            let reverse = foldl (fun acc x -> Cons x acc) Nil

            let range = 
                let rec range_rec acc n =
                    match n with
                    | 0 -> acc
                    | _ -> range_rec (Cons (n - 1) acc) (n - 1)
                in range_rec Nil

            let head xs =
                match xs with
                | Cons x _ -> x
            
            let tail xs =
                match xs with
                | Nil -> Nil
                | Cons x rest -> rest 
            
            let rec list_eq xs ys =
                match (xs, ys) with
                | (Nil, Nil) -> true
                | (Cons x xs, Cons y ys) if x == y -> list_eq xs ys
                | _ -> false
            
            let rec zip xs ys =
                match (xs, ys) with
                | (Nil, Nil) -> Nil
                | (Cons x xs, Cons y ys) -> Cons (Tuple2 x y) (zip xs ys)
                | _ -> Nil

            val putc : char -> Unit

            let discard x = Unit

            let rec do xs =
                match xs with
                | Nil -> Unit
                | Cons x xs -> discard (x, do xs)

            let rec putStr s = discard (map putc s)
            
            let putStrLn s = 
                do [
                    putStr s,
                    putc '\n'
                ]
            
            let intToString n =
                let rec intToString_rec acc n =
                    if n == 0 then acc else
                    intToString_rec (Cons (chr (ord '0' + (n % 10))) acc) (n / 10)
                in
                    if n == 0 
                    then \"0\" 
                    else if n < 0 
                         then cons '-' (intToString_rec Nil (neg n))
                         else intToString_rec Nil n
            
        ";

        env.eval_file(prelude_source).unwrap()
    }
}
