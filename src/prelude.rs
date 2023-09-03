use crate::{
    builtins::{BoolBinOps, BoolUnaryOps, IntBinOps, IntUnaryOps},
    env::Env,
    value::Value,
};

impl Env {
    pub fn prelude() -> Env {
        let env = Env::new()
            .extend("+", Value::new_builtin(IntBinOps::Plus))
            .extend("-", Value::new_builtin(IntBinOps::Minus))
            .extend("*", Value::new_builtin(IntBinOps::Mult))
            .extend("/", Value::new_builtin(IntBinOps::Div))
            .extend("<", Value::new_builtin(IntBinOps::Lt))
            .extend("<=", Value::new_builtin(IntBinOps::Leq))
            .extend(">", Value::new_builtin(IntBinOps::Gt))
            .extend(">=", Value::new_builtin(IntBinOps::Geq))
            .extend("neg", Value::new_builtin(IntUnaryOps::Neg))
            .extend("not", Value::new_builtin(BoolUnaryOps::Not))
            .extend("&&", Value::new_builtin(BoolBinOps::And))
            .extend("||", Value::new_builtin(BoolBinOps::Or))
            .extend("xor", Value::new_builtin(BoolBinOps::Xor))
            .extend("fix", Value::Fix);

        let prelude_source = "
            val (+) : int -> int -> int
            val (-) : int -> int -> int
            val (*) : int -> int -> int
            val (/) : int -> int -> int
            val (<) : int -> int -> bool
            val (<=) : int -> int -> bool
            val (>) : int -> int -> bool
            val (>=) : int -> int -> bool
            val neg : int -> int

            val not : bool -> bool
            val (&&) : bool -> bool -> bool
            val (||) : bool -> bool -> bool
            val xor : bool -> bool -> bool

            val fix : ('a -> 'a) -> 'a
            let (==) = fun x -> fun y -> (x <= y) && (x >= y)
                
            data Unit = Unit
            data Tuple1 'a = Tuple1 'a
            data Tuple2 'a 'b = Tuple2 'a 'b
            data Tuple3 'a 'b 'c = Tuple3 'a 'b 'c
            data Tuple4 'a 'b 'c 'd = Tuple4 'a 'b 'c 'd
            data Tuple5 'a 'b 'c 'd 'e = Tuple5 'a 'b 'c 'd 'e
            data Tuple6 'a 'b 'c 'd 'e 'f = Tuple6 'a 'b 'c 'd 'e 'f
            data Tuple7 'a 'b 'c 'd 'e 'f 'g = Tuple7 'a 'b 'c 'd 'e 'f 'g
            data Tuple8 'a 'b 'c 'd 'e 'f 'g 'h = Tuple8 'a 'b 'c 'd 'e 'f 'g 'h
            data Tuple9 'a 'b 'c 'd 'e 'f 'g 'h 'i = Tuple9 'a 'b 'c 'd 'e 'f 'g 'h 'i
            data Tuple10 'a 'b 'c 'd 'e 'f 'g 'h 'i 'j = Tuple10 'a 'b 'c 'd 'e 'f 'g 'h 'i 'j

            let fst = fun p -> 
                match p with 
                | (x, _) -> x

            let snd = fun p -> 
                match p with
                | (_, y) -> y

            data List 'a = Nil | Cons 'a (List 'a)
            let nil = Nil
            let cons = fun x -> fun xs -> Cons x xs
            let rec len = fun xs -> 
                match xs with 
                | Nil -> 0 
                | Cons x xs -> 1 + (len xs)

            let rec map = fun f -> fun xs ->
                match xs with
                | Nil -> Nil
                | Cons x xs -> Cons (f x) (map f xs)
            
            let rec foldl = fun f -> fun acc -> fun xs ->
                match xs with
                | Nil -> acc
                | Cons x xs -> foldl f (f acc x) xs

            let rec foldr = fun f -> fun acc -> fun xs ->
                match xs with
                | Nil -> acc
                | Cons x xs -> f x (foldr f acc xs)
            
            let rec sum = fun xs -> foldl (+) 0 xs
            let rec product = fun xs -> foldl (*) 1 xs
            let rec all = fun xs -> foldl (&&) true xs
            let rec any = fun xs -> foldl (||) false xs

            let rec take = fun n -> fun xs ->
                if n == 0 then Nil else
                match xs with
                | Nil -> Nil
                | Cons x xs -> Cons x (take (n - 1) xs)
            
            let rec drop = fun n -> fun xs ->
                if n == 0 then xs else
                match xs with
                | Nil -> Nil
                | Cons x xs -> drop (n - 1) xs
            
            let rec append = fun xs -> fun ys ->
                match xs with
                | Nil -> ys
                | Cons x xs -> Cons x (append xs ys)
            
            let rec reverse = fun xs -> foldl (fun acc -> fun x -> Cons x acc) Nil xs

            let rec range = fun n ->
                if n == 0 then Nil else
                Cons (n - 1) (range (n - 1))

            let head = fun xs ->
                match xs with
                | Cons x _ -> x
            
            let tail = fun xs ->
                match xs with
                | Nil -> Nil
                | Cons x rest -> rest 
            
            let rec list_eq = fun xs -> fun ys ->
                match (xs, ys) with
                | (Nil, Nil) -> true
                | (Cons x xs, Cons y ys) -> (x == y) && (list_eq xs ys)
                | _ -> false
            
            let rec zip = fun xs -> fun ys ->
                match (xs, ys) with
                | (Nil, Nil) -> Nil
                | (Cons x xs, Cons y ys) -> Cons (Tuple2 x y) (zip xs ys)
                | _ -> Nil
        ";

        env.eval_file(prelude_source).unwrap()
    }
}
