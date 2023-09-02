use crate::{
    builtins::{BoolBinOps, BoolUnaryOps, IntBinOps, IntUnaryOps},
    env::Env,
    value::Value,
};

impl Env {
    pub fn prelude() -> Env {
        let env = Env::new()
            .extend("plus", Value::new_builtin(IntBinOps::Plus))
            .extend("minus", Value::new_builtin(IntBinOps::Minus))
            .extend("mult", Value::new_builtin(IntBinOps::Mult))
            .extend("div", Value::new_builtin(IntBinOps::Div))
            .extend("lt", Value::new_builtin(IntBinOps::Lt))
            .extend("leq", Value::new_builtin(IntBinOps::Leq))
            .extend("gt", Value::new_builtin(IntBinOps::Gt))
            .extend("geq", Value::new_builtin(IntBinOps::Geq))
            .extend("neg", Value::new_builtin(IntUnaryOps::Neg))
            .extend("not", Value::new_builtin(BoolUnaryOps::Not))
            .extend("and", Value::new_builtin(BoolBinOps::And))
            .extend("or", Value::new_builtin(BoolBinOps::Or))
            .extend("xor", Value::new_builtin(BoolBinOps::Xor))
            .extend("fix", Value::Fix);

        let prelude_source = "
            val plus : int -> int -> int
            val minus : int -> int -> int
            val mult : int -> int -> int
            val div : int -> int -> int
            val lt : int -> int -> bool
            val leq : int -> int -> bool
            val gt : int -> int -> bool
            val geq : int -> int -> bool
            val neg : int -> int
            val not : bool -> bool
            val and : bool -> bool -> bool
            val or : bool -> bool -> bool
            val xor : bool -> bool -> bool

            val fix : ('a -> 'a) -> 'a
            let eq = fun x -> fun y -> and (leq x y) (geq x y)
                
            data Pair 'a 'b = Pair 'a 'b
            let fst = fun p -> match p with | Pair x y -> x
            let snd = fun p -> match p with | Pair x y -> y

            data List 'a = Nil | Cons 'a (List 'a)
            let nil = Nil
            let cons = fun x -> fun xs -> Cons x xs
            let rec len = fun xs -> 
                match xs with 
                | Nil -> 0 
                | Cons x xs -> plus 1 (len xs)

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
            
            let rec sum = fun xs -> foldl plus 0 xs
            let rec product = fun xs -> foldl mult 1 xs
            let rec all = fun xs -> foldl and true xs
            let rec any = fun xs -> foldl or false xs

            let rec take = fun n -> fun xs ->
                if eq n 0 then Nil else
                match xs with
                | Nil -> Nil
                | Cons x xs -> Cons x (take (minus n 1) xs)
            
            let rec drop = fun n -> fun xs ->
                if eq n 0 then xs else
                match xs with
                | Nil -> Nil
                | Cons x xs -> drop (minus n 1) xs
            
            let rec append = fun xs -> fun ys ->
                match xs with
                | Nil -> ys
                | Cons x xs -> Cons x (append xs ys)
            
            let rec reverse = fun xs -> foldl (fun acc -> fun x -> Cons x acc) Nil xs

            let rec range = fun n ->
                if eq n 0 then Nil else
                Cons (minus n 1) (range (minus n 1))

            let head = fun xs ->
                match xs with
                | Cons x _ -> x
            
            let tail = fun xs ->
                match xs with
                | Nil -> Nil
                | Cons x rest -> rest
        ";

        env.eval_file(prelude_source).unwrap()
    }
}
