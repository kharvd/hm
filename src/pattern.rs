use crate::{ast::ExprPattern, env::Env, value::Value};

pub fn try_pattern_match(
    env: &Env,
    eval_expr: &Value,
    choice_pattern: &ExprPattern,
) -> Option<Env> {
    match (eval_expr, choice_pattern) {
        (Value::Int(i), ExprPattern::Int(j)) if i == j => Some(env.clone()),
        (Value::Bool(b), ExprPattern::Bool(c)) if b == c => Some(env.clone()),
        (_, ExprPattern::Wildcard) => Some(env.clone()),
        (Value::Data(name, args), ExprPattern::Constructor(constructor_name, constructor_args)) => {
            if name == constructor_name && args.len() == constructor_args.len() {
                let mut new_env = env.clone();
                for (arg, pattern_arg) in args.iter().zip(constructor_args.iter()) {
                    if let Some(bound_env) = try_pattern_match(&new_env, arg, pattern_arg) {
                        new_env = bound_env;
                    } else {
                        return None;
                    }
                }
                Some(new_env)
            } else {
                None
            }
        }
        (v, ExprPattern::Variable(name2)) => Some(env.extend(&name2, v.clone())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use crate::{
        ast::ExprPattern, env::Env, p_constructor, p_int, p_var, p_wildcard,
        pattern::try_pattern_match, value::Value,
    };

    #[test]
    fn test_constant() {
        let env = Env::new();
        let res = try_pattern_match(&env, &Value::Int(42), &p_int!(42));
        assert_eq!(res, Some(env));
    }

    #[test]
    fn test_wildcard() {
        let env = Env::new();
        let res = try_pattern_match(&env, &Value::Int(42), &p_wildcard!());
        assert_eq!(res, Some(env));
    }

    #[test]
    fn test_binding() {
        let env = Env::new();
        let res = try_pattern_match(&env, &Value::Int(42), &p_var!("x"));
        assert_eq!(res, Some(env.extend("x", Value::Int(42))));
    }

    #[test]
    fn test_constructor_simple() {
        let env = Env::new();
        let res = try_pattern_match(
            &env,
            &Value::Data("True".to_string(), vec![]),
            &p_constructor!("True"),
        );
        assert_eq!(res, Some(env));
    }

    #[test]
    fn test_constructor_binding() {
        let env = Env::new();
        let res = try_pattern_match(
            &env,
            &Value::Data("Pair".to_string(), vec![Value::Int(1), Value::Bool(true)]),
            &p_constructor!("Pair", p_var!("x"), p_var!("y")),
        );
        assert_eq!(
            res,
            Some(
                env.extend("x", Value::Int(1))
                    .extend("y", Value::Bool(true))
            )
        );
    }

    #[test]
    fn test_constructor_nested() {
        let env = Env::new();
        let res = try_pattern_match(
            &env,
            &Value::Data(
                "Pair".to_string(),
                vec![
                    Value::Int(1),
                    Value::Data("Pair".to_string(), vec![Value::Bool(true), Value::Int(2)]),
                ],
            ),
            &p_constructor!(
                "Pair",
                p_var!("x"),
                p_constructor!("Pair", p_var!("y"), p_var!("z"))
            ),
        );
        assert_eq!(
            res,
            Some(
                env.extend("x", Value::Int(1))
                    .extend("y", Value::Bool(true))
                    .extend("z", Value::Int(2))
            )
        );
    }
}
