use std::{fmt::Debug, rc::Rc};

use rpds::HashTrieMap;

use crate::{ast::TypeExpr, value::Value};

#[derive(Clone, PartialEq)]
pub struct Env {
    vars: HashTrieMap<String, Value>,
    typings: HashTrieMap<String, Rc<TypeExpr>>,
}

impl Debug for Env {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Env").finish()
    }
}

impl Env {
    pub fn new() -> Self {
        Self {
            vars: HashTrieMap::new(),
            typings: HashTrieMap::new(),
        }
    }

    pub fn extend(&self, name: &str, value: Value) -> Self {
        Self {
            vars: self.vars.insert(name.to_string(), value),
            typings: self.typings.clone(),
        }
    }

    pub fn extend_type(&self, name: &str, var_type: Rc<TypeExpr>) -> Self {
        Self {
            vars: self.vars.clone(),
            typings: self.typings.insert(name.to_string(), var_type),
        }
    }

    pub fn resolve_value(&self, name: &String) -> Result<Value, String> {
        match self.vars.get(name) {
            Some(value) => Ok(value.clone()),
            None => Err(format!("Unknown identifier {}", name)),
        }
    }

    pub fn resolve_type(&self, name: &String) -> Result<Rc<TypeExpr>, String> {
        match self.typings.get(name) {
            Some(value) => Ok(value.clone()),
            None => Err(format!("Unknown identifier {}", name)),
        }
    }
}
