use std::fmt::{Debug, Display};

use rpds::{HashTrieMap, HashTrieSet};

use crate::{ast::TypeExpr, value::Value};

#[derive(Clone, PartialEq)]
pub struct Env {
    pub vars: HashTrieMap<String, Value>,
    pub typings: HashTrieMap<String, TypeExpr>,
}

impl Debug for Env {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Env").finish()
    }
}

impl Display for Env {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut vars = self.vars.iter().collect::<Vec<_>>();
        vars.sort_by_key(|(k, _)| k.clone());
        let mut typings = self.typings.iter().collect::<Vec<_>>();
        typings.sort_by_key(|(k, _)| k.clone());
        write!(f, "Env {{ vars: {:?}, typings: {:?} }}", vars, typings)
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

    pub fn extend_type(&self, name: &str, var_type: TypeExpr) -> Self {
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

    pub fn resolve_type(&self, name: &String) -> Result<TypeExpr, String> {
        match self.typings.get(name) {
            Some(value) => Ok(value.clone()),
            None => Err(format!("Unknown identifier {}", name)),
        }
    }

    pub fn free_type_vars(&self) -> HashTrieSet<String> {
        let mut free_vars = HashTrieSet::new();
        for (_, ty) in self.typings.iter() {
            for var in ty.free_variables().iter() {
                free_vars.insert_mut(var.clone());
            }
        }
        free_vars
    }
}
