use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    rc::Rc,
};

use rpds::{HashTrieMap, HashTrieSet};

use crate::{ast::TypeExpr, value::Value};

type ValueThunk = Rc<RefCell<Option<Value>>>;

#[derive(Clone, PartialEq)]
pub struct Env {
    pub vars: HashTrieMap<String, ValueThunk>,
    pub typings: HashTrieMap<String, TypeExpr>,
}

impl Debug for Env {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(self.vars.into_iter().map(|(k, v)| {
                (
                    k,
                    v.borrow()
                        .as_ref()
                        .map(|o| o.to_string())
                        .unwrap_or("None".to_string()),
                )
            }))
            .finish()
    }
}

impl Display for Env {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut vars = self.vars.iter().collect::<Vec<_>>();
        vars.sort_by_key(|(k, _)| (*k).clone());
        let mut typings = self.typings.iter().collect::<Vec<_>>();
        typings.sort_by_key(|(k, _)| (*k).clone());
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

    pub fn extend_value(&self, name: &str, value: Value) -> Self {
        Self {
            vars: self
                .vars
                .insert(name.to_string(), Rc::new(RefCell::new(Some(value)))),
            typings: self.typings.clone(),
        }
    }

    pub fn extend_thunk(&self, name: &str, value: ValueThunk) -> Self {
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

    pub fn resolve_value(&self, name: &str) -> Result<Option<Value>, String> {
        match self.vars.get(name) {
            Some(value) => Ok(value.borrow().clone()),
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
