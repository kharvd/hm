use std::fmt::Debug;

use rpds::HashTrieMap;

use crate::value::Value;

#[derive(Clone, PartialEq)]
pub struct Env {
    data: HashTrieMap<String, Value>,
}

impl Debug for Env {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Env").finish()
    }
}

impl Env {
    pub fn new() -> Self {
        Self {
            data: HashTrieMap::new(),
        }
    }

    pub fn extend(&self, name: &str, value: Value) -> Self {
        Self {
            data: self.data.insert(name.to_string(), value),
        }
    }

    pub fn resolve(&self, name: &String) -> Result<Value, String> {
        match self.data.get(name) {
            Some(value) => Ok(value.clone()),
            None => Err(format!("Unknown identifier {}", name)),
        }
    }
}
