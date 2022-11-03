pub mod string;

pub use string::*;

use crate::typecheck::typecheck::Type;

pub enum Value {
    String(string::String),
    Float(f32),
    Integer(i32),
}

impl From<string::String> for Value {
    fn from(x: string::String) -> Self {
        Value::String(x)
    }
}

impl From<i32> for Value {
    fn from(x: i32) -> Self {
        Value::Integer(x)
    }
}

impl From<f32> for Value {
    fn from(x: f32) -> Self {
        Value::Float(x)
    }
}

impl Value {
    pub fn get_type(&self) -> Type {
        match self {
            Value::String(_) => Type::String,
            Value::Float(_) => Type::Float,
            Value::Integer(_) => Type::Integer,
        }
    }
}
