pub mod value;

use crate::{
    parser::{
        self,
        ast::{Atom, Expr, ExprKind},
    },
    typecheck::typecheck::Type,
};
use std::marker::PhantomData;

pub trait IntoType {
    fn to_type(&self) -> Type;
}

pub enum Inst {
    LoadInt(i32),
    LoadFloat(f32),

    LoadConst(usize, Type),

    /// pop two strings from stack, concatenate them and
    /// push the new string on stack
    ConcatString,

    /// pop the a value from stack and convert it to string
    ToString(Type),

    AddInteger,
    AddFloat,

    /// convert integer to float
    ToFloat,
}

// TODO: separate into two VM and VMBuilder

#[derive(Default)]
pub struct VM<Output: IntoType> {
    // shared
    consts: Vec<u8>,
    insts: Vec<Inst>,
    _output_type: PhantomData<Output>,
}

#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("{0:?}")]
    ParseError(#[from] parser::parser::ParseError),

    #[error("{0:?}")]
    IAmLazy(&'static str),
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("expected {expected:?} got {got:?}")]
    UnexpectedType { expected: Type, got: Type },

    #[error("expected an item in stack but it was empty")]
    StackIsEmpty,
}

impl<T: IntoType> VM<T> {
    pub fn compile(code: &str) -> Result<Self, CompileError> {
        let expr = parser::parser::parse(code)?;
        let mut vm = Self {
            _output_type: PhantomData,
            consts: vec![],
            insts: vec![],
        };
        vm.compile_expr(&expr, code)?;
        Ok(vm)
    }

    pub fn run(&self) -> Result<T, RuntimeError> {
        let mut stack: Vec<value::Value> = Default::default();

        macro_rules! pop {
            ($t:ident) => {{
                match stack.pop() {
                    Some(value::Value::$t(x)) => x,
                    // if our code generation is correct we can skip these checks
                    // in the future if we got a way to verify this, this would be a
                    // good optimization.
                    Some(x) => {
                        return Err(RuntimeError::UnexpectedType {
                            expected: Type::$t,
                            got: x.get_type(),
                        })
                    }
                    None => return Err(RuntimeError::StackIsEmpty),
                }
            }};
        }

        macro_rules! push {
            ($v:expr) => {{
                stack.push($v.into());
            }};
        }

        let mut ic = 0;

        while let Some(inst) = self.insts.get(ic) {
            match inst {
                Inst::LoadConst(start, t) => {
                    let size = match t {
                        Type::String => std::mem::size_of::<value::String>(),
                        _ => todo!(),
                    };
                    push!(value::String::new_const(*start, size));
                }
                Inst::LoadInt(x) => push!(*x),
                Inst::LoadFloat(x) => push!(*x),
                Inst::ConcatString => {
                    let b = pop!(String);
                    let a = pop!(String);

                    let result = match (b, a) {
                        (value::String::Owned(a), value::String::Owned(mut b)) => {
                            b.extend(a);
                            value::String::Owned(b)
                        }
                        (value::String::Owned(a), value::String::Const { start, len }) => {
                            let mut result = Vec::with_capacity(a.len() + len);
                            result.extend_from_slice(&self.consts[start..start + len]);
                            result.extend(a);
                            value::String::Owned(result)
                        }
                        (value::String::Const { start, len }, value::String::Owned(mut b)) => {
                            b.extend(&self.consts[start..start + len]);
                            value::String::Owned(b)
                        }
                        (
                            value::String::Const {
                                start: start_a,
                                len: len_a,
                            },
                            value::String::Const {
                                start: start_b,
                                len: len_b,
                            },
                        ) => {
                            let mut result = Vec::with_capacity(len_a + len_b);
                            result.extend_from_slice(&self.consts[start_b..start_b + len_b]);
                            result.extend_from_slice(&self.consts[start_a..start_a + len_a]);
                            value::String::Owned(result)
                        }
                    };
                    push!(result);
                }
                Inst::ToString(x) => {
                    let result = match x {
                        Type::String => pop!(String),
                        _ => todo!(),
                    };

                    push!(result);
                }
                Inst::AddInteger => {
                    let b = pop!(Integer);
                    let a = pop!(Integer);
                    push!(b + a);
                }
                Inst::AddFloat => {
                    let b = pop!(Float);
                    let a = pop!(Float);
                    push!(b + a);
                }
                Inst::ToFloat => {
                    let x = pop!(Integer);
                    push!(x as f32);
                }
            }

            ic += 1;
        }
        todo!()
    }

    fn compile_expr(&mut self, expr: &Expr, code: &str) -> Result<(), CompileError> {
        macro_rules! inst {
            ($($val:expr),*) => {
                {
                    use Inst::*;
                    $(

                        self.insts.push($val);
                    )*
                }
            };
        }

        match &expr.inner {
            ExprKind::Atom(x) => self.compile_atom(code, x),
            ExprKind::Field(_) => todo!(),
            ExprKind::FunctionCall(_) => todo!(),
            ExprKind::Array {
                elements,
                start,
                end,
            } => todo!(),
            ExprKind::Indexed(_) => todo!(),
            ExprKind::DynamicField(_) => todo!(),
            ExprKind::Not(_) => todo!(),
            ExprKind::BitwiseAnd(_, _) => todo!(),
            ExprKind::BitwiseOr(_, _) => todo!(),
            ExprKind::Xor(_, _) => todo!(),
            ExprKind::Add(lhs, rhs) => {
                let lt = lhs.r#type.clone();
                let rt = lhs.r#type.clone();

                if lt.is_string() && rt.is_string() {
                    self.compile_expr(lhs, code)?;
                    self.compile_expr(rhs, code)?;
                    inst!(ConcatString);
                    Ok(())
                } else if lt.is_string() {
                    self.compile_expr(lhs, code)?;
                    self.compile_expr(rhs, code)?;
                    inst!(ToString(rt), ConcatString);
                    Ok(())
                } else if rt.is_string() {
                    self.compile_expr(lhs, code)?;
                    inst!(ToString(rt));
                    self.compile_expr(rhs, code)?;
                    inst!(ConcatString);
                    Ok(())
                } else if matches!(lt, Type::Integer) && matches!(rt, Type::Integer) {
                    self.compile_expr(lhs, code)?;
                    self.compile_expr(rhs, code)?;
                    inst!(AddInteger);
                    Ok(())
                } else if matches!(lt, Type::Integer) && matches!(rt, Type::Float) {
                    self.compile_expr(lhs, code)?;
                    inst!(ToFloat);
                    self.compile_expr(rhs, code)?;
                    inst!(AddFloat);
                    Ok(())
                } else if matches!(lt, Type::Float) && matches!(rt, Type::Integer) {
                    self.compile_expr(lhs, code)?;
                    self.compile_expr(rhs, code)?;
                    inst!(ToFloat);
                    inst!(AddFloat);
                    Ok(())
                } else if matches!(lt, Type::Float) && matches!(rt, Type::Float) {
                    self.compile_expr(lhs, code)?;
                    self.compile_expr(rhs, code)?;
                    inst!(AddFloat);
                    Ok(())
                } else {
                    Err(CompileError::IAmLazy(
                        "invalid types for lhs/rhs of the + operator",
                    ))
                }
            }
            ExprKind::Sub(_, _) => todo!(),
            ExprKind::Mul(_, _) => todo!(),
            ExprKind::Mod(_, _) => todo!(),
            ExprKind::Div(_, _) => todo!(),
            ExprKind::Or(_, _) => todo!(),
            ExprKind::And(_, _) => todo!(),
            ExprKind::Eq(_, _) => todo!(),
            ExprKind::NEq(_, _) => todo!(),
            ExprKind::Gt(_, _) => todo!(),
            ExprKind::Gte(_, _) => todo!(),
            ExprKind::Lt(_, _) => todo!(),
            ExprKind::Lte(_, _) => todo!(),
            ExprKind::Contains(_, _) => todo!(),
            ExprKind::Matches(_, _) => todo!(),
            ExprKind::In(_, _) => todo!(),
        }
    }

    fn compile_atom(&mut self, code: &str, atom: &Atom) -> Result<(), CompileError> {
        macro_rules! as_str {
            ($span:expr) => {{
                &code[(*$span.range.start() + 1)..=(*$span.range.end() - 1)]
            }};
        }
        match atom {
            Atom::StringLiteral(x) => {
                // TODO: handle escape characters
                // TODO: handle string interning
                let string = as_str!(x);
                let start = self.consts.len();
                let data = string.as_bytes();
                self.consts.extend_from_slice(data);
                self.insts.push(Inst::LoadConst(start, Type::String));
                Ok(())
            }
            Atom::NumberLiteral(x) => {
                let x = as_str!(x);
                match x.parse::<i32>() {
                    Ok(x) => self.insts.push(Inst::LoadInt(x)),
                    Err(_) => match x.parse::<f32>() {
                        Ok(x) => self.insts.push(Inst::LoadFloat(x)),
                        _ => return Err(CompileError::IAmLazy("invalid number literal")),
                    },
                }
                Ok(())
            }
            Atom::Ipv4(_) => todo!(),
            Atom::Ipv4Cidr(_) => todo!(),
            Atom::Ipv6(_) => todo!(),
            Atom::Ipv6Cidr(_) => todo!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_number_parsing() {
        assert!("1.2".parse::<i32>().is_err());
        assert!("1.2".parse::<f32>().is_ok());
    }
}
