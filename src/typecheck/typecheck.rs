use std::{collections::HashMap, ops::RangeInclusive};

use super::schema::Schema;
use crate::parser::ast::{self, ExprKind};

// TODO: double check https://developers.cloudflare.com/ruleset-engine/rules-language/values/#final-notes and
//       add tests for it.

#[derive(Debug, thiserror::Error)]
pub enum TypeCheckError {
    #[error(transparent)]
    InvalidFloatFormat(#[from] std::num::ParseFloatError),
    #[error(transparent)]
    InvalidIntegerFormat(#[from] std::num::ParseIntError),
    #[error(transparent)]
    InvalidIpFormat(#[from] std::net::AddrParseError),
    #[error(transparent)]
    InvalidIpCidrFormat(#[from] cidr::errors::NetworkParseError),

    #[error("expected types {expected} but found {found:?}")]
    MismatchedType { expected: String, found: Type },

    #[error("can't convert type {from:?} to {to:?}")]
    CantConvertTypes { from: Type, to: Type },

    #[error("could not infer type of the expression")]
    CouldNotInferTypeOfExpr,

    #[error("undefined field {0}")]
    UndefinedField(String),

    #[error("undefined function {0} or wrong parameter type/numbers")]
    UndefinedFunction(String),

    #[error("could not parse regex")]
    CouldNotParseRegex(#[from] regex::Error),

    // TODO: this is a temporary catch-all generic related error.
    #[error("invalid generic usage: {0}")]
    InvalidGenericUsage(String),

    #[error("this should be unreachable")]
    Unreachable,
}

#[derive(Default, Clone)]
pub enum Type {
    #[default]
    Placeholder,
    Infer,
    ConstString(String),
    ConstInteger(isize),
    ConstFloat(f32),
    ConstIpv4(std::net::Ipv4Addr),
    ConstIpv6(std::net::Ipv6Addr),
    ConstIpv4Cidr(cidr::Ipv4Cidr),
    ConstIpv6Cidr(cidr::Ipv6Cidr),
    ConstRegex(regex::Regex),
    Bool,
    String,
    Integer,
    Float,
    Ip,
    IpCidr,
    IPv4,
    IPv6,
    Ipv4Cidr,
    Ipv6Cidr,
    Regex,
    Array(Box<Type>),
    Map(Box<Type>, Box<Type>),
    Option(Box<Type>),
    Iterator(Box<Type>),
    Generic(usize),
}

impl std::fmt::Debug for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Placeholder => write!(f, "Placeholder"),
            Self::Infer => write!(f, "Infer"),
            Self::ConstString(_) => write!(f, "ConstString"),
            Self::ConstInteger(_) => write!(f, "ConstInteger"),
            Self::ConstFloat(_) => write!(f, "ConstFloat"),
            Self::ConstIpv4(_) => write!(f, "ConstIpv4"),
            Self::ConstIpv6(_) => write!(f, "ConstIpv6"),
            Self::ConstIpv4Cidr(_) => write!(f, "ConstIpv4Cidr"),
            Self::ConstIpv6Cidr(_) => write!(f, "ConstIpv6Cidr"),
            Self::ConstRegex(_) => write!(f, "ConstRegex"),
            Self::Bool => write!(f, "Bool"),
            Self::String => write!(f, "String"),
            Self::Integer => write!(f, "Integer"),
            Self::Float => write!(f, "Float"),
            Self::IPv4 => write!(f, "IPv4"),
            Self::Ip => write!(f, "IP"),
            Self::IpCidr => write!(f, "IpCidr"),
            Self::IPv6 => write!(f, "Ipv6"),
            Self::Ipv4Cidr => write!(f, "Ipv4Cidr"),
            Self::Ipv6Cidr => write!(f, "Ipv6Cidr"),
            Self::Regex => write!(f, "Regex"),
            Self::Option(arg0) => write!(f, "Option({:?})", arg0),
            Self::Array(arg0) => write!(f, "Array({:?})", arg0),
            Self::Iterator(arg0) => write!(f, "Iterator({:?})", arg0),
            Self::Map(l, r) => write!(f, "Map({:?}, {:?})", l, r),
            Self::Generic(x) => write!(f, "T{}", x),
        }
    }
}

impl Type {
    pub fn try_convert_to(&self, other: &Type) -> Option<Type> {
        // FIXME: these conversions are temporary and probably not correct
        use Type::*;
        match (self, other) {
            (Placeholder, _) => Some(other.clone()),
            (ConstInteger(_), Integer)
            | (ConstString(_), String)
            | (ConstFloat(_), Float)
            | (ConstIpv4(_), IPv4)
            | (ConstIpv6(_), IPv6)
            | (ConstIpv4Cidr(_), Ipv4Cidr)
            | (ConstIpv6Cidr(_), Ipv6Cidr)
            | (ConstRegex(_), Regex)
            | (Ip, IpCidr)
            | (IPv4, Ipv4Cidr)
            | (IPv6, Ipv6Cidr) => Some(other.clone()),
            (a, b) if a == b => Some(other.clone()),
            (ConstInteger(_) | Integer, ConstInteger(_)) => Some(Integer),
            (ConstString(_) | String, ConstString(_)) => Some(String),
            (ConstFloat(_) | Float, ConstFloat(_)) => Some(Float),
            (ConstIpv4(_) | IPv4, ConstIpv4(_)) => Some(IPv4),
            (ConstIpv6(_) | IPv6, ConstIpv6(_)) => Some(IPv6),
            (ConstIpv4(_) | ConstIpv4Cidr(_) | Ipv4Cidr, ConstIpv4Cidr(_)) => Some(Ipv4Cidr),
            (ConstIpv6(_) | ConstIpv6Cidr(_) | Ipv6Cidr, ConstIpv6Cidr(_)) => Some(Ipv6Cidr),
            (ConstRegex(_) | Regex, ConstRegex(_)) => Some(Regex),
            _ => None,
        }
    }

    pub fn is_generic(&self) -> bool {
        match self {
            Type::Array(a) => a.is_generic(),
            Type::Map(a, b) => a.is_generic() || b.is_generic(),
            Type::Option(a) => a.is_generic(),
            Type::Iterator(a) => a.is_generic(),
            Type::Generic(_) => true,
            _ => false,
        }
    }

    pub fn is_iterator(&self) -> bool {
        match self {
            Type::Array(a) => a.is_iterator(),
            Type::Map(a, b) => a.is_iterator() ^ b.is_iterator(),
            Type::Option(a) => a.is_iterator(),
            Type::Iterator(_) => true,
            _ => false,
        }
    }

    /// convert inner iterator type to it's inner type
    pub fn unwrap_iterator(&self) -> Type {
        match self {
            Type::Array(a) => Type::Array(Box::new(a.unwrap_iterator())),
            Type::Map(a, b) => {
                Type::Map(Box::new(a.unwrap_iterator()), Box::new(b.unwrap_iterator()))
            }
            Type::Option(a) => Type::Option(Box::new(a.unwrap_iterator())),
            Type::Iterator(x) => *x.clone(),
            x => x.clone(),
        }
    }

    fn replace_generic(&self, n: usize, t: Type) -> Type {
        match self {
            Type::Array(a) => Type::Array(Box::new(a.replace_generic(n, t))),
            Type::Map(a, b) => Type::Map(
                Box::new(a.replace_generic(n, t.clone())),
                Box::new(b.replace_generic(n, t)),
            ),
            Type::Option(a) => Type::Option(Box::new(a.replace_generic(n, t))),
            Type::Iterator(a) => Type::Iterator(Box::new(a.replace_generic(n, t))),
            Type::Generic(x) => {
                if *x == n {
                    t
                } else {
                    self.clone()
                }
            }
            x => x.clone(),
        }
    }

    fn try_find_generic_types(
        &self,
        other: &Type,
        storage: &mut HashMap<usize, Type>,
    ) -> Result<(), TypeCheckError> {
        match (self, other) {
            (Type::Array(a) | Type::Option(a), Type::Array(b) | Type::Option(b))
                if a.is_generic() || b.is_generic() =>
            {
                a.try_find_generic_types(b, storage)?;
            }
            (Type::Option(a), Type::Option(b)) if a.is_generic() || b.is_generic() => {
                a.try_find_generic_types(b, storage)?;
            }
            (Type::Iterator(a), Type::Iterator(b)) if a.is_generic() || b.is_generic() => {
                a.try_find_generic_types(b, storage)?;
            }
            (Type::Map(aa, ab), Type::Map(ba, bb))
                if aa.is_generic() || ab.is_generic() || ba.is_generic() || bb.is_generic() =>
            {
                aa.try_find_generic_types(aa, storage)?;
                ab.try_find_generic_types(bb, storage)?;
            }
            (Type::Generic(n), other) | (other, Type::Generic(n)) => {
                if let Some(x) = storage.insert(*n, other.clone()) {
                    return Err(TypeCheckError::InvalidGenericUsage(format!(
                        "generic parameter {} type can't be both {:?} and {:?}",
                        n, x, other
                    )));
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub fn try_expand_generic(
        &self,
        args: &[Type],
        params: &[Type],
    ) -> Result<Type, TypeCheckError> {
        let mut generic_types = HashMap::new();

        for (arg, par) in args.iter().zip(params) {
            arg.try_find_generic_types(par, &mut generic_types)?;
        }

        let mut r_type = self.clone();

        for (n, t) in generic_types {
            r_type = r_type.replace_generic(n, t);
        }

        if r_type.is_generic() {
            return Err(TypeCheckError::InvalidGenericUsage(format!(
                "cannot expand type of {:?} from args({:?}) and params({:?}",
                self, args, params
            )));
        }

        Ok(r_type)
    }
}

impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ConstString(l0), Self::ConstString(r0)) => l0 == r0,
            (Self::ConstInteger(l0), Self::ConstInteger(r0)) => l0 == r0,
            (Self::ConstFloat(l0), Self::ConstFloat(r0)) => l0 == r0,
            (Self::ConstIpv4(l0), Self::ConstIpv4(r0)) => l0 == r0,
            (Self::ConstIpv6(l0), Self::ConstIpv6(r0)) => l0 == r0,
            (Self::ConstIpv4Cidr(l0), Self::ConstIpv4Cidr(r0)) => l0 == r0,
            (Self::ConstIpv6Cidr(l0), Self::ConstIpv6Cidr(r0)) => l0 == r0,
            (Self::ConstRegex(l0), Self::ConstRegex(r0)) => l0.as_str() == r0.as_str(),
            (Self::Array(l0), Self::Array(r0)) => l0 == r0,
            (Self::Map(l0, l1), Self::Map(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Option(l0), Self::Option(r0)) => l0 == r0,
            (Self::Iterator(l0), Self::Iterator(r0)) => l0 == r0,
            (Self::Generic(_), _) => true,
            (_, Self::Generic(_)) => true,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

fn find_atom_type(input: &str, atom: &ast::Atom) -> Result<Type, TypeCheckError> {
    let t = match atom {
        ast::Atom::StringLiteral(x) => {
            // skip first and last character.
            // Example data: "test", 'test'
            let string = &input[(*x.range.start() + 1)..=(*x.range.end() - 1)];
            Type::ConstString(string.to_owned())
        }
        ast::Atom::NumberLiteral(x) => {
            let num = &input[x.range.clone()];
            if num.contains(".") {
                Type::ConstFloat(num.parse()?)
            } else {
                Type::ConstInteger(num.parse()?)
            }
        }
        ast::Atom::Ipv4(x) => Type::ConstIpv4(input[x.range.clone()].parse()?),
        ast::Atom::Ipv6(x) => Type::ConstIpv6(input[x.range.clone()].parse()?),
        ast::Atom::Ipv4Cidr(x) => {
            Type::ConstIpv4Cidr(input[x.range.clone()].parse::<cidr::Ipv4Inet>()?.network())
        }
        ast::Atom::Ipv6Cidr(x) => {
            Type::ConstIpv6Cidr(input[x.range.clone()].parse::<cidr::Ipv6Inet>()?.network())
        }
    };

    Ok(t)
}

pub fn infer_and_typecheck(
    input: &str,
    expr: &mut ast::Expr,
    schema: &Schema,
) -> Result<(), TypeCheckError> {
    macro_rules! handle_binary_op {
        ($lhs:ident, $rhs:ident) => {{
            infer_and_typecheck(input, $lhs, schema)?;
            infer_and_typecheck(input, $rhs, schema)?;
            // if the lhs is an iterator, binary operators like lhs == rhs
            // is a filter over that iterator. for example http.headers[*] == "test"
            // is equivalent to `http.headers.iter().filter(|x| x == "test")
            if let Type::Iterator(_) = $lhs.r#type {
                expr.r#type = $lhs.r#type.clone();
            } else {
                expr.r#type = Type::Bool;
            }
        }};
    }

    macro_rules! expect {
        ($expr:expr => $($type:pat),*) => {
            {
                #[allow(unused_imports)]
                use Type::*;
                if !matches!(&$expr.r#type, $(| $type)*) {
                    return Err(TypeCheckError::MismatchedType {
                        expected: stringify!($($type)or*).to_string(),
                        found: $expr.r#type.clone()
                    })
                }
            }
        };
    }

    match &mut expr.inner {
        ExprKind::Atom(x) => {
            let t = find_atom_type(input, x)?;
            expr.r#type = t;
        }
        ExprKind::Not(x) => {
            infer_and_typecheck(input, x, schema)?;
            expect!(x => Bool);
            expr.r#type = Type::Bool;
        }

        ExprKind::Eq(lhs, rhs) => handle_binary_op!(lhs, rhs),
        ExprKind::Gt(lhs, rhs) => handle_binary_op!(lhs, rhs),
        ExprKind::Gte(lhs, rhs) => handle_binary_op!(lhs, rhs),
        ExprKind::Lt(lhs, rhs) => handle_binary_op!(lhs, rhs),
        ExprKind::Lte(lhs, rhs) => handle_binary_op!(lhs, rhs),
        ExprKind::In(lhs, rhs) => handle_binary_op!(lhs, rhs),
        ExprKind::Contains(lhs, rhs) => handle_binary_op!(lhs, rhs),
        ExprKind::Matches(lhs, rhs) => {
            infer_and_typecheck(input, lhs, schema)?;
            expect!(lhs => String, ConstString(_));
            infer_and_typecheck(input, rhs, schema)?;
            rhs.r#type = match &rhs.r#type {
                Type::ConstString(x) => {
                    let pattern = regex::Regex::new(x)?;
                    Type::ConstRegex(pattern)
                }
                Type::String => Type::Regex,
                x => {
                    return Err(TypeCheckError::MismatchedType {
                        expected: String::from("RegexString"),
                        found: x.clone(),
                    })
                }
            };

            expr.r#type = Type::Bool;
        }
        ExprKind::Field(x) => {
            let start_idx = x.chain.first().unwrap().range.start();
            let end_idx = x.chain.last().unwrap().range.end();
            let name = &input[RangeInclusive::new(*start_idx, *end_idx)];
            if let Some(field) = schema.find_field(name) {
                expr.r#type = field.r#type.clone();
            } else {
                return Err(TypeCheckError::UndefinedField(name.to_string()));
            }
        }
        ExprKind::FunctionCall(x) => {
            // if there are no exact definition for a function with an iterator parameter, we should take the inner type
            // and act as if the function is called like param.iter().map(|x| fn(x)). It's also obvious that the return type of
            // the function will change.
            let start_idx = x.name.first().unwrap().range.start();
            let end_idx = x.name.last().unwrap().range.end();
            let name = &input[RangeInclusive::new(*start_idx, *end_idx)];

            for par in &mut x.args {
                infer_and_typecheck(input, par, schema)?;
            }

            let args: Vec<_> = x.args.iter().map(|x| x.r#type.clone()).collect();
            if let Some(func) = schema.find_function(name, args.as_slice()) {
                if func.r_type.is_generic() {
                    // find the actual generic type
                    expr.r#type = func
                        .r_type
                        .try_expand_generic(args.as_slice(), func.par_types.as_slice())?;
                } else {
                    expr.r#type = func.r_type.clone();
                }
            } else {
                // check if one and only one of the args is an iterator.
                let arg_iterator_count = args.iter().filter(|x| x.is_iterator()).count();
                match arg_iterator_count {
                    1 => {
                        // if args are like (String, Iterator<String>, Number) convert it to (String, String, Number)
                        // or (String, A<B<Iterator<String>>>, Number) to (String, A<B<String>>, Number)

                        // TODO: there are some duplication here

                        let args: Vec<_> = args.iter().map(|x| x.unwrap_iterator()).collect();
                        if let Some(func) = schema.find_function(name, args.as_slice()) {
                            if func.r_type.is_generic() {
                                // find the actual generic type
                                expr.r#type =
                                    Type::Iterator(Box::new(func.r_type.try_expand_generic(
                                        args.as_slice(),
                                        func.par_types.as_slice(),
                                    )?));
                            } else {
                                expr.r#type = Type::Iterator(Box::new(func.r_type.clone()));
                            }
                        } else {
                            let args: Vec<_> = args.iter().map(|x| format!("{:?}", x)).collect();
                            return Err(TypeCheckError::UndefinedFunction(format!(
                                "{}({})",
                                name,
                                args.join(", ")
                            )));
                        }
                    }
                    0 => {
                        let args: Vec<_> = args.iter().map(|x| format!("{:?}", x)).collect();
                        return Err(TypeCheckError::UndefinedFunction(format!(
                            "{}({})",
                            name,
                            args.join(", ")
                        )));
                    }
                    n => {
                        let args: Vec<_> = args.iter().map(|x| format!("{:?}", x)).collect();
                        return Err(TypeCheckError::UndefinedFunction(format!(
                            "{}({}). invalid usage of [*] operator. only a single iterator can be used as arg but found {}",
                            name,
                            args.join(", "),
                            n
                        )));
                    }
                }
            }
        }
        ExprKind::Array {
            elements,
            start: _,
            end: _,
        } => {
            let mut t = Type::Placeholder;
            for e in elements {
                infer_and_typecheck(input, e, schema)?;
                if let Some(to) = t.try_convert_to(&e.r#type) {
                    t = to;
                } else {
                    return Err(TypeCheckError::CantConvertTypes {
                        from: t,
                        to: e.r#type.clone(),
                    });
                }
            }

            expr.r#type = Type::Array(Box::new(t));
        }
        ExprKind::Indexed(x) => match &mut x.kind {
            ast::Index::Star => {
                infer_and_typecheck(input, &mut x.expr, schema)?;
                match &x.expr.r#type {
                    Type::Map(_, val_type) => {
                        expr.r#type = Type::Iterator(val_type.clone());
                    }
                    Type::Array(val_type) | Type::Iterator(val_type) => {
                        expr.r#type = Type::Iterator(val_type.clone());
                    }
                    x => {
                        return Err(TypeCheckError::MismatchedType {
                            expected: String::from("Map | Array"),
                            found: x.clone(),
                        })
                    }
                };
            }
            ast::Index::Expr(index_expr) => {
                infer_and_typecheck(input, &mut x.expr, schema)?;
                infer_and_typecheck(input, index_expr, schema)?;

                let (key_type, val_type) = match &x.expr.r#type {
                    Type::Map(key_type, val_type) => (key_type, val_type),
                    x => {
                        return Err(TypeCheckError::MismatchedType {
                            expected: String::from("Map"),
                            found: x.clone(),
                        })
                    }
                };

                if key_type.clone() != Box::new(index_expr.r#type.clone()) {
                    return Err(TypeCheckError::MismatchedType {
                        expected: format!("{:?}", key_type),
                        found: index_expr.r#type.clone(),
                    });
                }

                expr.r#type = *val_type.clone();
            }
        },
        ExprKind::DynamicField(_) => {
            // TODO: incomplete
            expr.r#type = Type::Infer;
        }
        ExprKind::BitwiseAnd(l, r) | ExprKind::BitwiseOr(l, r) => {
            infer_and_typecheck(input, l, schema)?;
            infer_and_typecheck(input, r, schema)?;
            expect!(l => Integer, ConstInteger(_));
            expect!(r => Integer, ConstInteger(_));
            expr.r#type = Type::Integer;
        }

        // TODO: incomplete:
        ExprKind::Xor(lhs, rhs) => handle_binary_op!(lhs, rhs),
        ExprKind::Add(lhs, rhs) => handle_binary_op!(lhs, rhs),
        ExprKind::Sub(lhs, rhs) => handle_binary_op!(lhs, rhs),
        ExprKind::Mul(lhs, rhs) => handle_binary_op!(lhs, rhs),
        ExprKind::Mod(lhs, rhs) => handle_binary_op!(lhs, rhs),
        ExprKind::Div(lhs, rhs) => handle_binary_op!(lhs, rhs),
        ExprKind::Or(lhs, rhs) => handle_binary_op!(lhs, rhs),
        ExprKind::And(lhs, rhs) => handle_binary_op!(lhs, rhs),
        ExprKind::NEq(lhs, rhs) => handle_binary_op!(lhs, rhs),
    }

    if let Type::Placeholder = &expr.r#type {
        return Err(TypeCheckError::CouldNotInferTypeOfExpr);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{
        parser::parser::{parse, tests::ast_to_text_verbose},
        schema,
    };

    use super::*;

    fn test_suit(name: &str, schema: &Schema) {
        let tests = std::fs::read_to_string(name).unwrap();
        let tests: Vec<_> = tests
            .split("###############################")
            .map(|x| x.trim())
            .collect();
        for sample in tests.iter() {
            let (input, expected_output) = sample
                .split_once("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                .unwrap();
            let expected_output = expected_output.trim();
            let mut expr = match parse(input) {
                Ok(x) => x,
                Err(err) => {
                    panic!("\ninput: {}\n{}\n", input, err.explain(input))
                }
            };
            infer_and_typecheck(input, &mut expr, schema).unwrap();
            let o = ast_to_text_verbose(&expr, input);
            println!("{}", &o);
            assert_eq!(o, expected_output);
        }
    }

    #[test]
    fn test_cloudflare_samples() {
        let schema = schema! {
            functions: [
                fn first(Iterator(T(0))) -> Option(T(0)),
                fn first(Array(T(0))) -> Option(T(0)),
                fn any(Iterator(T(0))) -> Bool,
                fn any(Array(T(0))) -> Bool,
                fn lower(String) -> String,
                fn to_string(T(0)) -> String,
                fn url_decode(String) -> String,
            ],
            fields: [
                http.request.uri.path: String,
                host: String,
                http.host: String,
                ip.geoip.country: String,
                http.request.uri.query: String,
                cf.edge.server_port: Integer,
                ip.geoip.asnum: Integer,
                ip.src: Ip,
                cf.threat_score: Integer,
                cf.bot_management.score: Integer,
                ssl: Bool,
                http.request.headers.names: Array(String),
                http.request.body.form.values: Array(String),
            ]
        };
        test_suit("./src/typecheck/cloudflare_docs_sample.test", &schema);
    }
}
