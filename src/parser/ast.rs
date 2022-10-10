use crate::typecheck::typecheck;

pub use super::tokenizer::Span;

/// Optional index syntax for fields.
/// Example:
///   - req.headers["content-type"]
#[derive(Debug)]
pub enum Index {
    /// a.b[*]
    Star,
    /// a.b["test"]
    Expr(Box<Expr>),
}

/// An indexed expression
/// Example:
///   - a.b["hello"]
///   - (1 + 2)[*]
#[derive(Debug)]
pub struct Indexed {
    pub kind: Index,
    pub expr: Box<Expr>,
}

/// A predefined defined variable
#[derive(Debug)]
pub struct Field {
    /// Chain of field name
    /// Example:
    ///   - [Ident("abc"), Dot, Ident("cd")]
    pub chain: Vec<Span>,
}

/// A function call expression
#[derive(Debug)]
pub struct FunctionCall {
    /// function name
    /// Example:
    ///   - a.b()
    ///   - b()
    pub name: Vec<Span>,
    /// function parameters
    pub args: Vec<Expr>,
}

/// Smallest unit of eval-able part that doesn't
/// need recursion to result to a value
#[derive(Debug)]
pub enum Atom {
    /// An string literal
    /// Example:
    ///   - "abc"
    ///   - 'hello'
    ///   - "/(.*?)/"
    StringLiteral(Span),
    /// A number literal
    /// Example:
    ///   - 1.2
    ///   - -1.12
    ///   - 3
    ///   - 4
    NumberLiteral(Span),
    /// Ipv4 literal
    /// Example:
    ///   - 127.0.0.1
    Ipv4(Span),
    /// Ipv4 Cidr literal
    /// Example:
    ///   - 127.0.0.1/24
    Ipv4Cidr(Span),
    /// Ipv6 literal
    /// Example:
    ///   - 2001:0:eab:DEAD:0:A0:ABCD:4E
    Ipv6(Span),
    /// Ipv6 Cidr literal
    /// Example:
    ///   - 2001:0:eab:DEAD:0:A0:ABCD:4E/24
    Ipv6Cidr(Span),
}

/// An expression that can be resulted to a value
#[derive(Debug)]
pub enum ExprKind {
    /// see [`Atom`]
    Atom(Atom),
    /// see [`Field`]
    Field(Field),
    /// see [`FunctionCall`]
    FunctionCall(FunctionCall),
    /// An array literal:
    /// Example:
    ///   - {1 2 3}
    ///   - {"hi"}
    ///   - {"hello" "hi"}
    Array {
        elements: Vec<Expr>,
        start: Span,
        end: Span,
    },
    /// see [`Indexed`]
    Indexed(Indexed),
    /// $hello
    DynamicField(Field),
    /// Not operator
    /// Example:
    ///   - not req.ssl
    ///   - not (not 1 == 2)
    Not(Box<Expr>),
    /// & operator
    /// Example:
    ///   - 1 & 2
    ///   - 1 bitwise_and 2
    BitwiseAnd(Box<Expr>, Box<Expr>),
    /// | operator
    /// Example:
    ///   - 1 | 2
    ///   - 1 bitwise_or 2
    BitwiseOr(Box<Expr>, Box<Expr>),
    /// xor operator
    /// Example:
    ///   - 1 xor 2
    ///   - 1 ^ 2
    Xor(Box<Expr>, Box<Expr>),
    /// + operator
    /// Example:
    ///   - 1 + 2
    Add(Box<Expr>, Box<Expr>),
    /// - operator
    /// Example:
    ///   - 1 - 2
    Sub(Box<Expr>, Box<Expr>),
    /// * operator
    /// Example:
    ///   - 1 * 2
    Mul(Box<Expr>, Box<Expr>),
    /// % operator
    /// Example:
    ///   - 1 % 2
    Mod(Box<Expr>, Box<Expr>),
    /// / operator
    /// Example:
    ///   - 1 / 2
    Div(Box<Expr>, Box<Expr>),
    /// or operator
    /// Example:
    ///   - 1 == 2 or 2 == 1
    ///   - 1 == 2 || 2 == 1
    Or(Box<Expr>, Box<Expr>),
    /// and operator
    /// Example:
    ///   - 1 == 2 and 2 == 1
    ///   - 1 == 2 && 2 == 1
    And(Box<Expr>, Box<Expr>),
    /// == operator
    /// Example:
    ///   - 1 == 2
    Eq(Box<Expr>, Box<Expr>),
    /// != operator
    /// Example:
    ///   - 1 != 2
    NEq(Box<Expr>, Box<Expr>),
    /// > operator
    /// Example:
    ///   - 1 > 2
    Gt(Box<Expr>, Box<Expr>),
    /// >= operator
    /// Example:
    ///   - 1 >= 2
    Gte(Box<Expr>, Box<Expr>),
    /// < operator
    /// Example:
    ///   - 1 <= 2
    Lt(Box<Expr>, Box<Expr>),
    /// <= operator
    /// Example:
    ///   - 1 / 2
    Lte(Box<Expr>, Box<Expr>),
    /// contains operator
    /// Example:
    ///   - "hello" contains "o"
    Contains(Box<Expr>, Box<Expr>),
    /// matches operator
    /// Example:
    ///   - req.body matches "<regex>"
    Matches(Box<Expr>, Box<Expr>),
    /// in operator
    /// Example:
    ///   - req.ip in { 1.2.3.4   2.3.4.5  }
    In(Box<Expr>, Box<Expr>),
}

#[derive(Debug)]
pub struct Expr {
    pub r#type: typecheck::Type,
    pub inner: ExprKind,
}

impl From<ExprKind> for Expr {
    fn from(inner: ExprKind) -> Self {
        Self {
            r#type: typecheck::Type::Placeholder,
            inner,
        }
    }
}
