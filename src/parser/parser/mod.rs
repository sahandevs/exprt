//! Grammar:
//!
//! ```text
//! Chain -> Ident (. Ident)*
//!
//! Field -> Chain ("[" ("*" | Expr) "]")?
//!
//! FunctionCall -> Chain "(" (Expr),* ")"
//!
//! Atom ->
//!     | StringLiteral
//!     | NumberLiteral
//!     | Ipv4
//!     | Ipv4Cidr
//!     | Ipv6
//!     | Ipv6Cidr
//!
//! Array -> "{" Expr* "}"
//!
//! Not -> "not" Expr
//!
//! BinOp<Op> -> Expr Op Expr
//!
//! Expr ->
//!     | Atom
//!     | "(" Expr ")"
//!     | Field
//!     | FunctionCall
//!     | Array
//!     | Not
//!     | BitwiseAnd = BinOp("bitwise_and")
//!     | BitwiseOr  = BinOp("bitwise_or")
//!     | Xor        = BinOp("xor")
//!     | Add        = BinOp("+")
//!     | Sub        = BinOp("-")
//!     | Mul        = BinOp("*")
//!     | Mod        = BinOp("%")
//!     | Div        = BinOp("/")
//!     | And        = BinOp("and")
//!     | Eq         = BinOp("==")
//!     | NEq        = BinOp("!=")
//!     | Gt         = BinOp(">")
//!     | Gte        = BinOp(">=")
//!     | Lt         = BinOp("<")
//!     | Lte        = BinOp("<=")
//!     | Contains   = BinOp("contains")
//!     | Matches    = BinOp("matches")
//!     | In         = BinOp("in")
//!
//! ```
//!

use std::ops::RangeInclusive;

use super::ast::{self, FunctionCall, Indexed, Span};
use super::tokenizer::{Token, Tokenizer, TokenizerError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error(transparent)]
    TokenizerError(#[from] TokenizerError),

    #[error("Expected a token but reached EOF")]
    NeedToken,

    #[error("Expected a closing paran but found {found:?}")]
    ExpectedClosingParan { opening: Span, found: Span },
    #[error("Expected a closing paran but found EOF")]
    ExpectedClosingParanButFoundEOF { opening: Span },
    #[error("Expected a closing brace but found EOF")]
    ExpectedClosingBraceButFoundEOF { opening: Span },

    #[error("Expected a comma between function parameters")]
    ExpectedACommaBetweenParams,

    #[error("Expected a closing bracket but found {found:?}")]
    ExpectedClosingBracket { opening: Span, found: Span },
    #[error("Expected a closing bracket but found EOF")]
    ExpectedClosingBracketButFoundEOF { opening: Span },

    #[error("Unexpected empty index")]
    UnexpectedEmptyIndex { opening: Span },

    #[error("this should not happen")]
    InternalError,

    #[error("Unexpected token {0:?}")]
    UnexpectedToken(Span),

    #[error("Expected EOF but found {0:?}")]
    ExpectedEOF(Vec<Span>),

    #[error("Expected an Ident token but found {0:?}")]
    ExpectedAnIdentFound(Span),

    #[error("Expected an Ident token but found EOF")]
    ExpectedAnIdentFoundEOF,
}

impl ParseError {
    pub fn explain(&self, input: &str) -> String {
        use ParseError::*;

        macro_rules! open_spans {
            ($($span:ident),*) => {
                {
                    #[allow(unused_mut)]
                    let mut r = format!("{:?}: ", &self);
                    $(
                        r.push_str(stringify!($span));
                        r.push_str(": `");
                        r.push_str(&input[$span.range.clone()]);
                        r.push('`');
                    )*
                    r

                }
            };
        }
        match self {
            TokenizerError(x) => open_spans!(x),
            NeedToken => open_spans!(),
            ExpectedClosingParan { opening, found } => open_spans!(opening, found),
            ExpectedClosingParanButFoundEOF { opening } => open_spans!(opening),
            ExpectedClosingBraceButFoundEOF { opening } => open_spans!(opening),
            ExpectedACommaBetweenParams => open_spans!(),
            ExpectedClosingBracket { opening, found } => open_spans!(opening, found),
            ExpectedClosingBracketButFoundEOF { opening } => open_spans!(opening),
            UnexpectedEmptyIndex { opening } => open_spans!(opening),
            InternalError => open_spans!(),
            UnexpectedToken(x) => open_spans!(x),
            ExpectedEOF(x) => {
                let start_idx = x.first().unwrap().range.start();
                let end_idx = x.last().unwrap().range.end();
                let range = RangeInclusive::new(*start_idx, *end_idx);
                format!("Expected EOF found {}", &input[range])
            }
            ExpectedAnIdentFound(x) => open_spans!(x),
            ExpectedAnIdentFoundEOF => open_spans!(),
        }
    }
}

type Result<T> = std::result::Result<T, ParseError>;

type State = std::collections::LinkedList<Span>;

fn parse_atom(state: &mut State) -> Result<ast::Atom> {
    let span = state.pop_front().ok_or(ParseError::NeedToken)?;

    let atom = match span.kind {
        Token::IPv4 => ast::Atom::Ipv4(span),
        Token::Ipv6 => ast::Atom::Ipv6(span),
        Token::Ipv4Cidr => ast::Atom::Ipv4Cidr(span),
        Token::Ipv6Cidr => ast::Atom::Ipv6Cidr(span),
        Token::String => ast::Atom::StringLiteral(span),
        Token::Number => ast::Atom::NumberLiteral(span),

        _ => return Err(ParseError::UnexpectedToken(span)),
    };

    Ok(atom)
}

impl Token {
    fn precedence(&self) -> Option<usize> {
        let x = match self {
            Token::OpContains => 6,
            Token::OpMatches => 6,

            Token::OpIn => 6,
            Token::OpMod => 4,
            Token::OpMul => 3,
            Token::OpDiv => 2,
            Token::OpAdd => 1,
            Token::OpSub => 0,

            Token::OpBitwiseAnd => 0,
            Token::OpBitwiseOr => 0,
            Token::OpXor => 0,
            Token::OpAnd => 0,
            Token::OpOr => 0,
            Token::OpEq => 0,
            Token::OpNEq => 0,
            Token::OpGtEq => 0,
            Token::OpLtEq => 0,
            Token::OpGt => 0,
            Token::OpLt => 0,
            _ => return None,
        };
        return Some(x);
    }
}

impl Span {
    fn is_operator(&self) -> bool {
        matches!(
            self.kind,
            Token::OpEq
                | Token::OpNEq
                | Token::OpGtEq
                | Token::OpLtEq
                | Token::OpGt
                | Token::OpLt
                | Token::OpContains
                | Token::OpMatches
                | Token::OpOr
                | Token::OpAnd
                | Token::OpBitwiseAnd
                | Token::OpBitwiseOr
                | Token::OpXor
                | Token::OpIn
                | Token::OpAdd
                | Token::OpSub
                | Token::OpMul
                | Token::OpDiv
                | Token::OpMod
        )
    }

    fn has_precedence(&self, other: &Span) -> Result<bool> {
        Ok(self.kind.precedence().ok_or(ParseError::InternalError)?
            > other.kind.precedence().ok_or(ParseError::InternalError)?)
    }

    fn to_expr(&self, lhs: ast::Expr, rhs: ast::Expr) -> Result<ast::Expr> {
        macro_rules! op {
            ($name: ident) => {
                ast::ExprKind::$name(Box::new(lhs), Box::new(rhs))
            };
        }
        let x = match self.kind {
            Token::OpEq => op!(Eq),
            Token::OpNEq => op!(NEq),
            Token::OpGtEq => op!(Gte),
            Token::OpLtEq => op!(Lte),
            Token::OpGt => op!(Gt),
            Token::OpLt => op!(Lt),
            Token::OpContains => op!(Contains),
            Token::OpMatches => op!(Matches),
            Token::OpOr => op!(Or),
            Token::OpAnd => op!(And),
            Token::OpBitwiseAnd => op!(BitwiseAnd),
            Token::OpBitwiseOr => op!(BitwiseOr),
            Token::OpXor => op!(Xor),
            Token::OpIn => op!(In),
            Token::OpAdd => op!(Add),
            Token::OpSub => op!(Sub),
            Token::OpMul => op!(Mul),
            Token::OpDiv => op!(Div),
            Token::OpMod => op!(Mod),
            _ => return Err(ParseError::InternalError),
        };
        Ok(x.into())
    }
}

fn parse_function_call(state: &mut State, chain: Vec<Span>) -> Result<ast::Expr> {
    let opening = state.pop_front().unwrap();
    let mut params = vec![];

    loop {
        let next = {
            let next = state.front();
            if next.is_none() {
                return Err(ParseError::ExpectedClosingParanButFoundEOF { opening });
            }
            next.unwrap()
        };

        if matches!(next.kind, Token::ParClose) {
            state.pop_front();
            break;
        }

        // if it's not the first param, we should expect a comma
        if !matches!(next.kind, Token::Comma) && !params.is_empty() {
            return Err(ParseError::ExpectedACommaBetweenParams);
        }

        if !params.is_empty() {
            state.pop_front();
        }

        let expr = parse_expr(state)?;
        params.push(expr);
    }

    Ok(ast::ExprKind::FunctionCall(FunctionCall {
        name: chain,
        args: params,
    })
    .into())
}

fn parse_field_or_function_call(state: &mut State) -> Result<ast::Expr> {
    let mut chain = vec![state.pop_front().ok_or(ParseError::NeedToken)?];

    let mut last_was_dot = false;
    // chain part
    while let Some(next) = state.front().clone() {
        match next.kind {
            Token::ParOpen => return parse_function_call(state, chain),

            Token::Dot if !last_was_dot => {
                last_was_dot = true;
                chain.push(state.pop_front().unwrap());
            }
            Token::Ident if last_was_dot => {
                last_was_dot = false;
                chain.push(state.pop_front().unwrap());
            }
            _ => break,
        }
    }

    Ok(ast::ExprKind::Field(ast::Field { chain }).into())
}

fn parse_dynamic_field(state: &mut State) -> Result<ast::Expr> {
    state.pop_front().ok_or(ParseError::NeedToken)?; // $

    if let Some(next) = state.pop_front() {
        if matches!(next.kind, Token::Ident) {
            // TODO: support $a.b
            return Ok(ast::ExprKind::DynamicField(next).into());
        } else {
            return Err(ParseError::ExpectedAnIdentFound(next));
        }
    } else {
        return Err(ParseError::ExpectedAnIdentFoundEOF);
    }
}

fn parse_shortest_expr(state: &mut State) -> Result<ast::Expr> {
    let first = state.front().ok_or(ParseError::NeedToken)?.kind.clone();

    let expr = match first {
        Token::ParOpen => {
            let opening_paran = state.pop_front().unwrap(); // pop the "(" token
            let expr = parse_expr(state)?;
            let next = state.pop_front();
            if let Some(Span {
                kind: Token::ParClose,
                range: _,
            }) = next
            {
                expr
            } else if let Some(found) = next {
                return Err(ParseError::ExpectedClosingParan {
                    opening: opening_paran,
                    found,
                });
            } else {
                return Err(ParseError::ExpectedClosingParanButFoundEOF {
                    opening: opening_paran,
                });
            }
        }
        Token::Not => {
            state.pop_front().unwrap(); // pop the "not" token
            let expr = parse_expr(state)?;
            ast::ExprKind::Not(Box::new(expr)).into()
        }
        Token::BraceOpen => {
            let opening = state.pop_front().unwrap(); // pop the "}" token
            let mut exprs = vec![];
            let mut is_closed = false;
            loop {
                if let Some(span) = state.front() {
                    if matches!(span.kind, Token::BraceClose) {
                        is_closed = true;
                        break;
                    }
                } else {
                    break;
                }
                let expr = parse_shortest_expr(state)?;
                exprs.push(expr);
            }
            if !is_closed {
                return Err(ParseError::ExpectedClosingBraceButFoundEOF { opening });
            }
            let closing = state.pop_front().unwrap();
            ast::ExprKind::Array {
                elements: exprs,
                start: opening,
                end: closing,
            }
            .into()
        }
        Token::Ident => parse_field_or_function_call(state)?,
        Token::Dollar => parse_dynamic_field(state)?,
        _ => {
            let atom = parse_atom(state)?;
            ast::ExprKind::Atom(atom).into()
        }
    };

    if matches!(state.front().map(|x| x.kind), Some(Token::BracketOpen)) {
        let opening = state.pop_front().unwrap();
        let next = state.front();
        let index = if let Some(Span {
            kind: Token::OpMul,
            range: _,
        }) = next
        {
            let _ = state.pop_front();
            ast::Index::Star
        } else if next.is_some() {
            let inner_expr = parse_expr(state)?;
            ast::Index::Expr(Box::new(inner_expr))
        } else {
            return Err(ParseError::UnexpectedEmptyIndex { opening });
        };
        state
            .pop_front()
            .filter(|x| matches!(x.kind, Token::BracketClose))
            .ok_or(ParseError::ExpectedClosingBracket {
                opening: opening.clone(),
                // i was lazy here. TODO: fix this
                found: opening,
            })?;
        return Ok(ast::ExprKind::Indexed(Indexed {
            expr: Box::new(expr),
            kind: index,
        })
        .into());
    }

    Ok(expr)
}

fn parse_binary_op(state: &mut State, lhs_expr: ast::Expr) -> Result<ast::Expr> {
    if let Some(next) = state.front() {
        if next.is_operator() {
            let next = state.pop_front().unwrap();
            let op = next;
            let rhs_expr = parse_shortest_expr(state)?;
            // order of precedence
            let next_op_has_precedence = state
                .front()
                .filter(|x| x.is_operator())
                .map(|x| x.has_precedence(&op).unwrap_or(false))
                .unwrap_or(false);

            let result = if next_op_has_precedence {
                op.to_expr(lhs_expr, parse_binary_op(state, rhs_expr)?)
            } else {
                op.to_expr(lhs_expr, rhs_expr)
            }?;
            parse_binary_op(state, result)
        } else {
            Ok(lhs_expr)
        }
    } else {
        Ok(lhs_expr)
    }
}

fn parse_expr(state: &mut State) -> Result<ast::Expr> {
    let lhs_expr = parse_shortest_expr(state)?;

    parse_binary_op(state, lhs_expr)
}

pub fn parse(expr: &str) -> Result<ast::Expr> {
    let mut spans = State::new();
    for token in Tokenizer::new(expr) {
        let span = token?;
        spans.push_back(span);
    }

    let expr = parse_expr(&mut spans)?;
    if spans.is_empty() {
        Ok(expr)
    } else {
        Err(ParseError::ExpectedEOF(spans.into_iter().collect()))
    }
}

#[cfg(test)]
pub mod tests {
    use std::ops::RangeInclusive;

    use crate::typecheck::typecheck::Type;

    use super::*;

    pub fn ast_to_text_verbose(ast: &ast::Expr, input: &str) -> String {
        use ast::ExprKind::*;

        macro_rules! binary {
            ($lhs:ident, $op:literal, $rhs:ident) => {{
                let lhs = ast_to_text_verbose(&$lhs, input);
                let rhs = ast_to_text_verbose(&$rhs, input);
                format!("{} {} {}", lhs, $op, rhs)
            }};
        }
        let r = match &ast.inner {
            Atom(
                ast::Atom::Ipv4(x)
                | ast::Atom::Ipv4Cidr(x)
                | ast::Atom::Ipv6(x)
                | ast::Atom::Ipv6Cidr(x)
                | ast::Atom::NumberLiteral(x)
                | ast::Atom::StringLiteral(x),
            ) => input[x.range.clone()].to_string(),
            Field(x) => {
                let start_idx = x.chain.first().unwrap().range.start();
                let end_idx = x.chain.last().unwrap().range.end();
                let name = &input[RangeInclusive::new(*start_idx, *end_idx)];
                format!("{}", name)
            }
            FunctionCall(x) => {
                let start_idx = x.name.first().unwrap().range.start();
                let end_idx = x.name.last().unwrap().range.end();
                let name = &input[RangeInclusive::new(*start_idx, *end_idx)];
                let elements = x
                    .args
                    .iter()
                    .map(|x| ast_to_text_verbose(x, input))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}({})", name, elements)
            }
            Array {
                elements,
                start: _,
                end: _,
            } => {
                let elements = elements
                    .iter()
                    .map(|x| ast_to_text_verbose(x, input))
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("{{ {} }}", elements)
            }
            Not(x) => {
                format!("not {}", ast_to_text_verbose(&x, input))
            }
            DynamicField(x) => format!("${}", &input[x.range.clone()]),
            BitwiseAnd(lhs, rhs) => binary!(lhs, "&", rhs),
            BitwiseOr(lhs, rhs) => binary!(lhs, "|", rhs),
            Xor(lhs, rhs) => binary!(lhs, "^", rhs),
            Add(lhs, rhs) => binary!(lhs, "+", rhs),
            Sub(lhs, rhs) => binary!(lhs, "-", rhs),
            Mul(lhs, rhs) => binary!(lhs, "*", rhs),
            Mod(lhs, rhs) => binary!(lhs, "%", rhs),
            Div(lhs, rhs) => binary!(lhs, "/", rhs),
            Or(lhs, rhs) => binary!(lhs, "||", rhs),
            And(lhs, rhs) => binary!(lhs, "&&", rhs),
            Eq(lhs, rhs) => binary!(lhs, "==", rhs),
            NEq(lhs, rhs) => binary!(lhs, "!=", rhs),
            Gt(lhs, rhs) => binary!(lhs, ">", rhs),
            Gte(lhs, rhs) => binary!(lhs, ">=", rhs),
            Lt(lhs, rhs) => binary!(lhs, "<", rhs),
            Lte(lhs, rhs) => binary!(lhs, "<=", rhs),
            Contains(lhs, rhs) => binary!(lhs, "contains", rhs),
            Matches(lhs, rhs) => binary!(lhs, "matches", rhs),
            In(lhs, rhs) => binary!(lhs, "in", rhs),
            Indexed(x) => {
                let rest = match &x.kind {
                    ast::Index::Star => "[*]".to_string(),
                    ast::Index::Expr(x) => format!("[{}]", ast_to_text_verbose(&x, input)),
                };
                format!("{}{}", ast_to_text_verbose(&x.expr, input), rest)
            }
        };

        if let Type::Unknown = ast.r#type {
            format!("({})", r)
        } else {
            format!("(<{}: {:?}>)", r, ast.r#type)
        }
    }

    fn test_suit(name: &str) {
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
            let expr = match parse(input) {
                Ok(x) => x,
                Err(err) => {
                    panic!("\ninput: {}\n{}\n", input, err.explain(input))
                }
            };
            let o = ast_to_text_verbose(&expr, input);
            println!("{}", &o);
            assert_eq!(o, expected_output);
            // TODO: parse the expected output, it should be same as before:
            // assert_eq!(
            //     ast_to_text_verbose(&parse(expected_output).unwrap(), expected_output),
            //     expected_output,
            // );
        }
    }

    #[test]
    fn test_cloudflare_samples() {
        test_suit("./src/parser/parser/cloudflare_docs_sample.test")
    }

    #[test]
    fn test_precedence_samples() {
        test_suit("./src/parser/parser/precedence.test")
    }
}
