use std::ops::RangeInclusive;

use crate::parser::ast::{self, Expr};

impl Expr {
    /// converts [`crate::parser::ast::Expr`] to text

    pub fn to_text(&self, input: &str) -> String {
        use ast::ExprKind::*;

        macro_rules! binary {
            ($lhs:ident, $op:literal, $rhs:ident) => {{
                let lhs = &$lhs.to_text(input);
                let rhs = &$rhs.to_text(input);
                format!("{} {} {}", lhs, $op, rhs)
            }};
        }
        let r = match &self.inner {
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

            DynamicField(x) => {
                let start_idx = x.chain.first().unwrap().range.start();
                let end_idx = x.chain.last().unwrap().range.end();
                let name = &input[RangeInclusive::new(*start_idx, *end_idx)];
                format!("${}", name)
            }
            FunctionCall(x) => {
                let start_idx = x.name.first().unwrap().range.start();
                let end_idx = x.name.last().unwrap().range.end();
                let name = &input[RangeInclusive::new(*start_idx, *end_idx)];
                let elements = x
                    .args
                    .iter()
                    .map(|x| x.to_text(input))
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
                    .map(|x| x.to_text(input))
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("{{ {} }}", elements)
            }
            Not(x) => {
                format!("not {}", x.to_text(input))
            }
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
                    ast::Index::Expr(x) => format!("[{}]", x.to_text(input)),
                };
                format!("{}{}", x.expr.to_text(input), rest)
            }
        };

        format!("{}", r)
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::parser::parse;

    fn test_suit(name: &str) {
        let tests = std::fs::read_to_string(name).unwrap();
        let tests: Vec<_> = tests
            .split("###############################")
            .map(|x| x.trim())
            .collect();
        for sample in tests.iter() {
            let (input, _) = sample
                .split_once("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                .unwrap();
            let expr = match parse(input) {
                Ok(x) => x,
                Err(err) => {
                    panic!("\ninput: {}\n{}\n", input, err.explain(input))
                }
            };
            let o1 = expr.to_text(input);

            let o2 = match parse(&o1) {
                Ok(x) => x.to_text(&o1),
                Err(err) => {
                    panic!("\ninput: {}\n{}\n", o1, err.explain(&o1))
                }
            };

            assert_eq!(o1.trim(), o2.trim());
        }
    }

    #[test]
    fn test_cloudflare_samples() {
        test_suit("./src/parser/parser/cloudflare_docs_sample.test")
    }
}
