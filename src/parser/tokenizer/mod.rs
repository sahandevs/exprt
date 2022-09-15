//! Tokenizer for the language
//!
//! ### Why write a tokenizer by hand instead of using parser/lexer generators?
//!
//! TODO:
use std::{iter::Enumerate, ops::RangeInclusive, str::Chars};

use peekmore::{PeekMore, PeekMoreIterator};

/// An iterator for tokenizing an expression
/// TODO: add sample
pub struct Tokenizer<'i> {
    stream: PeekMoreIterator<Enumerate<Chars<'i>>>,
    in_error: bool,
}

/// Token kind
#[derive(Debug)]
pub enum Token {
    /// '==' | 'eq'
    OpEq,
    /// '!-' | 'ne'
    OpNEq,
    /// '>=' | 'ge'
    OpGtEq,
    /// '<=' | 'le'
    OpLtEq,
    /// '>' | 'gt'
    OpGt,
    /// '<=' | 'lt'
    OpLt,
    /// 'contains'
    OpContains,
    /// 'matches' | '~'
    OpMatches,
    /// 'or' | '||'
    OpOr,
    /// 'and' | '&&'
    OpAnd,
    /// '&' | 'bitwise_and'
    OpBitwiseAnd,
    /// '|' | 'bitwise_or'
    OpBitwiseOr,
    /// 'xor' | '^'
    OpXor,
    /// 'in'
    OpIn,
    /// '+'
    OpAdd,
    /// '-'
    OpSub,
    /// '*'
    OpMul,
    /// '/'
    OpDiv,
    /// '%'
    OpMod,
    /// '('
    ParOpen,
    /// ')'
    ParClose,
    /// '['
    BracketOpen,
    /// ']'
    BracketClose,
    /// '{'
    BraceOpen,
    /// '}'
    BraceClose,
    /// '1.2.3.4'
    IPv4,
    /// '2001:db8::8a2e:370:7334'
    Ipv6,
    /// '1.2.3.4/24'
    Ipv4Cidr,
    /// '2001:db8::8a2e:370:7334/12'
    Ipv6Cidr,
    /// '"a\nhi"'
    String,
    /// '1', '-1', '11.1'
    Number,
    /// 'not' | '!'
    Not,
    /// '*'
    Star,
    /// '.'
    Dot,
    /// '$'
    Dollar,
    /// ','
    Comma,
    /// identifier like field name
    Ident,
}

/// A section or view of the expression
pub struct Span {
    /// type of section
    pub kind: Token,
    /// character range that have generated this span
    pub range: RangeInclusive<usize>,
}

/// Reason of the tokenizer failure
#[derive(Debug)]
pub enum TokenizerErrorType {
    /// Unexpected character after '=' sign
    UnexpectedCharacterAfterEqualSign,
    /// Unexpected end of stream after '=' sign
    UnexpectedEOFAfterEqualSign,
    /// Unexpected character (like emoji as ident)
    UnexpectedCharacter,
    /// Unrecognized escape character in the string literal
    UnexpectedEscapeCharacter,
    /// Found opening '"' but not the closing '"'
    UnexpectedEndOfString,
    /// Found an incomplete ipv4 token.
    /// Examples:
    ///  - '1.2.'
    ///  - '1.2.3'
    IncompleteIpv4,
}

/// Reason of the tokenizer failure
pub struct TokenizerError {
    /// Actual reason
    pub kind: TokenizerErrorType,
    /// character range that have caused this error
    pub range: RangeInclusive<usize>,
}

type Item = Result<Span, TokenizerError>;

impl<'i> std::iter::Iterator for Tokenizer<'i> {
    type Item = Item;

    fn next(&mut self) -> Option<Self::Item> {
        // emit a span or generate an error
        macro_rules! emit_or_error {
            ($e:expr) => {
                match $e {
                    Ok(x) => return Some(Ok(x)),
                    Err(e) => {
                        self.in_error = true;
                        return Some(Err(e));
                    }
                }
            };
        }

        if self.in_error {
            return None;
        }
        self.skip_whitespace()?;

        while let Some((i, next_char)) = self.stream.peek().copied() {
            self.stream.reset_cursor();
            macro_rules! emit {
                ($token:ident) => {{
                    self.stream.next();
                    emit_or_error!(Ok(Span {
                        kind: Token::$token,
                        range: i..=i
                    }));
                }};
            }
            match next_char {
                '/' => {
                    if let Some((_, '*')) = self.stream.peek_next().copied() {
                        self.skip_comment()?;
                    } else {
                        emit!(OpDiv);
                    }
                }
                '!' | '&' | '|' | '>' | '<' => {
                    return self.parse_op();
                }
                '=' => {
                    let next = self.stream.peek_next().copied();
                    if let Some((_, '=')) = next {
                        self.stream.next(); // because it has two parts
                        emit!(OpEq);
                    }
                    if let Some((_, _)) = next {
                        emit_or_error!(Err(TokenizerError {
                            kind: TokenizerErrorType::UnexpectedCharacterAfterEqualSign,
                            range: i..=(i + 1)
                        }))
                    } else {
                        emit_or_error!(Err(TokenizerError {
                            kind: TokenizerErrorType::UnexpectedEOFAfterEqualSign,
                            range: i..=i
                        }))
                    }
                }
                '"' | '\'' => return self.parse_string(),
                '.' => emit!(Dot),
                '^' => emit!(OpXor),
                '(' => emit!(ParOpen),
                ')' => emit!(ParClose),
                '[' => emit!(BracketOpen),
                ']' => emit!(BracketClose),
                '{' => emit!(BraceOpen),
                '}' => emit!(BraceClose),
                '$' => emit!(Dollar),
                '*' => emit!(Star),
                '+' => emit!(OpAdd),
                '%' => emit!(OpMod),
                ':' => {
                    self.stream.next();
                    return self.parse_ipv6(i);
                }
                '-' => {
                    if self
                        .stream
                        .peek_next()
                        .map(|x| x.1)
                        .map(char::is_numeric)
                        .unwrap_or(false)
                    {
                        return self.parse_negative_number();
                    } else {
                        emit!(OpSub)
                    }
                }
                x if x.is_numeric() => return self.parse_number_or_ipv4(),
                x if x.is_alphabetic() => return self.parse_ident_or_keyword(),
                _ => {
                    emit_or_error!(Err(TokenizerError {
                        kind: TokenizerErrorType::UnexpectedCharacter,
                        range: i..=i
                    }))
                }
            }
        }

        None
    }
}

impl<'i> Tokenizer<'i> {
    /// create a tokenizer from [`&str`]
    pub fn new(expr: &'i str) -> Self {
        Self {
            stream: expr.chars().enumerate().peekmore(),
            in_error: false,
        }
    }

    fn parse_string(&mut self) -> Option<Item> {
        let (start, char) = self.stream.next()?; // starting '"'
        let mut end = start;

        let mut in_escape = false;
        loop {
            // we don't to peek here because string hat at-least two characters
            let next = self.stream.next();
            if next.is_none() {
                return Some(Err(TokenizerError {
                    kind: TokenizerErrorType::UnexpectedCharacter,
                    range: start..=end,
                }));
            }
            let (i, c) = next.unwrap();
            end = i;
            match c {
                x if x == char => {
                    if in_escape {
                        in_escape = false;
                    } else {
                        break;
                    }
                }
                '\\' if !in_escape => {
                    in_escape = true;
                }
                '\\' | '.' => {
                    in_escape = false;
                }
                _ if in_escape => {
                    return Some(Err(TokenizerError {
                        kind: TokenizerErrorType::UnexpectedEscapeCharacter,
                        range: start..=end,
                    }));
                }
                _ => {}
            };
        }

        Some(Ok(Span {
            kind: Token::String,
            range: start..=end,
        }))
    }

    fn parse_negative_number(&mut self) -> Option<Item> {
        let (start, _) = self.stream.next()?; // '-' sign
        let mut end = start;

        let mut seen_decimal = false; // if already seen a decimal part ('.' character)
        while let Some((i, c)) = self.stream.peek().copied() {
            match c {
                '.' if !seen_decimal => {
                    if let Some((_, x)) = self.stream.peek_next().copied() {
                        if x.is_numeric() {
                            self.stream.next(); // consume the dot
                            seen_decimal = true;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                x if x.is_numeric() => {
                    end = i;
                    self.stream.next();
                }
                _ => break,
            }
        }
        Some(Ok(Span {
            kind: Token::Number,
            range: start..=end,
        }))
    }

    // actual ipv4/cidr parsing happens in parse time
    fn parse_number_or_ipv4(&mut self) -> Option<Item> {
        /*
         Start: Number

         (only if peek_next is numeric)
         Number(1) => '.' => DecimalNumber(1.2)
         DecimalNumber => '.' => IncompleteIpv4(1.2.3)
         IncompleteIpv4 => '.' => Ipv4(1.2.3.4)
         Ipv4 => '/' => Ipv4Cidr

         x => numeric => x
         Number | DecimalNumber => _ => Done
         IncompleteIpv4 => _ => error

        */
        #[derive(Clone, Copy)]
        enum State {
            Number,
            DecimalNumber,
            IncompleteIpv4,
            Ipv4,
            Ipv4Cidr,
        }

        use State::*;

        let mut state = Number;
        let (start, _) = self.stream.next()?;
        let mut end = start;

        while let Some((i, c)) = self.stream.peek().copied() {
            macro_rules! consume {
                (to $state:ident) => {{
                    self.stream.next();
                    end = i;
                    state = $state;
                }};
                (to $state:ident if |$c:ident| { $cond:expr } else $else:expr ) => {{
                    if self
                        .stream
                        .peek_next()
                        .copied()
                        .map(|(_, $c)| $cond)
                        .unwrap_or(false)
                    {
                        self.stream.reset_cursor();
                        consume!(to $state);
                    } else {
                        $else;
                    }
                }};
            }
            match (state.clone(), c) {
                (Number, '.') => {
                    consume!(to DecimalNumber
                             if |next| { next.is_numeric() }
                             else return Some(Ok(Span {
                                 kind: Token::Number,
                                 range: start..=end,
                             })
                    ));
                }
                (DecimalNumber, '.') => {
                    consume!(to IncompleteIpv4
                        if |next| { next.is_numeric() }
                        else return Some(Ok(Span {
                            kind: Token::Number,
                            range: start..=end,
                        })
                    ));
                }
                (Ipv4, '/') => {
                    consume!(to Ipv4Cidr
                        if |next| { next.is_numeric() }
                        else return Some(Ok(Span {
                            kind: Token::IPv4,
                            range: start..=end,
                        })
                    ));
                }
                (IncompleteIpv4, '.') => {
                    consume!(to Ipv4
                        if |next| { next.is_numeric() }
                        else return Some(Err(TokenizerError {
                            kind: TokenizerErrorType::IncompleteIpv4,
                            range: start..=end,
                        })
                    ));
                }
                (Number | DecimalNumber | IncompleteIpv4 | Ipv4 | Ipv4Cidr, x)
                    if x.is_numeric() =>
                {
                    self.stream.next();
                    end = i;
                }
                (Ipv4, '.') => {
                    return Some(Ok(Span {
                        kind: Token::Ipv6,
                        range: start..=end,
                    }))
                }
                (Number, x) if x.is_ascii_hexdigit() => {
                    return self.parse_ipv6(start);
                }
                (Number | DecimalNumber, '/') => {
                    return Some(Ok(Span {
                        kind: Token::Number,
                        range: start..=end,
                    }))
                }
                _ => break,
            }
        }

        match state {
            Number | DecimalNumber => {
                return Some(Ok(Span {
                    kind: Token::Number,
                    range: start..=end,
                }))
            }
            Ipv4 => {
                return Some(Ok(Span {
                    kind: Token::IPv4,
                    range: start..=end,
                }))
            }
            Ipv4Cidr => {
                return Some(Ok(Span {
                    kind: Token::Ipv4Cidr,
                    range: start..=end,
                }))
            }
            IncompleteIpv4 => {
                return Some(Err(TokenizerError {
                    kind: TokenizerErrorType::IncompleteIpv4,
                    range: start..=end,
                }))
            }
        }
    }

    // actual ipv6/cidr parsing happens in parse time
    fn parse_ipv6(&mut self, start: usize) -> Option<Item> {
        // Few examples:
        // - '2001:0000:0dea:C1AB:0000:00D0:ABCD:004E'
        // - '2001:0:eab:DEAD:0:A0:ABCD:4E'
        // - '2001:0:0eab:dead:0:a0:abcd:4e'
        // - '2001:0:0eab:dead::a0:abcd:4e'
        // - '2001:eab::1/128'
        // - '2001:eab::/64'
        let mut end = start;
        let mut in_cidr = false;
        while let Some((i, c)) = self.stream.peek().copied() {
            match (c, in_cidr) {
                (x, false) if x.is_ascii_hexdigit() || x == ':' => {
                    self.stream.next();
                    end = i;
                }
                ('/', false) => {
                    in_cidr = true;
                    self.stream.next();
                    end = i;
                }
                (x, true) if x.is_numeric() => {
                    self.stream.next();
                    end = i;
                }
                _ => break,
            }
        }
        return Some(Ok(Span {
            kind: if in_cidr {
                Token::Ipv6Cidr
            } else {
                Token::Ipv6
            },
            range: start..=end,
        }));
    }

    fn parse_op(&mut self) -> Option<Item> {
        let (start, c) = self.stream.next()?;
        let mut end = start;
        let next = self.stream.peek().copied().map(|x| x.1);
        let kind = match (c, next) {
            ('!', Some('=')) => {
                end += 1;
                Token::OpNEq
            }
            ('|', Some('|')) => {
                end += 1;
                Token::OpOr
            }
            ('&', Some('&')) => {
                end += 1;
                Token::OpAnd
            }
            ('>', Some('=')) => {
                end += 1;
                Token::OpGtEq
            }
            ('<', Some('=')) => {
                end += 1;
                Token::OpLtEq
            }
            ('!', _) => Token::Not,
            ('|', _) => Token::OpBitwiseOr,
            ('&', _) => Token::OpBitwiseAnd,
            ('>', _) => Token::OpGt,
            ('<', _) => Token::OpLt,
            _ => return None,
        };
        Some(Ok(Span {
            kind,
            range: start..=end,
        }))
    }

    fn parse_ident_or_keyword(&mut self) -> Option<Item> {
        let mut ident = String::new();
        let (start, c) = self.stream.next()?; // first character
        ident.push(c);
        let mut end = start;
        while let Some((i, c)) = self.stream.peek().copied() {
            match c {
                _ if c.is_alphanumeric() || c == '_' => {
                    self.stream.next();
                    end = i;
                    ident.push(c);
                }
                ':' => return self.parse_ipv6(start),
                _ => break,
            }
        }

        let kind = match ident.as_str() {
            "matches" => Token::OpMatches,
            "contains" => Token::OpContains,
            "or" => Token::OpOr,
            "and" => Token::OpAnd,
            "xor" => Token::OpXor,
            "bitwise_and" => Token::OpBitwiseAnd,
            "bitwise_or" => Token::OpBitwiseOr,
            "in" => Token::OpIn,
            "not" => Token::Not,
            "eq" => Token::OpEq,
            "ne" => Token::OpNEq,
            "ge" => Token::OpGtEq,
            "le" => Token::OpLtEq,
            "gt" => Token::OpGt,
            "lt" => Token::OpLt,
            _ => Token::Ident,
        };

        Some(Ok(Span {
            kind: kind,
            range: start..=end,
        }))
    }

    fn skip_comment(&mut self) -> Option<()> {
        // consume `/` and `*`
        self.stream.next();
        self.stream.next();

        while let Some((_, c)) = self.stream.next() {
            if c == '*' && matches!(self.stream.peek(), Some((_, '/'))) {
                self.stream.next();
                return Some(());
            }
        }

        None // EOF
    }

    fn skip_whitespace(&mut self) -> Option<()> {
        loop {
            match self.stream.peek() {
                Some((_, x)) if x.is_whitespace() => {
                    self.stream.next();
                }
                None => return None,
                _ => return Some(()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn explain_item(item: Item, input: &str) -> String {
        match item {
            Ok(x) => {
                format!("`{}`:{:?}", &input[x.range], x.kind)
            }
            Err(e) => {
                format!("`{}`:{:?}", &input[e.range], e.kind)
            }
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
            let expected_output: Vec<_> = expected_output.trim().split('\n').collect();
            for (idx, token) in Tokenizer::new(input).enumerate() {
                // println!("{}", explain_item(token, input));
                assert_eq!(explain_item(token, input), expected_output[idx]);
            }
        }
    }

    #[test]
    fn test_cloudflare_samples() {
        test_suit("./src/parser/tokenizer/cloudflare_docs_sample.test")
    }
}
