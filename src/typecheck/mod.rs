use crate::parser::ast;

use self::{schema::Schema, typecheck::TypeCheckError};
pub mod schema;
pub mod typecheck;
pub fn type_check(
    input: &str,
    expr: &mut ast::Expr,
    schema: &Schema,
) -> Result<(), TypeCheckError> {
    typecheck::infer_and_typecheck(input, expr, schema)?;
    Ok(())
}
