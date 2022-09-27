use super::typecheck::Type;

#[derive(Debug, Default)]
pub struct Schema {
    pub fields: Vec<FieldDef>,
    pub functions: Vec<FunctionDef>,
}

impl Schema {
    pub fn find_field(&self, name: &str) -> Option<&FieldDef> {
        self.fields.iter().find(|x| x.name == name)
    }

    pub fn find_function(&self, name: &str, pars: &[Type]) -> Option<&FunctionDef> {
        self.functions.iter().find(|x| {
            x.name == name && x.par_types.len() == pars.len() && x.par_types.as_slice() == pars
        })
    }
}

#[derive(Debug, Default)]
pub struct FieldDef {
    pub name: String,
    pub r#type: Type,
}

#[derive(Debug, Default)]
pub struct FunctionDef {
    pub name: String,
    pub r_type: Type,
    pub par_types: Vec<Type>,
}

#[macro_export]
macro_rules! schema {
    (
        functions: [ $(fn $name:ident($($par:ident),*) -> $r_type:ident),* ],
        fields: [$($($field_name:ident).* : $field_type:expr),*]
    ) => {{
        use $crate::typecheck::typecheck::Type::*;
        #[allow(unused_mut)]
        let mut schema = $crate::typecheck::schema::Schema::default();
        $(
            schema.functions.push(FunctionDef {
                name: stringify!($name).to_string(),
                r_type: crate::typecheck::typecheck::Type::$r_type,
                par_types: vec![$(crate::typecheck::typecheck::Type::$par),*]
            });
        )*
        
        $(
            schema.fields.push($crate::typecheck::schema::FieldDef {
                name: stringify!($($field_name).*).replace(" .", "."),
                r#type: $field_type
            });
        )*
        schema
    }};
}

// fn example() {
//     let a = schema! {
//         functions: [
//             fn a(String, String) -> String
//         ],
//         fields: [
//             a.b.c: String
//         ]
//     };
// }
