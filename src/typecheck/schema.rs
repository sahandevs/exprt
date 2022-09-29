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
        functions: [ $(fn $name:ident($($par:expr),*) -> $r_type:expr ,)* ],
        fields: [$($($field_name:ident).* : $field_type:expr ,)*]
    ) => {{
        use $crate::typecheck::typecheck::Type::*;

        #[allow(non_snake_case)]
        #[allow(dead_code)]
        fn Array(t: $crate::typecheck::typecheck::Type) -> $crate::typecheck::typecheck::Type {
            return $crate::typecheck::typecheck::Type::Array(Box::new(t));
        }

        #[allow(non_snake_case)]
        #[allow(dead_code)]
        fn Iterator(t: $crate::typecheck::typecheck::Type) -> $crate::typecheck::typecheck::Type {
            return $crate::typecheck::typecheck::Type::Iterator(Box::new(t));
        }

        #[allow(non_snake_case)]
        #[allow(dead_code)]
        fn Option(t: $crate::typecheck::typecheck::Type) -> $crate::typecheck::typecheck::Type {
            return $crate::typecheck::typecheck::Type::Option(Box::new(t));
        }

        #[allow(non_snake_case)]
        #[allow(dead_code)]
        fn T(n: usize) -> $crate::typecheck::typecheck::Type {
            return $crate::typecheck::typecheck::Type::Generic(n);
        }

        #[allow(non_snake_case)]
        #[allow(dead_code)]
        fn Map(key: $crate::typecheck::typecheck::Type, val: $crate::typecheck::typecheck::Type) -> $crate::typecheck::typecheck::Type {
            return $crate::typecheck::typecheck::Type::Map(Box::new(key), Box::new(val));
        }

        #[allow(unused_mut)]
        let mut schema = $crate::typecheck::schema::Schema::default();
        $(
            schema.functions.push($crate::typecheck::schema::FunctionDef {
                name: stringify!($name).to_string(),
                r_type: $r_type,
                par_types: vec![$($par),*]
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
