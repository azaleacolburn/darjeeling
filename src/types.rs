use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Types {
    String(String),
    Integer(i32),
    Float(f32),
    Boolean(bool),
}

impl Types {
    
    pub fn display(&self) {
        match self {  
            Types::String(string) => print!("{}", string),
            Types::Integer(int) => print!("{}", int),
            Types::Float(float) => print!("{}", float),
            Types::Boolean(boolean) => print!("{}", boolean),
        };

    }
}

impl Default for Types {
    fn default() -> Self {
        Types::Integer(Default::default())
    }
}

// Attempts to create generic type enum vector conversion
// pub fn fmt_type_vec<T: 'static>(list: Vec<T>) -> Result<Vec<Types<'static>>, DarjeelingError<'static>> 
// where
//     &'static str: From<T>, i32: From<T>, f32: From<T>, bool: From<T>, T: Clone, T: Default {

//         let value: T = Default::default();
//         let s: TypeId = TypeId::of::<&str>();
//         let i: TypeId = TypeId::of::<i32>();
//         let f: TypeId = TypeId::of::<f32>();
//         let b: TypeId = TypeId::of::<bool>();
//         let mut fmted_list: Vec<Types> = vec![];
//         match value.type_id() {

//             s => {
//                 for i in 0..list.len() {
//                     fmted_list.push(Types::String(list[i].clone().try_into().unwrap()))
//                 }
//             },
//             i => {
//                 for i in 0..list.len() {
//                     fmted_list.push(Types::Integer(list[i].clone().try_into().unwrap()))
//                 }
//             },
//             f => {
//                 for i in 0..list.len() {
//                     fmted_list.push(Types::Float(list[i].clone().try_into().unwrap()))
//                 }
//             },
//             b => {
//                 for i in 0..list.len() {
//                     fmted_list.push(Types::Boolean(list[i].clone().try_into().unwrap()))
//                 }
//             }
//             _ => return Err(DarjeelingError::FormatTypeIsNotValid(value.type_id()))
//         };

//         Ok(fmted_list)
// }

// pub fn fmt_type_vec<T>(list: Vec<T>) -> Result<Vec<Types<'static>>, DarjeelingError<'static>> 
// where 
//     T: Default + Clone {
    
//     let t: TypeId = TypeId::of::<T>();
//     let s: TypeId = TypeId::of::<&str>();
//     let i: TypeId = TypeId::of::<i32>();
//     let f: TypeId = TypeId::of::<f32>();
//     let b: TypeId = TypeId::of::<bool>();
//     Ok(match t {

//         s => {
//             fmt_str_type_vec(list)
//         },
//         i => {
//             fmt_int_type_vec(list)
//         },
//         f => {
//             fmt_float_type_vec(list)
//         },
//         b => {
//             fmt_bool_type_vec(list)
//         }
//         _ => return Err(DarjeelingError::FormatTypeIsNotValid(t))
//     })
// }

pub fn fmt_str_type_vec(list: Vec<&'static str>) -> Vec<Types> {
    let mut fmted_list: Vec<Types> = vec![];
    for i in 0..list.len() {
        fmted_list.push(Types::String(list[i].clone().try_into().unwrap()))
    }

    fmted_list
}

pub fn fmt_int_type_vec(list: Vec<i32>) -> Vec<Types> {
    let mut fmted_list: Vec<Types> = vec![];
    for i in 0..list.len() {
        fmted_list.push(Types::Integer(list[i].clone().try_into().unwrap()))
    }

    fmted_list
}

pub fn fmt_float_type_vec(list: Vec<f32>) -> Vec<Types> {
    let mut fmted_list: Vec<Types> = vec![];
    for i in 0..list.len() {
        fmted_list.push(Types::Float(list[i].clone().try_into().unwrap()))
    }

    fmted_list
}

pub fn fmt_bool_type_vec(list: Vec<bool>) -> Vec<Types> {
    let mut fmted_list: Vec<Types> = vec![];
    for i in 0..list.len() {
        fmted_list.push(Types::Boolean(list[i].clone().try_into().unwrap()))
    }

    fmted_list
}