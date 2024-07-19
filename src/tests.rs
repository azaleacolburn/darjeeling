use core::panic;
use std::{
    fs,
    io::{BufRead, BufReader},
};

use crate::{
    activation::ActivationFunction,
    categorize::CatNetwork,
    generation::GenNetwork,
    input::Input,
    // dataframe::DataFrame,
    // series::Series,
    types::Types,
    DEBUG,
};

// #[test]
// pub fn bench() {
//     bench!(CatNetwork::new(4, 4, 4, 1, ActivationFunction::Sigmoid));
//     let mut net = CatNetwork::new(2, 2, 2, 1, ActivationFunction::Sigmoid);
//     let _ = net.learn(&mut xor_file(), vec![types::Types::Float(0.0), types::Types::Float(1.0)], 0.5, "bench", 0.99, true);

// }

#[test]
pub fn train_test_xor() {
    let learning_rate = 0.1;
    let categories: Vec<Types> = categories_float_format(vec![1.0, 0.0]);
    let mut data: Vec<Input> = xor_file();

    // Panics if unwraps a None value
    let model_name: String =
        train_network_xor(data.clone(), categories.clone(), learning_rate).unwrap();
    data.iter_mut().for_each(|input| {
        input.answer = None;
    });
    CatNetwork::test(data, categories, model_name).unwrap();
}

/// Returns None if the learn function returns an Err variant
fn train_network_xor(
    mut data: Vec<Input>,
    categories: Vec<Types>,
    learning_rate: f32,
) -> Option<String> {
    let input_num = 2;
    let hidden_num = 2;
    let answer_num = 2;
    let hidden_layers = 1;
    let mut net = CatNetwork::new(
        input_num,
        hidden_num,
        answer_num,
        hidden_layers,
        ActivationFunction::Sigmoid,
    );

    match net.learn(&mut data, categories, learning_rate, "xor", 99.0, true) {
        Ok((model_name, _err_percent, _mse)) => Some(model_name.expect("write is true")),
        Err(_err) => None,
    }
}

/// Read the file you want to and format it as Inputs
pub fn xor_file<'a>() -> Vec<Input> {
    let file = match fs::File::open("training_data/xor.txt") {
        Ok(file) => file,
        Err(error) => panic!("Panic opening the file: {:?}", error),
    };

    let reader = BufReader::new(file);
    let mut inputs: Vec<Input> = vec![];

    for l in reader.lines() {
        let line: String = match l {
            Ok(line) => line,
            Err(error) => panic!("{:?}", error),
        };

        let init_inputs: Vec<&str> = line.split(";").collect();
        // Confused about how this is supposed to work
        let float_inputs: Vec<f32> = vec![
            init_inputs[0].split(" ").collect::<Vec<&str>>()[0]
                .parse()
                .unwrap(),
            init_inputs[0].split(" ").collect::<Vec<&str>>()[1]
                .parse()
                .unwrap(),
        ];
        let answer_inputs: Types =
            Types::Float((init_inputs[init_inputs.len() - 1]).parse().unwrap()); // TODO: Figure out what should be the row's answer; the last of a line for
        let input: Input = Input::new(float_inputs, Some(answer_inputs));
        inputs.push(input);
    }
    inputs
}

// TODO: Wite predictive model training first
// #[test]
// fn train_test_reverse() {
//     let learning_rate:f32 = 1.0;
//     let categories = categories_format()
// }

#[test]
fn train_test_digits() {
    let learning_rate: f32 = 0.05;
    let categories: Vec<Types> =
        categories_str_format(vec!["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]);
    let data: Vec<Input> = digits_file();

    let model_name: String =
        train_network_digits(data.clone(), categories.clone(), learning_rate).unwrap();

    CatNetwork::test(data, categories, model_name).unwrap();
}

/// # Panics
/// If the learn function returns an Err
fn train_network_digits(
    mut data: Vec<Input>,
    categories: Vec<Types>,
    learning_rate: f32,
) -> Option<String> {
    let mut net = CatNetwork::new(64, 128, 10, 1, ActivationFunction::Sigmoid);

    match net.learn(&mut data, categories, learning_rate, "digits", 99.0, true) {
        Ok((model_name, _err_percent, _mse)) => Some(model_name.expect("write is true")),

        Err(error) => panic!("{:?}", error),
    }
}

fn digits_file() -> Vec<Input> {
    let file = match fs::File::open("training_data/train-digits.txt") {
        Ok(file) => file,
        Err(error) => panic!("Panic opening the file: {:?}", error),
    };

    let reader = BufReader::new(file);
    let mut inputs: Vec<Input> = vec![];

    for l in reader.lines() {
        let line = match l {
            Ok(line) => line,
            Err(error) => panic!("{:?}", error),
        };

        let init_inputs: Vec<&str> = line.split(",").collect();
        let mut float_inputs: Vec<f32> = vec![];

        for i in 0..init_inputs.len() - 1 {
            float_inputs.push(init_inputs[i].parse().unwrap());
        }
        let input: Input = Input::new(
            float_inputs,
            Some(Types::String(
                init_inputs[init_inputs.len() - 1].to_string(),
            )),
        );
        if DEBUG {
            println!(
                "Correct Answer: {:?}",
                init_inputs[init_inputs.len() - 1].to_string()
            );
        }
        inputs.push(input);
    }

    inputs
}

#[test]
fn train_test_gen() {
    let model_name = train_gen();
    let data = gen_data_file();
    let data1 = data[data.len() - 1].clone();
    let mut model: GenNetwork = GenNetwork::read_model(model_name).unwrap();
    let output: Vec<Input> = model.test(&mut vec![data1]).unwrap();
    for i in 0..output.len() {
        println!("{}", output[i]);
    }
}

fn train_gen() -> String {
    let mut inputs = gen_data_file();
    let mut net = GenNetwork::new(8, 8, 8, 1, ActivationFunction::Sigmoid);
    net.learn(
        &mut inputs,
        1.0,
        "dummy_gen",
        100,
        0.5,
        8,
        1,
        ActivationFunction::Sigmoid,
        99.0,
    )
    .unwrap()
}

/// Read the file you want to and format it as Inputs
pub fn gen_data_file<'a>() -> Vec<Input> {
    let file = match fs::File::open("training_data/train-digits.txt") {
        Ok(file) => file,
        Err(error) => panic!("Panic opening the file: {:?}", error),
    };

    let reader = BufReader::new(file);
    let mut inputs: Vec<Input> = vec![];

    for l in reader.lines() {
        let line: String = match l {
            Ok(line) => line,
            Err(error) => panic!("{:?}", error),
        };

        let init_inputs: Vec<&str> = line.split(",").collect();
        let mut float_inputs: Vec<f32> = vec![];
        for i in 0..init_inputs.len() {
            float_inputs.push(init_inputs[i].parse().unwrap());
        }
        let input = Input::new(float_inputs, None);
        inputs.push(input);
    }
    inputs
}

#[test]
pub fn test_add_hidden_layers() {
    let mut net = CatNetwork::new(2, 2, 2, 1, ActivationFunction::Linear);
    println!("init network");
    net.add_hidden_layer_with_size(2);
    net.set_activation_func(ActivationFunction::Sigmoid);
    // println!("added layer");
    // let _ = net.learn(&mut xor_file(), vec![types::Types::Integer(1), types::Types::Integer(0)], 0.5, "test", 99.0, true);
}

/// Formats cateories from a vector of string slices to a vector of strings
/// # Params
/// - Categories Strings: A list of string literals, one for each answer option(category)
pub fn categories_str_format(categories_str: Vec<&str>) -> Vec<Types> {
    let mut categories: Vec<Types> = vec![];
    for category in categories_str {
        categories.push(Types::String(category.to_string()));
    }

    categories
}

pub fn categories_float_format(categories_str: Vec<f32>) -> Vec<Types> {
    let mut categories: Vec<Types> = vec![];
    for category in categories_str {
        categories.push(Types::Float(category));
    }

    categories
}

// #[test]
// fn dataframe_add_sub<'a>() {
//     let mut frame: DataFrame<'a> = quick_frame();
//     frame.display();
//     frame.add_row(
//         "Label!",
//         types::fmt_int_type_vec(vec![10, 11, 12]),
//     ).unwrap();
//     frame.display();
//     println!("{:?}\n", frame.index_at_labels("Label!", "col1"));
//     frame.delete_row("Label!");
//     let _ = frame.add_col(
//         "col3",
//         types::fmt_str_type_vec(vec!["hellow", "fun", "life"])
//     );
//     frame.display();
//     frame.delete_col("col3");
//     frame.display();
// }

// fn quick_frame() -> DataFrame<'static> {
//     DataFrame::from_2d_array(
//         vec![
//             // Each of inner vector is a row, each index is a column
//             types::fmt_int_type_vec(vec![0, 1, 2]),
//             types::fmt_str_type_vec(vec!["hello", "hi","heya"])
//         ],
//         vec!["row1", "row2"],
//         vec!["col1", "col2", "col3"]
//     )
// }

// #[test]
// fn series_add_sub<'a>() {
//     let mut series: Series<i32> = quick_series();
//     series.display();
//     series.mut_add(Types::String("data4".to_string()), 4);
//     series.display();
//     let mut new_series: Series<i32> = series.no_mut_add(Types::String("data5".to_string()), 5);
//     new_series.display();
//     let _ = new_series.mut_sub(5);
//     new_series.display();
//     new_series.mut_add(Types::Integer(83), 6);
//     new_series.display();
//     new_series.get(6).unwrap().display();
//     print!("\n");
// }

// fn quick_series() -> Series<i32> {

//     Series::new(
//         types::fmt_str_type_vec(vec!["data1", "data2", "data3"]),
//         vec![1,2,3]
//     )
// }
