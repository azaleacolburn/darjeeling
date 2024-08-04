use core::panic;
use std::{
    fs,
    io::{BufRead, BufReader},
};

use crate::{
    activation::ActivationFunction, categorize::CatNetwork, dbg_println, generation::GenNetwork,
    neural_network::NeuralNetwork, series::Series, DEBUG,
};

// #[test]
// pub fn bench() {
//     bench!(CatNetwork::new(4, 4, 4, 1, ActivationFunction::Sigmoid));
//     let mut net = CatNetwork::new(2, 2, 2, 1, ActivationFunction::Sigmoid);
//     let _ = net.learn(&mut xor_file(), vec![types::Types::Float(0.0), types::Types::Float(1.0)], 0.5, "bench", 0.99, true);

// }

#[test]
pub fn train_test_xor() {
    let learning_rate = 0.5;
    let categories: Box<[String]> = vec!["1".to_string(), "0".to_string()].into_boxed_slice();
    let mut data: Box<[Series]> = xor_file();

    // Panics if unwraps a None value
    let mut model = train_network_xor(data.clone(), categories.clone(), learning_rate);
    data.iter_mut().for_each(|input| {
        input.answer = String::new();
    });
    model.test(&data, categories).unwrap();
}

/// Returns None if the learn function returns an Err variant
fn train_network_xor(
    data: Box<[Series]>,
    categories: Box<[String]>,
    learning_rate: f32,
) -> CatNetwork {
    let input_num = 2;
    let hidden_num = 2;
    let answer_num = 2;
    let hidden_layers = 2;
    let mut net = CatNetwork::new(
        input_num,
        hidden_num,
        answer_num,
        hidden_layers,
        Some(ActivationFunction::Sigmoid),
    );

    net.train(&data, categories, learning_rate, "xor", 99.0, true)
        .expect("Failed to train xor network");
    net
}

/// Read the file you want to and format it as Inputs
pub fn xor_file<'a>() -> Box<[Series]> {
    let file = match fs::File::open("training_data/xor.txt") {
        Ok(file) => file,
        Err(error) => panic!("Panic opening the file: {:?}", error),
    };

    let reader = BufReader::new(file);
    let mut inputs: Vec<Series> = vec![];

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
        let answer_inputs = String::from(init_inputs[init_inputs.len() - 1]); // TODO: Figure out what should be the row's answer; the last of a line for
        let input = Series::new(float_inputs, answer_inputs);
        inputs.push(input);
    }
    inputs.into_boxed_slice()
}

// TODO: Wite predictive model training first
// #[test]
// fn train_test_reverse() {
//     let learning_rate:f32 = 1.0;
//     let categories = categories_format()
// }

#[test]
fn train_test_digits() {
    let learning_rate: f32 = 0.01;
    let categories: Box<[String]> = vec!["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        .into_iter()
        .map(|n| n.to_string())
        .collect();
    let data: Box<[Series]> = digits_file();

    let mut model = train_network_digits(data.clone(), categories.clone(), learning_rate);

    model.test(&data, categories).unwrap();
}

/// # Panics
/// If the learn function returns an Err
fn train_network_digits(
    data: Box<[Series]>,
    categories: Box<[String]>,
    learning_rate: f32,
) -> CatNetwork {
    let mut net = CatNetwork::new(64, 128, 10, 2, Some(ActivationFunction::Sigmoid));

    net.train(&data, categories, learning_rate, "digits", 99.0, true)
        .expect("Training Digits Network Failed");
    net
}

fn digits_file() -> Box<[Series]> {
    let file = match fs::File::open("training_data/train-digits.txt") {
        Ok(file) => file,
        Err(error) => panic!("Panic opening the file: {:?}", error),
    };

    let reader = BufReader::new(file);
    let mut inputs: Vec<Series> = vec![];

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
        let input = Series::new(
            float_inputs,
            String::from(init_inputs[init_inputs.len() - 1]),
        );
        dbg_println!(
            "Correct Answer: {:?}",
            init_inputs[init_inputs.len() - 1].to_string()
        );
        inputs.push(input);
    }

    inputs.into_boxed_slice()
}

#[test]
fn train_test_gen() {
    let model_name = train_gen();
    let data = gen_data_file();
    let mut model = GenNetwork::read_model(model_name).unwrap();
    let output = model.test(&data).unwrap();
    println!("{:?}", output);
}

fn train_gen() -> String {
    let inputs = gen_data_file();
    let mut net = GenNetwork::new(8, 8, 8, 1, Some(ActivationFunction::Sigmoid));
    net.train(
        &inputs,
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
pub fn gen_data_file<'a>() -> Box<[Box<[f32]>]> {
    let file = match fs::File::open("training_data/train-digits.txt") {
        Ok(file) => file,
        Err(error) => panic!("Panic opening the file: {:?}", error),
    };

    let reader = BufReader::new(file);
    let mut inputs: Vec<Box<[f32]>> = vec![];

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
        inputs.push(float_inputs.into_boxed_slice());
    }
    inputs.into_boxed_slice()
}
