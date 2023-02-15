#[cfg(test)]
use core::panic;
use std::{io::{BufReader, BufRead}, fs};
use crate::input::Input;
use crate::DEBUG;
use crate::network::NeuralNetwork;

#[test]
fn train_test_xor() {
    let learning_rate:f32 = 0.5;
    let categories = vec![String::from("1"), String::from("0")];
    let data = xor_file();


    let model_name: String = train_network_xor(data.clone(), categories.clone(), learning_rate).unwrap();

    NeuralNetwork::test(data, categories, model_name);
}

fn train_network_xor(mut data:Vec<Input>, categories: Vec<String>, learning_rate: f32) -> Option<String> {
    let mut net = NeuralNetwork::new(2, 2, 2, 1);

    
    match net.learn(&mut data, categories, learning_rate) {

        Some(name) => Some(name),
        None => None
    }
}

// Read the file you want to and format it as Inputs
fn xor_file() -> Vec<Input> {
    let file = match fs::File::open("training_data/xor.txt") {
        Ok(file) => file,
        Err(error) => panic!("Panic opening the file: {:?}", error)
    };

    let reader = BufReader::new(file);
    let mut inputs: Vec<Input> = vec![];

    for l in reader.lines() {

        let line = match l {

            Ok(line) => line,
            Err(error) => panic!("{:?}", error)
        };

        let init_inputs: Vec<&str> = line.split(";").collect();
        let float_inputs: Vec<f32> = vec![init_inputs[0].split(" ").collect::<Vec<&str>>()[0].parse().unwrap(), init_inputs[0].split(" ").collect::<Vec<&str>>()[1].parse().unwrap()];

        let input: Input = Input { inputs: float_inputs, answer:init_inputs.get(init_inputs.len()-1).as_ref().unwrap().to_owned().to_string() };
        inputs.push(input);
    }

inputs  
}

#[test]
fn train_test_digits() {
    let learning_rate:f32 = 1.0;
    let categories = NeuralNetwork::categories_format(vec!["0","1","2","3","4","5","6","7","8","9"]);
    let data = digits_file();

    let model_name: String = train_network_digits(data.clone(), categories.clone(), learning_rate).unwrap();

    NeuralNetwork::test(data, categories, model_name);
}

fn train_network_digits(mut data: Vec<Input>, categories: Vec<String>, learning_rate: f32) -> Option<String> {
    let mut net = NeuralNetwork::new(64, 128, 10, 1);

    match net.learn(&mut data, categories, learning_rate) {

        Some(name) => Some(name),
        None => None
    }
}

fn digits_file() -> Vec<Input> {
    let file = match fs::File::open("training_data/train-digits.txt") {
        Ok(file) => file,
        Err(error) => panic!("Panic opening the file: {:?}", error)
    };

    let reader = BufReader::new(file);
    let mut inputs: Vec<Input> = vec![];

    for l in reader.lines() {

        let line = match l {

            Ok(line) => line,
            Err(error) => panic!("{:?}", error)
        };

        let init_inputs: Vec<&str> = line.split(",").collect();
        let mut float_inputs: Vec<f32> = vec![];

        for i in 0..init_inputs.len()-1 {
            float_inputs.push(init_inputs[i].parse().unwrap());
        }
        let input: Input = Input::new(float_inputs, init_inputs[init_inputs.len()-1].to_string());
        if DEBUG { println!("Correct Answer: {:?}", init_inputs[init_inputs.len()-1].to_string()); }
        inputs.push(input);
    }

inputs 
}