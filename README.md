# DARJEELING
Machine learning tools for Rust

# Installation
Add the following to your `Cargo.toml` file
```
darjeeling = "0.1.0"
```
# Example
```
use darjeeling::{neural_network::NeuralNetwork, input::Input};
use std::io::{BufReader, BufRead};
use std::fs;

fn main() {
  train_network_xor(1.0);
}

// Create data, categories, and a network for models to be trained on
pub fn train_network_xor(learning_rate: f32) {
  let categories = vec![String::from("1"), String::from("0")];
  let mut data = xor_file();
  let mut net = NeuralNetwork::new(2, 2, 2, 1, false);
  net.learn(&mut data, categories, learning_rate);
}

// Read the file you want to and format it as Inputs
fn xor_file() -> Vec<Input> {
  let file = match fs::File::open("xor.txt") {
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
      let mut float_inputs: Vec<f32> = vec![init_inputs[0].split(" ").collect::<Vec<&str>>()[0].parse().unwrap(), init_inputs[0].split(" ").collect::<Vec<&str>>()[1].parse().unwrap()];

      let input: Input = Input { inputs: float_inputs, answer:init_inputs.get(init_inputs.len()-1).as_ref().unwrap().to_owned().to_string() };
      inputs.push(input);
  }

  inputs  
}
```
