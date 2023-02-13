# DARJEELING
Machine learning tools for Rust

### Contact
elocolburn@comcast.net

# Installation
Add the following to your `Cargo.toml` file
```
darjeeling = "0.1.2"
```
# Example
```rust
    use core::{panic};
    use std::{fs};
    use crate::input::Input;
    use crate::neural_network::NeuralNetwork;
    use std::io::{BufReader, BufRead};

    fn train_test_xor() {
        let learning_rate:f32 = 1.0;
        let categories = vec![String::from("1"), String::from("0")];
        let data = xor_file();


        let model_name: String = train_network_xor(data.clone(), categories.clone(), learning_rate).unwrap();

        test_network_xor(data, categories, model_name)
    }

    fn train_network_xor(mut data:Vec<Input>, categories: Vec<String>, learning_rate: f32) -> Option<String> {
        let mut net = NeuralNetwork::new(2, 2, 2, 1, false);

        
        match net.learn(&mut data, categories, learning_rate) {

            Some(name) => Some(name),
            None => None
        }
    }

    fn test_network_xor(data: Vec<Input>, categories: Vec<String>, model_name: String) {

        NeuralNetwork::test(data, categories, model_name);
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
```
