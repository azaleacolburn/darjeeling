# DARJEELING
Machine learning tools for Rust

### Contact
elocolburn@comcast.net

# Installation
Add the following to your `Cargo.toml` file
```
darjeeling = "0.2.0"
```
# Example
```rust
    use core::{panic};
    use darjeeling::{ categorize::NeuralNetwork, tests, input::Input}
    use std::{io::{BufReader, BufRead}, fs};

    /// This program  function would read from a file containing all possible inputs to a binary logic gate, and all the correct answers.
    /// Then it would train a model with 1 hidden layer. 
    /// 2 nodes in its input layer, because there are two inputs.
    /// 2 Nodes for its output layer because there are two possible answers(the "brighter" one is selected a the chosen answer)
    /// and 2 nodes in its hidden layer, because I like patterns.
    /// If this doesn't work, check the tests.ts source code for verified working code.
    /// Hint: Try fiddling with the learning rate you're using if things aren't working properly
    /// Different problems work differently with different learning rates, although I recommend one of 0.5 to start.m
    fn train_test_xor() {
        let learning_rate:f32 = 1.0;
        let categories = NeuralNetwork::categories_format(vec!["0","1"]);
        let data = xor_file();


        let model_name: String = train_network_xor(data.clone(), categories.clone(), learning_rate).unwrap();

        test_network_xor(data, categories, model_name)
    }

    fn train_network_xor(mut data:Vec<Input>, categories: Vec<String>, learning_rate: f32) -> Option<String> {
        let input_num: i32 = 2;
        let hidden_num: i32 = 2;
        let answer_num: i32 = 2;
        let hidden_layers: i32 = 1;
        let mut net = NeuralNetwork::new(input_num, hidden_num, answer_num, hidden_layers);

        
        match net.learn(&mut data, categories, learning_rate) {

            Some(name) => Some(name),
            None => None
        }
    }

    fn test_network_xor(data: Vec<Input>, categories: Vec<String>, model_name: String) {

        NeuralNetwork::test(data, categories, model_name);
    }

    // This isn't very important, this just reads the file you want to and format it as Inputs
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
