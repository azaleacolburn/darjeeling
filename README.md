# DARJEELING
Machine learning and data manipulation tools for Rust

### Contact
elocolburn@comcast.net

# Installation
Add the following to your `Cargo.toml` file
```
darjeeling = "0.3.1"
```

# Basic Setup
1. Create a network
```rust
    use darjeeling::{
        categorize,
        activation::ActivationFunction
    };
    let input_num = 2;
    let hidden_num = 2;
    let answer_num = 2;
    let hidden_layers = 1;
    let mut net = categorize::NeuralNetwork::new(input_num, hidden_num, answer_num, hidden_layers, ActivationFunction::Sigmoid);
```
2. Format your data as Inputs
```rust
    use darjeeling::{
        types::Types,
        input::Input
    };
    // Do this for every input
    let float_inputs: Vec<f32> = vec![];
    let answer_input: Types = Types::String("Bee");
    let input = Input::new(float_inputs, answer_input);
```
Inputs represent a set of floating point numbers and which answer node they should be mapped to. For example, if the input is a picture of a Bee, the float_inputs might be the hex value of every pixel, while the answer input might be "Bee". Make sure the answer is always a valid category.
3. Train your network
```rust
    let learning_rate: f32 = 3.0;
    let categories: Vec<Types> = categories_str_format(vec!["Bee","3"]);
    let data: Vec<Input> = data_fmt();
    match net.learn(&mut data, categories, learning_rate, "bees3s")d {
        // Do whatever you want with this data
        Ok((model_name, err_percent, mse)) => Some(()),
        Err(_err) => None
    }
```
If the training is successful, the model_name is returned along with the percent of the training inputs the network correctly categorized on it's last epoch, and the mean squared error of the training.
4. Test your network
```rust
    match categorize::NeuralNetwork::test(data, categories, model_name) {
        Ok() => 
    };
```
During testing, the answers in the input data should be set to None. The testing returns a vector of all the categories assigned to the data in the same order as the data.

# Examples
## Categorization
This program reads from a file containing all possible 
inputs to a binary logic gate, and all the correct answers.

Then it trains a model with 1 hidden layer. 
2 nodes in its input layer, because there are two inputs.
2 Nodes for its output layer because there are two possible answers(the "brighter" one is selected a the chosen answer)
And 2 nodes in its hidden layer, because I like patterns.

If this doesn't work, check the tests.ts source code for verified working code.

Hint: Try fiddling with the learning rate you're using if things aren't working properly.

Different problems work differently with different learning rates, although I recommend one of 0.5 to start.
```rust
    use core::{panic};
    use darjeeling::{ categorize::NeuralNetwork, tests, input::Input, series::Series, dataframe::{DataFrame, Point}};
    use std::{io::{BufReader, BufRead}, fs};

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

        
        match net.learn(&mut data, categories, learning_rate).unwrap()
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
## Generation

This program doesn't have a large enough dataset to get interesting results.
All it does is just create a network and 
```rust
    use darjeeling::{
         generation::NeuralNetwork,
         activation::ActivationFunction,
         input::Input, 
         // This file may not be avaliable
         // Everything found here will be hyper-specific to your project.
         tests::{categories_str_format, file}
    };
     
    // A file with data
    // To make sure the networked is properly trained, make sure it follows some sort of pattern
    // This is just sample data, for accurate results, around 3800 datapoints is needed
    // 1 2 3 4 5 6 7 8
    // 3 2 5 4 7 6 1 8
    // 0 2 5 4 3 6 1 8
    // 7 2 3 4 9 6 1 8
    // You also need to write the file input function
    // Automatic file reading and formatting function coming soon
    let mut data: Vec<Input> = file();
    let mut net = NeuralNetwork::new(2, 2, 2, 1, ActivationFunction::Sigmoid);
    let learning_rate = 1.0;
    let model_name = net.learn(&mut data, categories, learning_rate, "gen").unwrap();
    let new_data: Vec<Input> = net.test(data).unwrap();
```

# TODO:
- Add Support for [Polars](https://www.pola.rs/) Dataframes

Dataframes are now deprecated
