# darjeeling

[![Latest version](https://img.shields.io/crates/v/darjeeling.svg?style=for-the-badge)](https://crates.io/crates/darjeeling)
[![Documentation](https://img.shields.io/docsrs/darjeeling.svg?style=for-the-badge)](https://docs.rs/darjeeling)
[![License](https://img.shields.io/crates/l/darjeeling.svg?style=for-the-badge)](https://github.com/azaleacolburn/darjeeling/blob/main/LICENSE)

Machine learning tools for Rust

# Installation

Add the following dependencies to your `Cargo.toml` file

```toml
darjeeling = "0.3.5"
```

# Basic Setup

1. Create a network

``` rust
    use darjeeling::{
        activation::ActivationFunction,
        categorize::CatNetwork,
        neural_network::NeuralNetwork
    };
    
    let inputs: usize = 10;
    let hidden: usize = 40;
    let answer: usize = 2;
    let hidden_layers: usize = 1;
    let mut net = CatNetwork::new(inputs, hidden, answer, hidden_layers, Some(ActivationFunction::Sigmoid));
```
You can also add hidden layers with a set number of neurons, since during initialization, all hidden layers must be the same size.

2. Format your data as a set of Series

```rust
    use darjeeling::{
        series::Series
    };
    // Do this for every input
    let float_inputs: Box<[Series]> = vec![0.0, 3.2, 4.2, 5.3, 663.3, 35.2];
    let answer_input: = "Bee";
    let input = Series::new(float_inputs, answer_input);
```

A Series represents a single data point:
- A set of floating point numbers and which is their correct answer.
For example, if the input is a picture of a Bee, the float_inputs might be the hex value of every pixel, while the answer input might be "Bee". Make sure the answer is always a valid category.

3. Train your network

```rust
    let learning_rate: f32 = 3.0;
    let categories: Vec<> = vec!["Bee", "3"];
    let data: Vec<Input> = data_fmt();
    let target_err_percent = 95.0;
    match net.learn(&mut data, categories, learning_rate, "bees3s", target_err_percent) {
        // Do whatever you want with this data
        Ok((model_name, err_percent, mse)) => Some(()),
        Err(_err) => None
    }
```

If the training is successful, the model_name is returned along with the percent of the training inputs the network correctly categorized on it's last epoch, and the mean squared error of the training.

4. Test your network

```rust
    // Do whatever you want with this data
    match categorize::NeuralNetwork::test(data, categories, model_name) {
        // Vec<Types>
        Ok(types) => {},
        // DarjeelingError
        Err(err) => {}
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

        NeuralNetwork::test(data, categories, model_name);
    }

    fn train_network_xor(mut data:Vec<Input>, categories: Vec<String>, learning_rate: f32) -> Option<String> {
        let input_num: i32 = 2;
        let hidden_num: i32 = 2;
        let answer_num: i32 = 2;
        let hidden_layers: i32 = 1;
        let model_name: &str = "xor";
        let target_err_percent =  99.0;
        // Creates a new Neural Network
        let mut net = NeuralNetwork::new(input_num, hidden_num, answer_num, hidden_layers);
        // Trains the Neural Network
        match net.learn(&mut data, categories, learning_rate, model_name, target_err_percent) {
            // Mean Squared Error
            Ok((model_name, _err_percent, _mse)) => Some(model_name),
            Err(_err) => None
        }
    }
```

## Generation

DISCLAIMER: I don't actually recommend using the generative AI feature for interesting results, it's horribly unoptimized, and isn't training very well right now.

This program doesn't have a large enough dataset to get interesting results.
All it does is just trains a network to verify a proof of concept.

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
    // This is just sample data, for accurate results, around 3800 datapoints are needed
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

# FAQ

## Why is it called Darjeeling?

Because that's the WiFi password at the Tea Shop where I do most of my programming.

# Contribution

If you want to contribute, check the todo or issues list, fork the code, and make a pull request whenever you're ready. I'm more than open to reviewing any code people want to add. I'd be happy to give any assistance to people wanting to contribute, including but not limited to teaching about Machine Learning, Rust, and how Darjeeling works. We welcome all experience levels here. Just email me if you need anything. If there's a bug that you can't/don't want to fix, open an issue. I hope to one day make this a great project to get started with open source on, although there isn't the infrastructure in place right now to make that happen.

## Guidelines

- Be kind, respectful, and patient.
- Don't be an "AI Bro"

# Contact Me (azaleacolburn@gmail.com)

# TODO:

- [ ] Make generation better (seriously, it doesn't give good results right now)
- [ ] Add Support for [Polars](https://www.pola.rs/) Dataframes
- [ ] Make data manipulation and input formatting easier
- [ ] Optimize a lot

# Notes
- Darjeelings implementations of Dataframes is now deprecated
- I don't like AI, I'm just doing this because I like Rust, programming. Don't let AI slop replace human innovation and art.
