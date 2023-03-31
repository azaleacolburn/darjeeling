# DARJEELING
Machine learning and data manipulation tools for Rust

### Contact
elocolburn@comcast.net

# Installation
Add the following to your `Cargo.toml` file
```
darjeeling = "0.2.3"
```
# Example
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
This program creates a test dataframe and series, then performs several test actions on them.

I'm sorry about the formatting, I'm not very good at IO formatting

Feel Free to reach out and offer a better DataFrame display function

Expected Output:
```
       "col1" "col2" "col3" 
 "row1" 0    1    2    
 "row2" 3    4    5    
 "row3" 6    7    8    
       "col1" "col2" "col3" 
 "row1"   0      1      2      
 "row2"   3      4      5      
 "row3"   6      7      8      
 "Label!" 10    11    12    
       "col1" "col2" "col3" 
 "row1" 0    1    2    
 "row2" 3    4    5    
 "row3" 6    7    8

1 "data1"
2 "data2"
3 "data3"


1 "data1"
2 "data2"
3 "data3"
4 "data4"


1 "data1"
2 "data2"
3 "data3"
4 "data4"
5 "data5"


1 "data1"
2 "data2"
3 "data3"
4 "data4"
```
```rust
    fn dataframe_add_sub() {
        let mut frame: DataFrame<i32> = quick_frame();
        frame.display();
        frame.add_row("Label!", 
            Point::point_vector(
                frame.get_cols_len() - 1 as usize, 
                vec![10,11,12]
            )
        );
        frame.display();
        frame.delete_row("Label!").unwrap();
        frame.display()
    }
  
    fn series_add_sub() {
        let mut series = quick_series();
        series.display();
        series.mut_add("data4", 4);
        series.display();
        let mut new_series: Series<&str, i32> = series.no_mut_add("data5", 5);
        new_series.display();
        new_series.mut_sub(5).unwrap();
        new_series.display();
    }

    // Generates example datastructures
    fn quick_frame() -> DataFrame<'static, i32> {
        DataFrame::new(
            vec![
                vec![0,1,2],
                vec![3,4,5],
                vec![6,7,8]
            ],
            vec!["row1", "row2", "row3"],
            vec!["col1", "col2", "col3"]
        )
    }

    fn quick_series() -> Series<&'static str, i32> {

        Series::new(
            vec!["data1", "data2", "data3"],
            vec![1,2,3]
        )
    }
```
