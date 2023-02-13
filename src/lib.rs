pub static DEBUG: bool = false;

#[allow(dead_code)]
mod node {

    use serde::{Deserialize, Serialize};
    use crate::DEBUG;

    /// Represents a node in the network
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Node {
        pub link_weights: Vec<f32>,
        pub link_vals: Vec<Option<f32>>,
        pub links: i32,
        pub err_sig: Option<f32>,
        pub correct_answer: Option<f32>,
        pub cached_output: Option<f32>,
        pub category: Option<String>,
        pub b_weight: Option<f32>,
    }

    impl Node {
        fn input(&mut self) -> Option<f32>{
            let mut sum:f32 = 0.00;
            for i in 0..self.links {
                if DEBUG { println!("Node: {:?}", self); }
                sum += self.link_vals[i as usize].unwrap() * self.link_weights[i as usize];
            }
            Some(sum + self.b_weight.unwrap())
        }
    
        pub fn output(&mut self) -> Option<f32> {
            self.cached_output = Some(Node::sigmoid(self.input().unwrap()));
            
            self.cached_output
        }
        
        pub fn compute_answer_err_sig(&mut self){
            if DEBUG { println!("Err Signal Pre: {:?}", self.err_sig); }
            self.err_sig = Some((self.correct_answer.unwrap() - self.cached_output.unwrap()) * self.cached_output.unwrap() * (1.00 - self.cached_output.unwrap()));
            if DEBUG { println!("Err Signal Post: {:?}", self.err_sig.unwrap()) }
        }
    
        pub fn adjust_weights(&mut self, learning_rate: f32){
            self.b_weight = Some(self.b_weight.unwrap() + self.err_sig.unwrap() * learning_rate);
            for link in 0..self.links {
                if DEBUG {
                    println!("\nInitial weights: {:?}", self.link_weights[link as usize]);
                    println!("Link Value: {:?}", self.link_vals[link as usize]);
                    println!("Err: {:?}", self.err_sig);
                }
                self.link_weights[link as usize] += self.err_sig.unwrap() * self.link_vals[link as usize].unwrap() * learning_rate;
                if DEBUG { println!("Adjusted Weight: {:?}", self.link_weights[link as usize]); }
            }
        }
    
        fn sigmoid(x: f32) -> f32{
    
            1.0/(1.0+((-x).exp()))
        }
    }
}

#[allow(dead_code)]
pub mod neural_network {
    use crate::DEBUG;
    use core::panic;
    use std::{fs, path::Path};
    use serde::{Deserialize, Serialize};
    use rand::{Rng, seq::SliceRandom, thread_rng};
    use crate::node::Node;
    use crate::input::Input;

    /// The top-level neural network struct
    /// sensor and answer represents which layer sensor and answer are on
    #[derive(Debug, Serialize, Deserialize)]
    pub struct NeuralNetwork {
        node_array: Vec<Vec<Node>>,
        sensor: Option<usize>,
        answer: Option<usize>,
    }

    impl NeuralNetwork {
        
        /// Constructor function for the neural network
        /// Fills a Neural Network's node_array with empty nodes. 
        /// Initializes random starting link and bias weights between -.5 and .5
        /// 
        /// # Params
        /// - Inputs: The number of sensors in the input layer
        /// - Hidden: The number of hidden nodes in each layer
        /// - Answer: The number of answer nodes, or possible categories
        /// - Hidden Layers: The number of different hidden layers
        /// 
        /// # Examples
        /// ```
        /// use darjeeling::neural_network::NeuralNetwork;
        /// 
        /// let inputs: i32 = 10;
        /// let hidden: i32 = 40;
        /// let answer: i32 = 2;
        /// let hidden_layers: i32 = 1;
        /// let mut net: NeuralNetwork = NeuralNetwork::new(inputs, hidden, answer, hidden_layers);
        /// ```
        pub fn new(input_num: i32, hidden_num: i32, answer_num: i32, hidden_layers: i32) -> NeuralNetwork {
            let mut net: NeuralNetwork = NeuralNetwork { node_array: vec![], sensor: Some(0), answer: Some(hidden_layers as usize + 1) };
            let mut rng = rand::thread_rng();

            net.node_array.push(vec![]);    
            for _i in 0..input_num {
                net.node_array[net.sensor.unwrap()].push(Node { link_weights: vec![], link_vals: vec![], links: 0, err_sig: None, correct_answer: None, cached_output: None, category: None, b_weight: None });
            }

            for i in 1..hidden_layers + 1 {
                let mut hidden_vec:Vec<Node> = vec![];
                let hidden_links = net.node_array[(i - 1) as usize].len() as i32;
                if DEBUG { println!("Hidden Links: {:?}", hidden_links) }
                for _j in 0..hidden_num{
                    hidden_vec.push(Node { link_weights: vec![], link_vals: vec![], links: hidden_links, err_sig: None, correct_answer: None, cached_output: None, category: None, b_weight: None });
                }
                net.node_array.push(hidden_vec);
            }

            net.node_array.push(vec![]);
            let answer_links = net.node_array[hidden_layers as usize].len() as i32;
            println!("Answer Links: {:?}", answer_links);
            for _i in 0..answer_num {
                net.node_array[net.answer.unwrap()].push(Node { link_weights: vec![], link_vals: vec![], links: answer_links, err_sig: None, correct_answer: None, cached_output: Some(0.0), category: None, b_weight: None });
            }
            
            for layer in &mut net.node_array{
                for node in layer{
                    node.b_weight = Some(rng.gen_range(-0.5..0.5));
                    if DEBUG { println!("Made it to pushing link weights") }
                    for _i in 0..node.links {
                        node.link_weights.push(rng.gen_range(-0.5..0.5));
                        node.link_vals.push(None);
                    }
                }
            
            }
            net
        }

        /// Trains the neural network model to be able to categorize data in a dataset
        /// 
        /// # Params
        /// - Data: List of inputs
        /// - Categories: List of Strings, each denoting an answer category. 
        /// The number of answer nodes should be the same of the number of categories
        /// - Learning Rate: The modifier that is applied to link weights as they're adjusted.
        /// Try fiddling with this one, but 1.5 - 0.5 is recommended to start.
        /// 
        /// # Returns
        /// The name of the model that this neural network trained, returned in the option enum.
        /// 
        /// # Examples
        /// ```rust
        /// use darjeeling::{neural_network::NeuralNetwork, input::Input};
        /// 
        /// let categories: Vec<String> = NeuralNetwork::categories_format(vec!["0", "1"]);
        /// // xor_file() needs to be written by you, see README.md for example
        /// let mut data: Vec<Input> = xor_file();
        /// let mut net = NeuralNetwork::new(2, 2, 2, 1);
        /// let learning_rate = 1.0;
        /// let model_name = net.learn(&mut data, categories, learning_rate).unwrap();
        /// ```
        /// 
        pub fn learn(&mut self, data: &mut Vec<Input>, categories: Vec<String>, learning_rate: f32) -> Option<String> {
            let mut epochs:f32 = 0.0;
            let mut sum:f32 = 0.0;
            let mut count:f32 = 0.0;
            let mut err_percent:f32 = 0.0;
            let hidden_layers = self.node_array.len() - 2;

            self.categorize(categories);

            while err_percent < 99.0 {
                count = 0.0;
                sum = 0.0;
                data.shuffle(&mut thread_rng());

                for line in 0..data.len() {
                    if DEBUG { println!("Training Checkpoint One Passed") }
                    self.assign_answers(data, line as i32);

                    self.push_downstream(data, line as i32);

                    if DEBUG { println!("Sum: {:?} Count: {:?}", sum, count); }

                    self.self_analysis(Some(epochs), &mut sum, &mut count, data, line);

                    if DEBUG { println!("Sum: {:?} Count: {:?}", sum, count); }
                    
                    self.backpropogate(learning_rate, hidden_layers as i32);
                }

                // let _old_err_percent = err_percent;
                err_percent = (sum/count) * 100.0;
                epochs += 1.0;
                println!("Epoch: {:?}", epochs);
                println!("Training Accuracy: {:?}", err_percent);
                //if err_percent - old_err_percent < 0.00000001 { break; }

            }

            let name: String = self.write_model().unwrap();
            println!("Training: Finished with accuracy of {:?}/{:?} or {:?} percent after {:?} epochs", sum, count, err_percent, epochs);

            Some(name)
        }

        /// Tests a pretrained model
        pub fn test(mut data: Vec<Input>, categories: Vec<String>, model_name: String) -> Option<String> {
            let mut sum:f32 = 0.0;
            let mut count:f32 = 0.0;
            let mut category: String = String::from("");

            let mut net: NeuralNetwork = NeuralNetwork::read_model(model_name);

            for node in 0..net.node_array[net.answer.unwrap()].len() {
                net.node_array[net.answer.unwrap()][node].category = Some(categories[node].clone());
                if DEBUG { println!("{:?}", net.node_array[net.answer.unwrap()][node].category); }
            }

            data.shuffle(&mut thread_rng());

            for line in 0..data.len() {
                if DEBUG { println!("Training Checkpoint One Passed") }
                net.assign_answers(&mut data, line as i32);

                net.push_downstream(&mut data, line as i32);

                if DEBUG { println!("Sum: {:?} Count: {:?}", sum, count); }

                category = net.self_analysis(None, &mut sum, &mut count, &mut data, line);

                if DEBUG { println!("Sum: {:?} Count: {:?}", sum, count); }

                println!("Correct answer: {:?}", data[line].answer)
            }

            // let _old_err_percent = err_percent;
            let err_percent: f32 = (sum/count) * 100.0;
            println!("Testing: Finished with accuracy of {:?}/{:?} or {:?} percentt", sum, count, err_percent);

            Some(category)
        }

        /// Assigns categories to answer nodes based on a list of given categories
        fn categorize(&mut self, categories: Vec<String>) {
            let mut count:usize = 0;
            self.node_array[self.answer.unwrap()].iter_mut().for_each(|node| {
                node.category = Some(categories[count].clone());
                count += 1;
            });
        }

        fn assign_answers(&mut self, data: &mut Vec<Input>, line: i32){
            for node in 0..self.node_array[self.answer.unwrap()].len() {
                if self.node_array[self.answer.unwrap()][node].category.as_ref().unwrap().eq(&data[line as usize].answer) {
                    self.node_array[self.answer.unwrap()][node].correct_answer = Some(1.0);
                } else {
                    self.node_array[self.answer.unwrap()][node].correct_answer = Some(0.0);
                }
            }
        }
        
        /// Passes in data to the sensors, pushs data 'downstream' through the network
        fn push_downstream(&mut self, data: &mut Vec<Input>, line: i32){

            // Passes in data for input layer
            for i in 0..self.node_array[self.sensor.unwrap()].len(){
                let input  = data[line as usize].inputs[i];

                self.node_array[self.sensor.unwrap()][i].cached_output = Some(input);
            }

            // Feed-forward values for hidden and output layers
            for layer in 1..self.node_array.len() {

                for node in 0..self.node_array[layer].len() {

                    for prev_node in 0..self.node_array[layer-1].len() {
                        
                        // self.node_array[layer][node].link_vals.push(self.node_array[layer-1][prev_node].cached_output.unwrap());
                        self.node_array[layer][node].link_vals[prev_node] = Some(self.node_array[layer-1][prev_node].cached_output.unwrap());
                        // self.node_array[layer][node].output();
                        if DEBUG { if layer == self.answer.unwrap() { println!("Ran output on answer {:?}", self.node_array[layer][node].cached_output) } }
                    }
                    self.node_array[layer][node].output();
                }
            }
        }

        /// Analyses the chosen answer node's result.
        /// Also increments sum and count
        fn self_analysis<'a>(&'a self, epochs: Option<f32>, sum: &'a mut f32, count: &'a mut f32, data: &mut Vec<Input>, line: usize) -> String {

            let brightest_node: &Node = &self.node_array[self.answer.unwrap()][self.largest_node()];
            let brightness: f32 = brightest_node.cached_output.unwrap();

            if !(epochs.is_none()) {
                if epochs.unwrap() % 10 as f32 == 0.0 {
                    println!("\n-------------------------\n");
                    println!("Epoch: {:?}", epochs);
                    if DEBUG {
                        let non_brightest_node: &Node = &self.node_array[self.answer.unwrap()][self.node_array[self.answer.unwrap()].len()-1-self.largest_node()];
                        println!("Non category: {:?} \nnon-brightest-brightness: {:?}", non_brightest_node.category.as_ref().unwrap(), non_brightest_node.cached_output.unwrap());
                    }
                }
            }

            println!("Category: {:?} \nBrightness: {:?}", brightest_node.category.as_ref().unwrap(), brightness);
            if brightest_node.category.as_ref().unwrap() == &data[line].answer { println!("Correct Answer Chosen"); }

            if brightest_node.category.as_ref().unwrap() == &data[line].answer {
                if DEBUG { println!("Sum++"); }
                *sum += 1.0;
            }
            *count += 1.0;

            brightest_node.category.clone().unwrap()
        }

        /// Finds the index and the brightest node in an array and returns it
        fn largest_node(&self) -> usize {
            let mut largest_node = 0;
            for node in 0..self.node_array[self.answer.unwrap()].len() {
                if self.node_array[self.answer.unwrap()][node].cached_output > self.node_array[self.answer.unwrap()][largest_node].cached_output {
                    largest_node = node;
                }
            }

            largest_node
        }
        /// Goes back through the network adjusting the weights of the all the neurons based on their error signal
        fn backpropogate(&mut self, learning_rate: f32, hidden_layers: i32) {

            for answer in 0..self.node_array[self.answer.unwrap()].len() {
                if DEBUG { println!("Node: {:?}", self.node_array[self.answer.unwrap()][answer]); }

                self.node_array[self.answer.unwrap()][answer].compute_answer_err_sig();

                if DEBUG { println!("Error: {:?}", self.node_array[self.answer.unwrap()][answer].err_sig.unwrap()) }
            }

            self.adjust_hidden_weights(learning_rate, hidden_layers);

            // Adjusts weights for answer neurons
            for answer in 0..self.node_array[self.answer.unwrap()].len() {
                self.node_array[self.answer.unwrap()][answer].adjust_weights(learning_rate);
            }
        }

        #[allow(non_snake_case)]
        /// Adjusts the weights of all the hidden neurons in a network
        fn adjust_hidden_weights(&mut self, learning_rate: f32, hidden_layers: i32) {

            // HIDDEN represents the layer, while hidden represents the node of the layer
            for HIDDEN in 1..(hidden_layers + 1) as usize {
                
                for hidden in 0..self.node_array[HIDDEN].len() {
                    
                    self.node_array[HIDDEN as usize][hidden].err_sig = Some(0.0);

                    for next_layer in 0..self.node_array[(HIDDEN + 1 )].len() {

                        let next_weight = self.node_array[(HIDDEN + 1)][next_layer].link_weights[hidden];
                        let new_error_signal = self.node_array[HIDDEN][hidden].err_sig.unwrap() + (self.node_array[(HIDDEN + 1)][next_layer].err_sig.unwrap() * next_weight);
                        if DEBUG { 
                            println!("next err sig {:?}", self.node_array[(HIDDEN + 1)][next_layer].err_sig.unwrap());
                            println!("next weight {:?}", next_weight);
                            println!("new hidden errsig add {:?}", new_error_signal);
                        }
                        self.node_array[HIDDEN][hidden].err_sig = Some(new_error_signal);
                    }
                    let hidden_result = self.node_array[HIDDEN][hidden].cached_output.unwrap();
                    let multiplied_value = self.node_array[HIDDEN][hidden].err_sig.unwrap() * (hidden_result * (1.0 - hidden_result));
                    if DEBUG { println!("new hidden errsig multiply: {:?}", multiplied_value); }
                    self.node_array[HIDDEN][hidden].err_sig = Some(multiplied_value);

                    if DEBUG { 
                        println!("\nLayer: {:?}", HIDDEN);
                        println!("Node: {:?}", hidden) 
                    }

                    self.node_array[HIDDEN][hidden].adjust_weights(learning_rate);
                }
            }
        }

        /// Serializes a trained model so it can be used later
        /// 
        /// # Returns
        /// The name of the model
        pub fn write_model(&self) -> Option<String>{
            
            let mut rng = rand::thread_rng();
            let file_num: u32 = rng.gen();
            let name = format!("model{:?}", file_num);

            match Path::new(&name).try_exists() {

                Ok(false) => {
                    let _file = fs::File::create(&name).unwrap();
                    let serialized = serde_json::to_string(&self).unwrap();
                    if DEBUG { println!("Serialized: {:?}", serialized); }
                    match fs::write(&name, serialized) {
                        
                        Ok(()) => {
                            println!("Model {:?} Saved", file_num);

                            Some(name)
                        },
                        Err(error) => {
                            panic!("Cannot write to the file: {:?}", error);
                        }
                    }
                },
                Ok(true) => {

                    None
                },

                Err(error) => panic!("{:?}", error)
            }
        }

        /// Reads a serizalized Neural Network
        /// 
        /// # Params
        /// - Model Name: The name(or more helpfully the path) of the model to be read
        /// 
        /// # Returns
        /// A neural network read from a serialized neural network file
        pub fn read_model(model_name: String) -> NeuralNetwork {

            println!("Loading model");

            let serialized_net: String = match fs::read_to_string(model_name) {

                Ok(serizalized_net) => serizalized_net,
                Err(error) => panic!("{:?}", error)
            };
            
            let net: NeuralNetwork = match serde_json::from_str(&serialized_net) {

                Ok(net) => net,
                Err(error) => panic!("{:?}", error),
            };

            net
        }
        /// Formats cateories from a vector of string slices to a vector of strings
        pub fn categories_format(categories_str: Vec<&str>) -> Vec<String> {
            let mut categories:Vec<String> = vec![];
            for category in categories_str {
                categories.push(category.to_string());
            }
        
            categories
        }
        
    }
}

#[allow(dead_code)]
pub mod input {

    #[derive(Debug, Clone)]
    /// Represents the input for the neural network, in a way it can understand
    pub struct Input {
        pub inputs: Vec<f32>,
        pub answer: String
    }

    impl Input {

        /// Creates new input
        /// # Params
        /// - Inputs: A list of 32-bit floating point numbers.
        /// - Answer: A string, representing the correct answer(or cateogry) of this input.
        /// 
        /// # Examples
        /// This example is for one input into an XOR gate
        /// ```
        /// use darjeeling::input::Input;
        /// let inputs: Vec<f32> = vec![0.0,1.0];
        /// let answer: String = String::from("1");
        /// let formated_input: Input = Input::new(inputs, answer);
        /// ```
        pub fn new(inputs: Vec<f32>, answer: String) -> Input {

            Input { inputs, answer }
        }

        // TODO: Write format_as_input function
    }
    
}

#[cfg(test)]
pub mod tests {
    use core::panic;
    use std::{io::{BufReader, BufRead}, fs};
    use crate::input::Input;
    use crate::neural_network::NeuralNetwork;

    #[test]
    fn train_test_xor() {
        let learning_rate:f32 = 1.0;
        let categories = vec![String::from("1"), String::from("0")];
        let data = xor_file();


        let model_name: String = train_network_xor(data.clone(), categories.clone(), learning_rate).unwrap();

        test_network_xor(data, categories, model_name)
    }

    fn train_network_xor(mut data:Vec<Input>, categories: Vec<String>, learning_rate: f32) -> Option<String> {
        let mut net = NeuralNetwork::new(2, 2, 2, 1);

        
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
}