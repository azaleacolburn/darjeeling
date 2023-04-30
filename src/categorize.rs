use crate::{
    DEBUG,
    error::DarjeelingError,
    types::Types,
    node::Node,
    input::Input
};
use std::{fs, path::Path};
use serde::{Deserialize, Serialize};
use rand::{Rng, seq::SliceRandom, thread_rng};

/// The top-level neural network struct
/// sensor and answer represents which layer sensor and answer are on
#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetwork {
    node_array: Vec<Vec<Node>>,
    sensor: Option<usize>,
    answer: Option<usize>,
    parameters: Option<u128>
}

impl NeuralNetwork {
    
    /// Constructor function for the neural network
    /// Fills a Neural Network's node_array with empty nodes. 
    /// Initializes random starting link and bias weights between -.5 and .5
    /// 
    /// ## Params
    /// - Inputs: The number of sensors in the input layer
    /// - Hidden: The number of hidden nodes in each layer
    /// - Answer: The number of answer nodes, or possible categories
    /// - Hidden Layers: The number of different hidden layers
    /// 
    /// ## Examples
    /// ``` rust
    /// use darjeeling::categorize::NeuralNetwork;
    /// 
    /// let inputs: i32 = 10;
    /// let hidden: i32 = 40;
    /// let answer: i32 = 2;
    /// let hidden_layers: i32 = 1;
    /// let mut net: NeuralNetwork = NeuralNetwork::new(inputs, hidden, answer, hidden_layers);
    /// ```
    pub fn new(input_num: i32, hidden_num: i32, answer_num: i32, hidden_layers: i32) -> NeuralNetwork {
        let mut net: NeuralNetwork = NeuralNetwork { node_array: vec![], sensor: Some(0), answer: Some(hidden_layers as usize + 1), parameters: None};
        let mut rng = rand::thread_rng();
        net.node_array.push(vec![]);    
        for _i in 0..input_num {
            net.node_array[net.sensor.unwrap()].push(Node::new(&vec![], None));
        }

        for i in 1..hidden_layers + 1 {
            let mut hidden_vec:Vec<Node> = vec![];
            let hidden_links = net.node_array[(i - 1) as usize].len();
            if DEBUG { println!("Hidden Links: {:?}", hidden_links) }
            for _j in 0..hidden_num{
                hidden_vec.push(Node { link_weights: vec![], link_vals: vec![], links: hidden_links, err_sig: None, correct_answer: None, cached_output: None, category: None, b_weight: None });
            }
            net.node_array.push(hidden_vec);
        }

        net.node_array.push(vec![]);
        let answer_links = net.node_array[hidden_layers as usize].len();
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
        let mut params = 0;
        for i in 0..net.node_array.len() {
            for j in 0..net.node_array[i].len() {
                params += 1 + net.node_array[i][j].links as u128;
            }
        }
        net.parameters = Some(params);
        net
    }

    /// Trains the neural network model to be able to categorize data in a dataset
    /// 
    /// # Params
    /// - Data: List of inputs
    /// - Categories: List of Strings, each denoting an answer category. 
    /// The number of answer nodes should be the same of the number of categories
    /// - Learning Rate: The modifier that is applied to link weights as they're adjusted.
    /// Try fiddling with this one, but -1.5 - 1.5 is recommended to start.
    /// 
    /// # Returns
    /// The falable name of the model that this neural network trained
    /// 
    /// # Err
    /// 
    /// 
    /// # Examples
    /// ```ignore
    /// use darjeeling::{categorize::NeuralNetwork, input::Input, tests::{categories_str_format, xor_file}};
    /// 
    /// let categories: Vec<String> = categories_str_format(vec!["0", "1"]);
    /// // A file containing all possible inputs and correct outputs still needs to be make by you
    /// // 0 0;0
    /// // 0 1;1
    /// // 1 0;1
    /// // 1 1;0
    /// let mut data: Vec<Input> = xor_file();
    /// let mut net = NeuralNetwork::new(2, 2, 2, 1);
    /// let learning_rate = 1.0;
    /// let model_name = net.learn(&mut data, categories, learning_rate).unwrap();
    /// ```
    /// 
    pub fn learn(&mut self, data: &mut Vec<Input>, categories: Vec<Types>, learning_rate: f32) -> Result<String, DarjeelingError> {
        let mut epochs: f32 = 0.0;
        let mut sum: f32 = 0.0;
        let mut count: f32 = 0.0;
        let mut err_percent: f32 = 0.0;
        let hidden_layers = self.node_array.len() - 2;

        self.categorize(categories);

        while err_percent < 90.0 {
            count = 0.0;
            sum = 0.0;
            data.shuffle(&mut thread_rng());

            for line in 0..data.len() {
                if DEBUG { println!("Training Checkpoint One Passed") }
                self.assign_answers(data, line as i32);

                self.push_downstream(data, line as i32);

                if DEBUG { println!("Sum: {:?} Count: {:?}", sum, count); }

                self.self_analysis(&mut Some(epochs), &mut sum, &mut count, data, line);

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
        #[allow(unused_mut)]
        let mut name: String;
        match self.write_model() {

            Ok(model_name) => {
                name = model_name;
            },

            Err(error) => return Err(error)
        }

        println!("Training: Finished with accuracy of {:?}/{:?} or {:?} percent after {:?} epochs", sum, count, err_percent, epochs);

        Ok(name)
    }

    /// Tests a pretrained model
    pub fn test(mut data: Vec<Input>, categories: Vec<Types>, model_name: String) -> Result<Types, DarjeelingError<'static>> {
        let mut sum:f32 = 0.0;
        let mut count:f32 = 0.0;
        let mut category: Option<Types> = None;

        let mut net: NeuralNetwork = match NeuralNetwork::read_model(model_name.clone()) {

            Ok(net) => net,
            Err(error) => return Err(DarjeelingError::ReadModelFunctionFailed(model_name, Box::new(error)))
        };

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

            category = Some(net.self_analysis(&mut None, &mut sum, &mut count, &mut data, line));

            if DEBUG { println!("Sum: {:?} Count: {:?}", sum, count); }

            println!("Correct answer: {:?}", data[line].answer)
        }

        // let _old_err_percent = err_percent;
        let err_percent: f32 = (sum/count) * 100.0;
        println!("Testing: Finished with accuracy of {:?}/{:?} or {:?} percentt", sum, count, err_percent);

        Ok(category.unwrap())
    }

    /// Assigns categories to answer nodes based on a list of given categories
    fn categorize(&mut self, categories: Vec<Types>) {
        let mut count:usize = 0;
        self.node_array[self.answer.unwrap()].iter_mut().for_each(|node| {
            node.category = Some(categories[count].clone());
            count += 1;
        });
    }
    
    fn assign_answers(&mut self, data: &mut Vec<Input>, line: i32){
        for node in 0..self.node_array[self.answer.unwrap()].len() {
            if *self.node_array[self.answer.unwrap()][node].category.as_ref().unwrap() == data[line as usize].answer {
                self.node_array[self.answer.unwrap()][node].correct_answer = Some(1.0);
            } else {
                self.node_array[self.answer.unwrap()][node].correct_answer = Some(0.0);
            }
        }
    }

    /// Passes in data to the sensors, pushs data 'downstream' through the network
    fn push_downstream(&mut self, data: &mut Vec<Input>, line: i32) {

        // Passes in data for input layer
        for i in 0..self.node_array[self.sensor.unwrap()].len() {
            let input  = data[line as usize].inputs[i];

            self.node_array[self.sensor.unwrap()][i].cached_output = Some(input);
        }

        // Feed-forward values for hidden and output layers
        for layer in 1..self.node_array.len() {

            for node in 0..self.node_array[layer].len() {

                for prev_node in 0..self.node_array[layer-1].len() {
                    
                    // self.node_array[layer][node].link_vals.push(self.node_array[layer-1][prev_node].cached_output.unwrap());
                    self.node_array[layer][node].link_vals[prev_node] = Some(self.node_array[layer-1][prev_node].cached_output.unwrap());
                    // I think this line needs to be un-commented
                    self.node_array[layer][node].output();
                    if DEBUG { if layer == self.answer.unwrap() { println!("Ran output on answer {:?}", self.node_array[layer][node].cached_output) } }
                }
                self.node_array[layer][node].output();
            }
        }
    }

    /// Analyses the chosen answer node's result.
    /// Also increments sum and count
    fn self_analysis<'b>(&'b self, epochs: &mut Option<f32>, sum: &'b mut f32, count: &'b mut f32, data: &mut Vec<Input>, line: usize) -> Types {

        println!("answer {}", self.answer.unwrap());
        println!("largest index {}", self.largest_node());
        // println!("{:?}", self);
        let brightest_node: &Node = &self.node_array[self.answer.unwrap()][self.largest_node()];
        let brightness: f32 = brightest_node.cached_output.unwrap();

        if !(epochs.is_none()) {
            if epochs.unwrap() % 10.0 == 0.0 {
                println!("\n-------------------------\n");
                println!("Epoch: {:?}", epochs);
                println!("Category: {:?} \nBrightness: {:?}", brightest_node.category.as_ref().unwrap(), brightness);
                if DEBUG {
                    let dimest_node: &Node = &self.node_array[self.answer.unwrap()][self.node_array[self.answer.unwrap()].len()-1-self.largest_node()];
                    println!("Chosen category: {:?} \nDimest Brightness: {:?}", dimest_node.category.as_ref().unwrap(), dimest_node.cached_output.unwrap());
                }
            }
        }

        if DEBUG { println!("Category: {:?} \nBrightness: {:?}", brightest_node.category.as_ref().unwrap(), brightness); }
        if brightest_node.category.as_ref().unwrap().eq(&data[line].answer) { println!("Correct Answer Chosen"); }

        if brightest_node.category.as_ref().unwrap().eq(&data[line].answer) {
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
                self.node_array[HIDDEN][hidden].err_sig = Some(0.0);
                for next_layer in 0..self.node_array[HIDDEN + 1 ].len() {
                    let next_weight = self.node_array[HIDDEN + 1][next_layer].link_weights[hidden];
                    self.node_array[HIDDEN + 1][next_layer].err_sig = match self.node_array[HIDDEN + 1][next_layer].err_sig.is_none() {
                        true => {
                            Some(0.0)
                        }, 
                        false => {
                            self.node_array[HIDDEN + 1][next_layer].err_sig
                        }
                    };
                    // This changes based on the activation function
                    self.node_array[HIDDEN][hidden].err_sig = Some(self.node_array[HIDDEN][hidden].err_sig.unwrap() + (self.node_array[HIDDEN + 1][next_layer].err_sig.unwrap() * next_weight));
                    if DEBUG { 
                        println!("next err sig {:?}", self.node_array[HIDDEN + 1][next_layer].err_sig.unwrap());
                        println!("next weight {:?}", next_weight);
                    }
                }
                let hidden_result = self.node_array[HIDDEN][hidden].cached_output.unwrap();
                let multiplied_value = self.node_array[HIDDEN][hidden].err_sig.unwrap() * (hidden_result) * (1.0 - hidden_result);
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
    /// ## Returns
    /// The name of the model
    /// 
    /// ## Err
    /// ### WriteModelFailed:   Wraps the models name
    /// ### ModelNameAlreadyExists: Wraps the potential model name
    /// ### UnknownError: Wraps error
    ///  
    pub fn write_model(&mut self) -> Result<String, DarjeelingError> {
        
        let mut rng = rand::thread_rng();
        let file_num: u32 = rng.gen();
        let name: String = format!("model{}.darj", file_num);

        match Path::new(&name).try_exists() {

            Ok(false) => {
                let _file: fs::File = fs::File::create(&name).unwrap();
                // for i in 0..self.node_array.len() {
                //     for j in 0..self.node_array[i].len() {
                //         self.node_array[i][j].cached_output = None;
                //     }
                // }
                // let serialized: String = serde_json::to_string(&self).unwrap();
                
                let mut serialized = "".to_string();
                for node in 0..self.node_array[0].len() {
                    for k in 0..self.node_array[0][node].link_weights.len() {
                        if k == self.node_array[0][node].link_weights.len() - 1 {
                            let _ = &serialized.push_str(format!("{}", self.node_array[0][node].link_weights[k]).as_str());
                        } else {
                            let _ = &serialized.push_str(format!("{},", self.node_array[0][node].link_weights[k]).as_str());
                        }
                    }
                }
                for i in 1..self.node_array.len() {
                    let _ = &serialized.push_str("lb\n");
                    for j in 0..self.node_array[i].len() {
                        for k in 0..self.node_array[i][j].link_weights.len() {
                            print!("{}", self.node_array[i][j].link_weights[k]);
                            if k == self.node_array[i][j].link_weights.len() - 1 {
                                let _ = &serialized.push_str(format!("{}", self.node_array[i][j].link_weights[k]).as_str());
                            } else {
                                let _ = &serialized.push_str(format!("{},", self.node_array[i][j].link_weights[k]).as_str());
                            }                        
                        }
                        let _ = &serialized.push_str(format!(";{}", self.node_array[i][j].b_weight.unwrap().to_string()).as_str()); 
                        let _ = &serialized.push_str("\n");
                    }
                }
                serialized.push_str("lb");
                println!("Serialized: {:?}", serialized);
                match fs::write(&name, serialized) {
                    
                    Ok(()) => {
                        println!("Model {:?} Saved", file_num);

                        Ok(name)
                    },

                    Err(_error) => {

                        Err(DarjeelingError::WriteModelFailed(name))
                    }
                }
            },
            Ok(true) => {
                
                Err(DarjeelingError::ModelNameAlreadyExists(name))
            },
            Err(error) => Err(DarjeelingError::UnknownError(error))
        }
    }

    /// Reads a serizalized Neural Network
    /// 
    /// ## Params
    /// - Model Name: The name(or more helpfully the path) of the model to be read
    /// 
    /// ## Returns
    /// A neural network read from a serialized neural network file
    /// 
    /// ## Err
    /// If the file cannnot be read, or if the file does not contain a valid serialized Neural Network
    pub fn read_model<'b>(model_name: String) -> Result<NeuralNetwork, DarjeelingError<'static>> {

        println!("Loading model");
        
        // Err if the file reading fails
        let serialized_net: String = match fs::read_to_string(&model_name) {
            
            Ok(serizalized_net) => serizalized_net,
            Err(error) => return Err(DarjeelingError::ReadModelFailed(model_name.clone(), error))
        };
 
        let mut node_array: Vec<Vec<Node>> = vec![];
        let mut layer: Vec<Node> = vec![];
        for i in serialized_net.lines() {

            if i.trim() == "lb" {
                node_array.push(layer.clone());
                println!("pushed layer {:?}", layer.clone());
                layer = vec![];
                continue;
            }
            let node_data: Vec<&str> = i.trim().split(";").collect();
            let str_weight_array: Vec<&str> = node_data[0].split(",").collect();
            let mut weight_array: Vec<f32> = vec![];
            let b_weight: &str = node_data[1];
            for i in 0..str_weight_array.len() {
                // print!("testing here {}", str_weight_array[i]);
                let val = str_weight_array[i].parse().unwrap();
                weight_array.push(val);
            }
            // print!("{}", b_weight);
            
            let node = Node::new(&weight_array, Some(b_weight.parse().unwrap()) );
            layer.push(node);
            println!("layer: {:?}", layer.clone())
        }
        println!("node array size {}", node_array.len());
        let sensor: Option<usize> = Some(0);
        let answer: Option<usize> = Some(node_array.len() - 1);
        
        let net = NeuralNetwork {
            node_array,
            sensor,
            answer,
            parameters: None
        };
        println!("node array {:?}", net.node_array);

        Ok(net)
    }
}