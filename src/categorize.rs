use crate::{
    activation::ActivationFunction, bench, dbg_println, error::DarjeelingError, input::Input,
    node::Node, types::Types, utils::RandomIter, DEBUG,
};
use rand::{seq::SliceRandom, thread_rng, Rng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{self, Debug},
    fs,
    path::Path,
};

/// The categorization Neural Network struct
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CatNetwork {
    node_array: Box<[Box<[Node]>]>,
    activation_function: Option<ActivationFunction>,
}

impl CatNetwork {
    /// Constructor function for a categorization neural network
    /// Fills a Neural Network's node_array with empty nodes.
    /// Initializes random starting link and bias weights between -.5 and .5
    ///
    /// ## Params
    /// - Inputs: The number of sensors in the input layer
    /// - Hidden: The number of hidden nodes per hidden layer
    /// - Answer: The number of answer nodes, or possible categories
    /// - Hidden Layers: The number of different hidden layers
    /// - Activation Function: Which activation function is used by the network. This can be changed layer with the [`set_activation_func`](fn@set_activation_func) method.
    ///
    /// ## Examples
    /// ``` rust
    /// use darjeeling::{
    ///     activation::ActivationFunction,
    ///     categorize::CatNetwork
    /// };
    ///
    /// let inputs: usize = 10;
    /// let hidden: usize = 40;
    /// let answer: usize = 2;
    /// let hidden_layers: usize = 1;
    /// let mut net = CatNetwork::new(inputs, hidden, answer, hidden_layers, ActivationFunction::Sigmoid);
    /// ```
    pub fn new(
        input_nodes: usize,
        hidden_nodes: usize,
        answer_nodes: usize,
        hidden_layers: usize,
    ) -> CatNetwork {
        let mut rng = rand::thread_rng();

        let mut node_array: Vec<Box<[Node]>> = vec![];

        let input_row: Box<[Node]> = (0..input_nodes)
            .map(|_| Node::new(vec![1.00; hidden_nodes].into_boxed_slice(), 1.0))
            .collect::<Box<[Node]>>();
        node_array.push(input_row);

        (1..hidden_layers + 1).into_iter().for_each(|_| {
            let hidden_vec: Box<[Node]> = (0..hidden_nodes)
                .into_iter()
                .map(|_| {
                    Node::new(
                        vec![rng.gen_range(-0.5..0.5); answer_nodes].into_boxed_slice(),
                        rng.gen_range(-0.5..0.5),
                    )
                })
                .collect::<Box<[Node]>>();
            node_array.push(hidden_vec);
        });

        // TODO: Make these link weights None or smth
        let answer_row: Box<[Node]> = (0..answer_nodes)
            .into_iter()
            .map(|_| Node::new(vec![0.00; 1].into_boxed_slice(), rng.gen_range(-0.5..0.5)))
            .collect::<Box<[Node]>>();

        node_array.push(answer_row);

        CatNetwork {
            node_array: node_array.into_boxed_slice(),
            activation_function: None,
        }
    }

    /// Trains the neural network model to be able to categorize items in a dataset into given categories
    ///
    /// ## Params
    /// - Data: List of inputs
    /// - Categories: List of Strings, each denoting an answer category.
    /// The number of answer nodes should be the same of the number of categories
    /// - Learning Rate: The modifier that is applied to link weights as they're adjusted.
    /// Try fiddling with this one, but -1.5 - 1.5 is recommended to start.
    /// - Name: The name of the network
    /// - Target Error Percent: The error percent at which the network will be stop training, checked at the begining of each new epoch.
    /// - Write: True of you want to write the model to a file, false otherwise
    ///
    /// ## Returns
    /// The fallible:
    /// - name of the model that this neural network trained(the name parameter with a random u32 appended)
    /// some if write is true, none is write is false
    /// - the error percentage of the last epoch
    /// - the mse of the training
    ///
    /// ## Err
    /// - ### WriteModelFailed
    /// There was a problem when saving the model to a file
    /// - ### ModelNameAlreadyExists
    /// The random model name chosen already exists
    ///
    /// Change the name or retrain
    /// - ### UnknownError
    /// Not sure what happened, but something failed
    ///
    /// Make an issue on the [darjeeling](https://github.com/Ewie21/darjeeling) github page
    ///
    /// Or contact me at elocolburn@comcast.net
    ///
    /// ## Examples
    /// ```ignore
    /// use darjeeling::{
    /// categorize::CatNetwork,
    /// activation::ActivationFunction,
    /// input::Input,
    /// // This file may not be avaliable
    /// // Everything found here will be hyper-specific to your project.
    /// tests::{categories_str_format, xor_file}
    /// };
    ///
    /// // A file containing all possible inputs and correct outputs still needs to be make by you
    /// // 0 0;0
    /// // 0 1;1
    /// // 1 0;1
    /// // 1 1;0
    /// // You also need to write the file input function
    /// // Automatic file reading and formatting function coming soon
    /// let categories: Vec<String> = categories_str_format(vec!["0", "1"]);
    /// let mut data: Vec<Input> = xor_file();
    /// let mut net = CatNetwork::new(2, 2, 2, 1, ActivationFunction::Sigmoid);
    /// let learning_rate = 1.0;
    /// let (model_name, error_percentage, mse) = net.train(&data, categories, learning_rate, "xor", 99.0, true).unwrap();
    /// ```
    pub fn train<'b>(
        &'b mut self,
        data: &Box<[Input]>,
        categories: Box<[Types]>,
        learning_rate: f32,
        name: &str,
        target_err_percent: f32,
        write: bool,
    ) -> Result<(Option<String>, f32, f32), DarjeelingError> {
        let activation_function = match self.activation_function {
            Some(s) => s,
            None => return Err(DarjeelingError::SaveModelFailed),
        };

        let mut epochs = 0.0;
        let mut sum = 0.0;
        let mut count = 0.0;
        let mut err_percent = 0.0;
        let mut mse = 0.0;

        println!("Categorize");
        bench!(self.categorize(categories));

        while err_percent < target_err_percent {
            count = 0.0;
            sum = 0.0;

            let data_iter = RandomIter::new(data);
            for series in data_iter {
                dbg_println!("Training Checkpoint One Passed");

                println!("Assign");
                bench!(self.assign_answers(series));

                println!("Push");
                bench!(self.push_downstream(series, activation_function));

                dbg_println!("Sum: {:?} Count: {:?}", sum, count);

                println!("Analysis");
                bench!(self.self_analysis(
                    &mut Some(epochs),
                    &mut sum,
                    &mut count,
                    series,
                    &mut mse,
                ));

                dbg_println!("Sum: {:?} Count: {:?}", sum, count);

                println!("Backpropogate");
                bench!(self.backpropogate(learning_rate, activation_function));
            }

            // let _old_err_percent = err_percent;
            err_percent = (sum / count) * 100.0;
            epochs += 1.0;
            println!("Epoch: {:?}", epochs);
            println!("Training Accuracy: {:?}", err_percent);
            //if err_percent - old_err_percent < 0.00000001 { break; }
        }
        let mut model_name: Option<String> = None;
        if write {
            match self.write_model(&name) {
                Ok(m_name) => {
                    model_name = Some(m_name);
                }
                Err(err) => return Err(err),
            }
        }

        println!("Training: Finished with accuracy of {:?}/{:?} or {:?} percent after {:?} epochs\nmse: {}", sum, count, err_percent, epochs, mse);

        Ok((model_name, err_percent, mse))
    }

    /// Tests a pretrained model
    pub fn test(
        data: &Box<[Input]>,
        categories: Box<[Types]>,
        model_name: String,
    ) -> Result<Vec<Types>, DarjeelingError> {
        let mut sum = 0.0;
        let mut count = 0.0;
        // let mut category: Option<Types> = None;
        let mut answers: Vec<Types> = vec![];
        let mut mse = 0.0;

        let mut net: CatNetwork = match CatNetwork::read_model(model_name.clone()) {
            Ok(net) => net,
            Err(error) => return Err(DarjeelingError::LoadModelFailed("later")),
        };

        net.node_array
            .last()
            .iter()
            .enumerate()
            .for_each(|(i, node)| {
                node.category = Some(categories[i].clone());
                dbg_println!("{:?}", node.category);
            });

        for line in 0..data.len() {
            dbg_println!("Testing Checkpoint One Passed");
            if data[line].answer.is_some() {
                net.assign_answers(&mut data[line]);
            }
            // Do we actually want to do this?
            net.push_downstream(&mut data, line);
            dbg_println!("Sum: {:?} Count: {:?}", sum, count);
            answers.push(
                net.self_analysis(&mut None, &mut sum, &mut count, &mut data, &mut mse, line)
                    .0
                    .clone(),
            );

            dbg_println!("Sum: {:?} Count: {:?}", sum, count);

            // println!("Correct answer: {:?}", data[line].answer)
        }

        // let _old_err_percent = err_percent;
        let err_percent: f32 = (sum / count) * 100.0;
        mse /= count;
        println!(
            "Testing: Finished with accuracy of {:?}/{:?} or {:?} percent\nMSE: {}",
            sum, count, err_percent, mse
        );

        Ok(answers)
    }

    /// Assigns categories to answer nodes based on a list of given categories
    fn categorize(&mut self, categories: Box<[Types]>) {
        self.node_array
            .last_mut()
            .expect("Network has no answer layer")
            .iter_mut()
            .enumerate()
            .for_each(|(i, node)| node.category = Some(categories[i].clone()));
    }

    fn assign_answers(&mut self, input: &Input) {
        self.node_array
            .last_mut()
            .expect("Network has no answer layer")
            .iter_mut()
            .for_each(|node| {
                node.correct_answer = if node.category == input.answer {
                    Some(1.0)
                } else {
                    Some(0.0)
                }
            });
    }

    /// Passes in data to the sensors, pushs data 'downstream' through the network
    fn push_downstream(&mut self, data: &Input, activation_function: ActivationFunction) {
        // Pass data to the input layer
        if let Some(input_layer) = self.node_array.first_mut() {
            input_layer.iter_mut().enumerate().for_each(|(i, node)| {
                node.cached_output = Some(data.inputs[i]);
            });
        }

        // Push forward hidden and output layers
        for layer_i in 1..self.node_array.len() {
            // Clone the cached outputs from the previous layer
            let prev_cached_outputs: Box<[f32]> = self.node_array[layer_i - 1]
                .iter()
                .map(|node| {
                    node.cached_output
                        .expect("Previous nodes do not have cached outputs")
                })
                .collect();

            let layer = self.node_array[layer_i].iter_mut();

            layer.for_each(|node| {
                prev_cached_outputs
                    .iter()
                    .enumerate()
                    .for_each(|(prev_node_i, prev_output)| {
                        node.link_vals[prev_node_i] = *prev_output;
                    });

                node.output(activation_function);
            });
        }
    }

    /// Analyses the chosen answer node's result.
    /// Also increments sum and count
    fn self_analysis<'b>(
        &'b self,
        epochs: &mut Option<f32>,
        sum: &'b mut f32,
        count: &'b mut f32,
        series: &Input,
        mse: &mut f32,
    ) -> (Types, Option<f32>) {
        dbg_println!("answer {}", self.node_array.len() - 1);
        dbg_println!("largest index {}", self.largest_node());
        dbg_println!("{:?}", self);

        let answer_layer = self.node_array.last().expect("Network has no layer");
        let brightest_node: &Node = &answer_layer[self.largest_node()];
        let brightness: f32 = brightest_node.cached_output.unwrap();

        dbg_println!(
            "Category: {:?} \nBrightness: {:?}",
            brightest_node.category.as_ref().unwrap(),
            brightness
        );

        if series.answer.is_some() {
            if brightest_node.category.eq(&series.answer) {
                dbg_println!("Sum++");
                *sum += 1.0;
            }
            *count += 1.0;
        }

        match epochs {
            Some(epochs) => {
                // This won't happen during testing
                if *epochs % 10.0 != 0.0 || *epochs == 0.0 {
                    return (brightest_node.category.clone().unwrap(), None);
                }
                println!("\n-------------------------\n");
                println!("Epoch: {:?}", epochs);
                println!(
                    "Category: {:?} \nBrightness: {:?}",
                    brightest_node.category.as_ref().unwrap(),
                    brightness
                );
                if DEBUG {
                    let dimest_node: &Node =
                        &answer_layer[answer_layer.len() - self.largest_node() - 1];
                    println!(
                        "Chosen category: {:?} \nDimest Brightness: {:?}",
                        dimest_node.category.as_ref().unwrap(),
                        dimest_node.cached_output.unwrap()
                    );
                }
                (brightest_node.category.clone().unwrap(), None)
            }
            None => (
                brightest_node.category.clone().unwrap(),
                Some(CatNetwork::calculate_err_for_generation_model(
                    mse,
                    brightest_node,
                )),
            ),
        }
    }

    fn calculate_err_for_generation_model(mse: &mut f32, node: &Node) -> f32 {
        *mse += f32::powi(
            node.correct_answer.unwrap() - node.cached_output.unwrap(),
            2,
        );
        *mse
    }

    /// Finds the index and the brightest node in an array and returns it
    fn largest_node(&self) -> usize {
        let mut largest_index = 0;
        let answer_layer = self.node_array.last().expect("Network has no layers");
        for (i, node) in answer_layer.iter().enumerate() {
            if node.cached_output > answer_layer[largest_index].cached_output {
                largest_index = i;
            }
        }

        largest_index
    }
    /// Goes back through the network adjusting the weights of the all the neurons based on their error signal
    fn backpropogate(&mut self, learning_rate: f32, activation_function: ActivationFunction) {
        let hidden_layers = self.node_array.len() - 2;

        self.node_array
            .last_mut()
            .expect("Network has no layers")
            .iter_mut()
            .for_each(|node| {
                node.compute_answer_err_sig(activation_function);

                dbg_println!("Answer Node(Post Error Calc): {:?}", node);
            });

        self.adjust_hidden_weights(learning_rate, hidden_layers);

        self.node_array
            .last_mut()
            .expect("Network has no layers")
            .iter_mut()
            .for_each(|node| node.adjust_weights(learning_rate));
    }

    #[allow(non_snake_case)]
    /// Adjusts the weights of all the hidden neurons in a network
    fn adjust_hidden_weights(&mut self, learning_rate: f32, hidden_layers: usize) {
        for layer in 1..=hidden_layers {
            for node in 0..self.node_array[layer].len() {
                self.node_array[layer][node].err_sig = Some(0.0);
                // Link weights, err sigs
                let mut next_layer: Box<[(f32, f32)]> = self.node_array[layer + 1]
                    .iter()
                    .map(|next_node| {
                        (
                            next_node.link_weights[node],
                            next_node.err_sig.unwrap_or(0.00),
                        )
                    })
                    .collect::<Box<[(f32, f32)]>>();

                next_layer.iter_mut().for_each(|(next_weight, err_sig)| {
                    // This changes based on the activation function
                    self.node_array[layer][node].err_sig =
                        Some(*err_sig + (*err_sig * *next_weight));

                    dbg_println!("next err sig {:?}", err_sig);
                    dbg_println!("next weight {:?}", next_weight);
                });

                let hidden_result = self.node_array[layer][node].cached_output.unwrap();
                let multiplied_value = self.node_array[layer][node].err_sig.unwrap()
                    * (hidden_result)
                    * (1.0 - hidden_result);

                dbg_println!("New hidden errsig multiply: {:?}", multiplied_value);

                self.node_array[layer][node].err_sig = Some(multiplied_value);

                dbg_println!("\nLayer: {:?}", layer);
                dbg_println!("Node: {:?}", node);

                self.node_array[layer][node].adjust_weights(learning_rate);
            }
        }
    }

    /// Serializes a trained model so it can be used later
    ///
    /// ## Returns
    /// The name of the model
    ///
    /// ## Err
    /// ### WriteModelFailed:
    /// Wraps the models name
    /// ### UnknownError:
    /// Wraps error
    ///  
    pub fn write_model(&mut self, name: &str) -> Result<String, DarjeelingError> {
        match self.serialize() {
            Ok(_) => {
                println!("Model: {name} Saved");
            }
            Err(err) => {}
        }
    }
    /// Reads a serizalized Neural Network
    ///
    /// ## Params
    /// - Model Name: The name(or more helpfully the path) of the model to be read
    ///
    /// ## Returns
    /// A neural network read from a serialized .darj file
    ///
    /// ## Err
    /// If the file cannnot be read, or if the file does not contain a valid serialized Neural Network
    pub fn read_model(model_name: String) -> Result<CatNetwork, DarjeelingError> {
        println!("Loading model");
        // Err if the file reading fails
        let serialized_net: String = match fs::read_to_string(&model_name) {
            Ok(serizalized_net) => serizalized_net,
            Err(error) => {
                return Err(DarjeelingError::ReadModelFailed(
                    model_name.clone() + ";" + &error.to_string(),
                ))
            }
        };
        let mut node_array: Vec<Vec<Node>> = vec![];
        let mut layer: Vec<Node> = vec![];
        let mut activation: Option<ActivationFunction> = None;
        for i in serialized_net.lines() {
            match i {
                "sigmoid" => activation = Some(ActivationFunction::Sigmoid),

                "linear" => activation = Some(ActivationFunction::Linear),

                // "tanh" => activation = Some(ActivationFunction::Tanh),

                // "step" => activation = Some(ActivationFunction::Step),
                _ => {
                    if i.trim() == "lb" {
                        node_array.push(layer.clone());
                        // println!("pushed layer {:?}", layer.clone());
                        layer = vec![];
                        continue;
                    }
                    #[allow(unused_mut)]
                    let mut node: Option<Node>;
                    if node_array.len() == 0 {
                        let b_weight: Vec<&str> = i.split(";").collect();
                        // println!("b_weight: {:?}", b_weight);
                        node = Some(Node::new(
                            &vec![],
                            match b_weight[1].parse() {
                                Ok(weight) => Some(weight),
                                Err(err) => {
                                    return Err(DarjeelingError::InvalidNodeValueRead(
                                        err.to_string() + "; Bias: " + b_weight[1],
                                    ))
                                }
                            },
                        ));
                    } else {
                        let node_data: Vec<&str> = i.trim().split(";").collect();
                        let str_weight_array: Vec<&str> = node_data[0].split(",").collect();
                        let mut weight_array: Vec<f32> = vec![];
                        let b_weight: &str = node_data[1];
                        // println!("node_data: {:?}", node_data);
                        // println!("array {:?}", str_weight_array);
                        for weight in 0..str_weight_array.len() {
                            // println!("testing here {:?}", str_weight_array[weight]);
                            let val: f32 = match str_weight_array[weight].parse() {
                                Ok(v) => v,
                                Err(err) => {
                                    return Err(DarjeelingError::InvalidNodeValueRead(
                                        err.to_string() + "; Weight: " + str_weight_array[weight],
                                    ))
                                }
                            };
                            weight_array.push(val);
                        }
                        // print!("{}", b_weight);
                        node = Some(Node::new(
                            &weight_array,
                            match b_weight.parse() {
                                Ok(weight) => Some(weight),
                                Err(err) => {
                                    return Err(DarjeelingError::InvalidNodeValueRead(
                                        err.to_string() + " ;" + b_weight,
                                    ))
                                }
                            },
                        ));
                    }

                    layer.push(node.expect("Both cases provide a Some value for node"));
                    // println!("layer: {:?}", layer.clone())
                }
            }
        }
        //println!("node array size {}", node_array.len());
        let answer: Option<usize> = Some(node_array.len() - 1);

        let net = CatNetwork {
            node_array,
            answer,
            parameters: None,
            activation_function: match activation {
                Some(acti) => acti,
                None => {
                    return Err(DarjeelingError::ActivationFunctionNotRead(format!(
                        "While attempting to read file {}",
                        model_name
                    )))
                }
            },
        };
        // println!("node array {:?}", net.node_array);

        Ok(net)
    }

    pub fn set_activation_func(&mut self, new_activation_function: ActivationFunction) {
        self.activation_function = new_activation_function;
    }

    pub fn add_hidden_layer_with_size(&mut self, size: usize) {
        let mut rng = rand::thread_rng();
        let a = self.answer.expect("initialized network");
        self.node_array.push(self.node_array[a].clone());
        self.node_array[a] = vec![];
        let links = self.node_array[a - 1].len();
        (0..size).into_iter().for_each(|i| {
            self.node_array[a].push(Node::new(&vec![], Some(rng.gen_range(-0.5..0.5))));
            self.node_array[a][i].links = links;
            (0..self.node_array[a][i].links).into_iter().for_each(|_| {
                self.node_array[a][i]
                    .link_weights
                    .push(rng.gen_range(-0.5..0.5));
                self.node_array[a][i].link_vals.push(None);
            })
        });
        self.answer = Some(a + 1);
    }
}

impl fmt::Display for CatNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buff = String::from("");
        self.node_array.iter().for_each(|layer| {
            layer.iter().for_each(|node| {
                buff.push_str(format!("{:?}", node).as_str());
            })
        });
        write!(f, "{}", buff)
    }
}
