// Deprecated

use std::{fmt::Debug, collections::hash_map::DefaultHasher, vec};
use serde::Serialize;
use crate::{types::Types, input::Input, error::DarjeelingError};
use std::hash::{Hash, Hasher};

static INITIAL_SIZE: usize = 10;

// TODO: Fix hash overflow issue
#[derive(Debug, Serialize)]
pub struct DataFrame<'a> {
    frame: Vec<Vec<Types>>,
    row_labels: Vec<&'a str>,
    col_labels: Vec<&'a str>,
    inner_capacity: usize
}

// Managing same capacity is difficult
impl<'a> DataFrame<'a> {
    
    /// Creates an empty dataframe with inner capacity of 10
    pub fn new() -> DataFrame<'a> {
        DataFrame { frame: vec![], row_labels: vec![], col_labels: vec![], inner_capacity: 10 }
    }

    /// Creates a new Dataframe
    pub fn from_2d_array(mut frame: Vec<Vec<Types>>, row_labels: Vec<&'a str>, col_labels: Vec<&'a str>) -> DataFrame<'a> {
        let mut inner = 0;
        for i in 0..frame.len() {
            let len = frame[i].len();
            if len > inner { inner = len }
        }
        for i in 0..frame.len() {
            let old = frame[i].capacity();
            frame[i].reserve(inner * 4 - old);
        }
        let outer = frame.len();
        println!("unordered frame: {:?}", frame);
        let mut final_frame = DataFrame { frame, row_labels, col_labels, inner_capacity: inner };
        final_frame.rehash_dataframe(outer * 2);
        final_frame
    }

    /// Gets the value at a hash
    /// # Err
    /// If the column or row doesn't exist
    pub fn value_at_index(&self, row: usize, col: usize) -> Result<&Types, DarjeelingError> {
        if col < self.frame.len() {
            if row < self.frame[col].len() { return Ok(&&self.frame[col as usize][row as usize]); }
            return Err(DarjeelingError::RowDoesNotExist(format!("{}", row)));
        }
        Err(DarjeelingError::ColumnDoesNotExist(format!("{}", col)))
    }
    
    /// Gets the index at labels
    /// # Panics 
    /// If hashing overflows or if labels don't exist
    pub fn index_at_labels(&self, row: &str, col: &str) -> (usize, usize){
        (self.label_index_col(row), self.label_index_row(col))
    }

    /// Gets the value in the dataframe at the coordinates of the labels
    pub fn value_at_labels(&self, row: &str, col: &str) ->  Result<&Types, DarjeelingError> {
        let coords: (usize, usize) = self.index_at_labels(row, col);
        self.value_at_index(coords.0 as usize, coords.1 as usize)
    }

    /// Adds a row to a dataframe
    /// # Panics 
    /// If reallocation fails
    pub fn add_row(&mut self, label: &'a str, content: Vec<Types>) -> Result<(), DarjeelingError> {
        // Checks to make sure the dataframe is still valid
        //assert_eq!(self.row_labels.len(), self.frame.len());
        if self.row_labels.contains(&label) { return Err(DarjeelingError::RowAlreadyExists(format!("{}", label))); }
        if content.len() > self.inner_capacity { self.rehash_dataframe(content.len() * 2) }
        self.row_labels.push(label);
        let row_index = self.label_index_row(label);
        self.frame[row_index] = content;
        Ok(())
    }

    /// Adds a column to a dataframe
    /// # Panics 
    /// If reallocation fails
    pub fn add_col(&mut self, label: &'a str, content: Vec<Types>) -> Result<(), DarjeelingError> {
        // Checks to make sure the dataframe is still valid
        // This was searching for an index of a label to delete instead of
        // adding the index to the end
        println!("inner capacity: {}\nframecapacity: {}", self.inner_capacity, self.frame[0].capacity());
        if self.frame[0].capacity() != self.inner_capacity {
            self.fix_inner_capacity();
        }
        if self.col_labels.contains(&label) { return Err(DarjeelingError::RowAlreadyExists(format!("{}", label))); }
        if content.len() > self.inner_capacity { self.rehash_dataframe(content.len() * 2); }

        self.col_labels.push(label);
        let col_index = self.label_index_col(label);
        for i in 0..self.frame.len() { self.frame[i][col_index] = content[i].clone(); }
        assert_eq!(self.frame[0].capacity(), self.inner_capacity);
        Ok(())
    }

    /// Deletes a row of a dataframe give a label
    /// # Panics 
    /// If the index does not exist
    pub fn delete_row(&mut self, label: &'a str) {
        let index = self.label_index_row(label);
        search_remove(&mut self.row_labels, label);
        self.frame.remove(index);
    }

    /// Deletes a column of a dataframe given a label
    /// # Panics 
    /// If the index does not exist
    pub fn delete_col(&mut self, label: &'a str) {
        let index = self.label_index_col(label);
        search_remove(&mut self.row_labels, label);
        for i in 0..self.frame.len() {
            self.frame[i].remove(index);
        }
    }

    pub fn get_cols_len(&self) -> usize {
        self.col_labels.len()
    }

    pub fn get_rows_len(&self) -> usize {
        self.row_labels.len()
    }

    pub fn display(&mut self) {
        for _i in 0..self.row_labels[0].len() + 2 {
            print!(" ");
        }
        for i in &self.col_labels {
            print!("{:?} ", i)
        }
        print!("\n");
        for row in &self.row_labels {
            // self.fix_inner_capacity();
            let display_row_index = self.label_index_row(row);
            print!("row name: {}, index: {}", row, display_row_index);
            let longest = find_longest(&self.row_labels);
            let shortest = find_shortest(&self.row_labels);
            if longest != row.len() {
                for _l in 0..longest - self.row_labels[0].len() {
                    print!(" ");
                }
            }
            for col in &self.col_labels {
                let display_col_index = self.label_index_col(col);
                self.frame[display_row_index as usize][display_col_index as usize].display();
                if longest != col.len() && shortest != longest{
                    for _k in 0..col.len() + 2 {
                        print!(" ");
                    }
                } else {
                    for _k in 0..col.len() {
                        print!(" ");
                    }
                }
            }
            print!("\n");
        }
    }
    // Refactor to specialize for generative networks
    pub fn frame_to_categorization_inputs(&self, answers: Vec<i32>) -> Result<Vec<Input>, DarjeelingError> {
        let mut inputs: Vec<Input> = vec![];
        for i in 0..self.row_labels.len() {
            let mut input: Input = Input::new(vec![], Some(self.frame[i][answers[i] as usize].clone()));
            for j in 0..self.col_labels.len() - 1 {
                let wrapped = self.value_at_index(i, j);
                match wrapped {
                    Ok(val) => {
                        match val {
                            Types::Boolean(boolean) => {
                                if *boolean {
                                    input.inputs.push(1.0);
                                } else {
                                    input.inputs.push(0.0);
                                }
                            },
                            Types::Float(float) => {
                                input.inputs.push(*float)
                            },
                            Types::Integer(int) => {
                                input.inputs.push(*int as f32);
                            },
                            Types::String(str) => {
                                let ascii = ascii_converter::string_to_decimals(str).unwrap();
                                let mut string = String::new();
                                for i in ascii {
                                    string.push_str(&i.to_string());
                                }
                                input.inputs.push(string.parse().unwrap());
                            }
                        }
                    },
                    Err(err) => return Err(err)
                }
                
            }
            inputs.push(input);
        }
        Ok(inputs)
    }

    pub fn frame_to_generation_inputs(&self) -> Result<Vec<Input>, DarjeelingError> {
        let mut inputs: Vec<Input> = vec![];
        for i in 0..self.row_labels.len() {
            let mut input = Input::new(vec![], None);
            for j in 0..self.col_labels.len() - 1 {
                let wrapped = self.value_at_index(i, j);
                match wrapped {
                    Ok(val) => {
                        match val {
                            Types::Boolean(boolean) => {
                                if *boolean {
                                    input.inputs.push(1.0);
                                } else {
                                    input.inputs.push(0.0);
                                }
                            },
                            Types::Float(float) => {
                                input.inputs.push(*float)
                            },
                            Types::Integer(int) => {
                                input.inputs.push(*int as f32);
                            },
                            Types::String(str) => {
                                let ascii = ascii_converter::string_to_decimals(str).unwrap();
                                let mut string = String::new();
                                for i in ascii {
                                    string.push_str(&i.to_string());
                                }
                                input.inputs.push(string.parse().unwrap());
                            }
                        }
                    },
                    Err(err) => return Err(err)
                }
            }
            inputs.push(input);
        }
        Ok(inputs)
    }

    fn label_index_row(&self, label: &str) -> usize {
        (hash_str(label) % self.frame.capacity() as u64) as usize
    }

    fn label_index_col(&self, label: &str) -> usize {
        (hash_str(label) % self.inner_capacity as u64) as usize
    }

    fn rehash_dataframe(&mut self, new_capacity: usize) {
        // This starts with self.inner_capacity being old
        let mut new_frame: Vec<Vec<Types>> = Vec::with_capacity(self.frame.len() * 2);
        unsafe { new_frame.set_len(self.frame.len()) }
        println!("capacity {}", new_frame.capacity());
        println!("inner capacity: {}", self.inner_capacity);
        self.fix_inner_capacity();
        let old_capacity = self.inner_capacity;
        // Rehashes row
        println!("old_capacity: {}, new_capacity: {}", old_capacity, new_capacity);
        for i in 0..self.row_labels.len() { 
            new_frame[i].reserve(new_capacity - old_capacity); 
            unsafe { new_frame[i].set_len(self.frame[i].len()) }
        }
        println!("rows: {:?}", self.row_labels);
        println!("cols: {:?}", self.col_labels);
        println!("frame: {:?}", self.frame);
        // Rehashes cols
        for row in &self.row_labels {
            for col in &self.col_labels {
                let old_row = self.label_index_row(row);
                let new_row = (hash_str(row) % new_capacity as u64) as usize;
                let old_col = self.label_index_col(col);
                let new_col = (hash_str(col) % new_capacity as u64) as usize;
                println!("old_col: {}, new_col: {}, old_row: {}, new_row: {}", old_col, new_col, old_row, new_row);
                new_frame[new_row][new_col] = self.frame[old_row][old_col].clone(); // i = 2 is the problem
            }
        }
        self.inner_capacity = new_capacity;
        println!("inner capacity: {}\nnew frame capacity: {}", self.inner_capacity, self.frame[0].capacity());
        self.frame = new_frame;
    }

    fn fix_inner_capacity(&mut self) {
        for i in 0..self.frame.len() {
            self.frame[i].reserve(self.inner_capacity);
        }
    }
}

pub fn find_longest(vec: &Vec<&str>) -> usize {
    let mut longest_len: usize = vec[0].len();
    for i in 1..vec.len() {
        let length = vec[i].len();
        if length > longest_len { longest_len = length }
    }
    longest_len
}

pub fn find_shortest(vec: &Vec<&str>) -> usize {
    let mut shortest_len: usize = vec[0].len();
    for i in 1..vec.len() {
        let length = vec[i].len();
        if length < shortest_len { shortest_len = length; }
    }
    shortest_len
}

pub fn search_remove<T>(vec: &mut Vec<T>, val: T)
    where T: PartialEq {
        for i in 0..vec.len() {
            if vec[i] == val {
                vec.remove(i);
                break;
            }
        }
    }

fn hash_str(string: &str) -> u64 {
    let mut s = DefaultHasher::new();
    string.hash(&mut s);
    s.finish()
}
