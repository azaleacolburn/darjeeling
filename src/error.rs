use core::fmt;
use std::any::TypeId;

#[derive(Debug, Clone)]
pub enum DarjeelingError {
    ColumnDoesNotExist(String),
    RowDoesNotExist(String),
    PointDoesNotExist((String, String)),
    RowAlreadyExists(String),
    ColumnAlreadyExists(String),
    ModelMissingActivationFunction,
    SaveModelFailed(String),
    LoadModelFailed(String),
    InvalidFormatType(TypeId),
    DisinguishingModelError(String),
    SelfAnalysisStringConversion(String),
    RemoveModelFailed(String),
    ActivationFunctionNotRead(String),
    InvalidNodeValueRead(String),

    UnknownError(String),
}

impl<'a> fmt::Display for DarjeelingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DarjeelingError::ColumnDoesNotExist(column) => write!(f, 
                "{:?} isn't a valid column", 
                column
            ),
            DarjeelingError::RowDoesNotExist(row) => write!(f, 
                "{:?} isn't a valid row", 
                row
            ),
            DarjeelingError::PointDoesNotExist((row, column)) => write!(f, 
                "There is no point at row: {:?}, column: {:?}", 
                row, column
            ),
            DarjeelingError::SaveModelFailed(model_name) => write!(f,
                "Unable to read model {:?}, \n Hint: Double check the model_name \n Error Message: {}",
                model_name.split(";").collect::<Vec<&str>>()[0], model_name.split(";").collect::<Vec<&str>>()[1]
            ),
            DarjeelingError::LoadModelFailed(model_name) => write!(f,
                "Unable to write model {:?}, \n Hint: This is probably because the random name already exists, try saving it again",
                model_name.as_str()
            ),
            DarjeelingError::InvalidFormatType(type_id) => write!(f,
                "We couldn't format this value because the type: {:?} wasn't valid",
                type_id
            ),
            DarjeelingError::DisinguishingModelError(err) => write!(f,
                "Error while training distinguishing model. Error: {}",
                err
            ),
            DarjeelingError::SelfAnalysisStringConversion(err) => write!(f,
                "Issue converting neural values to string (u8 limit exceeded). Full string support coming soon: Error message {}",
                err
            ),
            DarjeelingError::RemoveModelFailed(err) => write!(f,
                "Failed to delete unused distinguishing model. Not fatal. Error message: {}",
                err
            ),
            DarjeelingError::ActivationFunctionNotRead(err) => write!(f,
                "Tried to read a .darj file without a valid activation function. Error message: {}",
                err
            ),
            DarjeelingError::InvalidNodeValueRead(err) => write!(f,
                "Tried to parse a value from a .darj file that wasn't a valid f32. Error message: {}",
                err
            ),
            DarjeelingError::ColumnAlreadyExists(label) => write!(f,
                "Attempted to add a column labeled: {}, that already exist in the dataframe",
                label
            ),
            DarjeelingError::RowAlreadyExists(label) => write!(f,
                "Attemtped to add a row labeled: {}, that already exists",
                label
            ),
            DarjeelingError::UnknownError(error) => write!(f,
                "Non-Darjeeling error encountered: \n {:?}",
                error
            ),
            _ => write!(f, "Other error I forgot about")
        }
    }
}
