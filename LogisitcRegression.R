###########################################################
######################### LOGISTIC REGRESSION #############
###########################################################
#####https://www.datacamp.com/tutorial/logistic-regression-R
library(readr)
library(tidymodels)
library(glmnet)

# Read the dataset and convert the target variable to a factor
train_df = read_csv("/Users/leylaciner/Downloads/trainData.csv")
train_df$target = as.factor(train_df$target)

# Split data into train and test
train = train_df
test = read_csv("/Users/leylaciner/Downloads/testData.csv")
test$target = as.factor(test$target)

# Train a logistic regression model
model = logistic_reg(mixture = double(1), penalty = double(1)) %>%
  set_engine("glmnet") %>%
  set_mode("classification") %>%
  fit(target ~ ., data = train)

# Model summary
##Shows coefficients of predictor
tidy(model)

# Class Predictions
pred_class = predict(model,
                      new_data = test,
                      type = "class")

# Class Probabilities
pred_prob = predict(model,
                      new_data = test,
                      type = "prob")

#Evaluate model 
results = test %>%
  select(target, rank) %>%
  bind_cols(pred_class, pred_prob)

#Accuracy
accuracy(results, truth = target, estimate = .pred_class)

# Create confusion matrix
conf_mat(results, truth = target,
         estimate = .pred_class)
#Precision
precision(results, truth = target,
          estimate = .pred_class)
#Recall
recall(results, truth = target,
       estimate = .pred_class)

##How well the features predict top 16; only keep ones w/ abs value > 0.5\
##Higher number = better predictor
coeff = tidy(model) %>% 
  arrange(desc(abs(estimate))) %>% 
  filter(abs(estimate) > 0.5)

#Plot of feature importance
ggplot(coeff, aes(x = term, y = estimate, fill = term)) + geom_col() + coord_flip()
