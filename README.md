# Text-Classification-of-Amazon-Reviews
This is a Natural Language Processing class project that focuses on text categorization and sentiment analysis in Amazon reviews.

-Read the content of the `AmazonReviews.csv` into a dataframe called `reviews_df`
-Preprocess the `reviews_df` dataframe to prepare it for the following questions. Your preprocessing must include the following:**
  -Add a new column to the dataframe called `sentiment` calculated based on the `Score` field values as follows: If the Score value is less than 3 then the sentimnet is 0 'negative` and if the Score value is greater than or equal to 3 then the sentimnet is 1 `positive`
-Clean the `Text` field by performing the following:
        -make text lowercase
        -remove text in square brackets,
        -remove links,
        -remove punctuation and 
        -remove words containing numbers.
  -Add a new column to the dataframe called `token_list` and store a list of the individual words in the `Text` field
    
  -Remove the stopwords from the `token_list` column and store the results in a column called `no_stop_tokens`
    
  -Lemmatize the tokens in the `no_stop_tokens` field and store the results in a column called `Lemmatized_tokens`
    
  -Create a bag of words of all lemmatized words in the dataframe, then find the top 10000 most common words and store them in a list called `top_words_list`
    
  -Create a dictionary called `word_dict` using the following function: (Note: this function takes the `top_words_list` created in the previous step)
    ```python
    def create_word_dict(top_words_list):
        word_dict = {}
        word_dict["<PAD>"] = 0
        word_dict["<START>"] = 1
        word_dict["<UNK>"] = 2
        word_dict["UNUSED"] = 3

        for i, word in enumerate(top_words_list):
            word_dict[word] = i+4

        return word_dict
    ```
    
  -Add a new column to the dataframe called `word_ids` calculated using the following function: (Note: this function takes the `Lemmatized_tokens` field from the dataframe. Therefore, you must use an apply function)
    ```python
    def word_to_id(words):
        return [1] + [word_dict[w] if w in word_dict.keys() else word_dict['<UNK>'] for w in words]
    ```
    
-Extract the `word_ids` and `sentiment` fields into a new dataframe called `dataset`. You are going to use this dataset for the following questions. 
    
-Build a Feed Forward Model to predict the sentiment of each review. Then, evaluate the performance of the model in terms of loss and accuracy

-Build a Convolutional Model to predict the sentiment of each review. Then, evaluate the performance of the model in terms of loss and accuracy
