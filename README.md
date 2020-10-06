# LT2316 H20 Assignment A1

Name: Kate Viloria  

## Notes on Part 1.
#### Load Dataset and Explore  

__Helper Functions__  
_Get all paths for each xml file._  
```bash
get_subdir(path_to_folder)
get_xmls(path)  

```  
_Extra DDI Data in Test Folder._  
(inside _parse_data)  
After reading some articles online and looking at the current ratio (70/30 train test split) of the data, I decided to go for a 70% train, 15% validation/development, and 15% test split. I split them through the documents instead of the tokens to keep the context of the tokens and entities in tact. I decided to add the documents for the DDI Extraction task in the training data. Then I randomly sampled ~around (used floor division) the same amount of documents that were in the NER test folder and took them from the training set to create the validation set. 

_Create dataframes._
```bash
open_xmls(file_list)
process_text(str)
char_offset(original_string, tokens)
```  
- process_text : I used the WhitespaceTokenizer to separate each word in each sentence given. I chose this tokenizer since it is able to keep contractions, parenthetical phrases, and hyphenated words (ie. 'calcium-rich'). After being tokenized, I iterated through each token to check if the token has punctuation at the beginning or end of the word (ie. 'word.', '(melatonin'). I separated them through WordPunctTokenizer so that the word and the punctuation mark are two separate tokens. I kept the punctuation because I think they are still significant and can provide context as to what is an entity or not. From examining the data, there are times when opening and closing parentheses appear right after an entity for further explanations or dose specifications (ie. sentence id="DDI-MedLine.d21.s6", text="Imipramine (5 mg/kg), moclobemide (30 mg/kg), clonazepam (0.25 mg/kg), fluoxetine (20 mg/kg) sertraline (30 mg/kg) or vehicle was administered. "). Some things I would solve if I had more time were separating tokens with two punctuation marks ('(:'), multi-word expressions (ie. statistics, words with slashes in between ('Drug/Laboratory')).  
- char_offset : I thought it would be more efficient if I created a function that takes the original sentence as a string and the tokenized list as arguments then returns a list of tuples. Each tuple represents a token and within each tuple is the char_start and char_end. Before I start creating the list representing each row in the dataframe, I added an assertion to make sure that the char_list and tokenized list have the same lengths (basically to make sure everything was lining up correctly).  
- open_xmls : Within this function, the two functions above are called. This is also where the token2id dictionary is created. I originally used the gensim import which was great for creating a token2id dictionary. However because of how I designed my function, it was more efficient and worked faster to create the dictionary itself through a counter. I decided to make the dictionary start from 1 since I wanted to designate 0 to be my padding throughout the assignment. This function also creates the ner_df and checks if an entity has two char offsets. If that is the case, it accesses both and creates two tokens with the same entity_id. For both dataframes, I created a list of lists (wherein the initial list, I had already added a list with the column names). This was the method I was most comfortable with and have used many times before in past assignments.  
- ner_df : I included all 4 groups because they still appear in the same contexts regardless if they are referred to by brand, group, a drug, or a drug that isn't approved for humans. I considered all examples to be valuable and more data for the model to learn from. In terms of the task of NER, I think all 4 categories possess similar characteristics in terms of distribution. However, I decided to keep them within their own labels instead of creating a model that would only identify whether it is a drug or not (basically binary). It's important to remember that these entities were grouped in this way for a reason. Although these entity groups share distributions, they also have specific characteristics that apply to their own entity group. For example, the entity type "group" refers to more than one drug and "brand" refers to a specific company's drug. These differentiations are still important to keep in the real world.  I also started this dictionary with 1 in order to leave 0 strictly for padding consistently throughout the assignment.  

_Split NER Distribution_  
For this distribution, I decided a bar graph makes more sense to display this information since the x axis will be divided by labels and by split dataframes (rather than histograms which should be a continuous, quantitative variable). Keeping the counts of each type of token within the split dataframes also gives us a better idea of the data's distribution.  

 
## Notes on Part 2.
#### Extract Features  
_Extra Arguments_  
- id2word - token id to word dictionary for creating the word embeddings and for accessing the string representations of each token id for pos-tagging  
_Features_  
- POS-tagging : uses NLTK POS-tagger to iterate through each sentence and give each word a lexical category. I thought this would be useful since entities are usually nouns or adjectives. Entities are also typically surrounded by words of the same lexical category especially since the corpus is based on talking about the effects of the drugs or the interactions between drugs. Sentences in the corpus have similar formulations since they are quite terse or concise. 
- .istitle() : checks whether or not each string token is capitalised (1 for true, 0 for false) - From exploring the corpus, a lot of the entities are capitalised. Close to all of the entities under the entity group brand are capitalised and capitalised entities are quite common in the 3 other entity groups. I also noticed that there were sentences that would begin with the entity--meaning that they would also be capitalised. I figured this would be a simple but useful feature to add onto the model. 
- Word Embeddings : created tensors through PyTorch that represented the words in the vocabulary - Word embeddings would be useful in addition to POS-tagging. They capture the semantic meaning of a word through the foundations of distributional hypothesis. Creating a model that maps a certain word to a vector in a 'space' based on the words around them would be very useful for NER since entities appear in very similar contexts.  

## Notes on Part Bonus.  
#### Extended Data Exploration  
###### Sample Length Distribution  
To show the counts of the lengths of sentences in the dataframe, I created a list of all the sentences through accesing the dataframe and using .unique. I used that list to create smaller dataframes for each sentence and using pandas to count how many rows (aka how many words) in each sentence and appending that to a final list of sentence lengths. After turning that into a numpy array, I passed that as an argument to create a histogram to matplotlib.  
###### NER Per Sample Distribution  
Similar to the sample length distribution above, I did the same method of making a smaller dataframe for each sentence and counting how many entities there are in the sentence through the ner_df. Due to how I tokenized my data and created my ner_df, entities that have 2 char offsets are counted as two entities in the dataframe. This will slightly skew the information being presented. I decided to also include the bar graph along with the histogram since both graphs display the information quite differently. In the bar graph, we can see which number of entities are more frequent. However in the histogram, we can see a much more fluid data presentation and the general pattern of how many entities there are in a sentence.
###### Venn Diagram  
*I had to pip install pyvenn in order to create the venn diagram.   
I created a list for each drug type and extracted the sentence_id and ner_id columns from ner_df. I zipped the two columns and iterated through this object with a bunch of if statements. Depending on the ner_id, the sentence_id will be appended to its respective list. I created an assert statement that made sure only the existing ner_id's are in the data. I then made each list into a set and used the venn function to create the diagram.  
