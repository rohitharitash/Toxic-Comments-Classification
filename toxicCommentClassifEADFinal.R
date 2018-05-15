# Toxic Comment Classification

rm(list = ls())
library(tidyverse)
library(tidytext)
library(ggplus)
library(tm)


#reading training dataset
train <-
  read.csv(
    "train.csv",
    sep = ',',
    header = TRUE,
    stringsAsFactors = FALSE
  )

)
str(train)

train$comment_text <-
  iconv(train$comment_text, 'UTF-8', 'ASCII', sub = "")

#To remove all characters not in the list of alphanumeric and punctuation
train$comment_text <-
  str_replace_all(train$comment_text, "[^[:graph:]]", " ")


#After analyzing the train data, we have 8 columns :
#id - comment id
#comment_text - wikipedia comment
#toxic        : comment clasiification class
#severe_toxic : comment clasiification class
#obscene      : comment clasiification class
#threat       : comment clasiification class
#insult       : comment clasiification class
#identity_hate: comment clasiification class



############ Performing Exploratery data analysis ################################################

# now look at top  common words
visualize_topwords <- function(train) {
  train %>%
    unnest_tokens(word, comment_text) %>% #tokenizing comments
    anti_join(stop_words) %>% #removing stop words
    count(word, sort = TRUE) %>% #counting tokens
    filter(n > 10000) %>% #filtering words greater than 10000
    mutate(word = reorder(word, n)) %>%
    ggplot() +
    geom_col(mapping = aes(x = word, y = n, fill = word),
             show.legend = FALSE) +
    labs(x = 'word', y = 'word count', title = 'Top words from Input text') +
    coord_flip()
}

visualize_topwords(train)

#First step of any text analysis or text mining is tokenization.
#COnverting our text(sentences or comments) into individual words or one-token-per-row format

training_words <- train %>%
  unnest_tokens(word, comment_text)

# counting frequency of a token in each target class
trainWords <- training_words %>%
  count(toxic,
        severe_toxic,
        obscene,
        threat,
        insult,
        identity_hate,
        word) %>%
  ungroup()

# upon examining trainwords, it is observed that a word is present in multiple targetclass.
#So we know that a single comment may belong to more than one target class. So a word can occur
# or present in many categories, which can be a combination of 6 target class.
#Lets find out , how many such unique categories exist.

words_by_cat <- trainWords %>%
  group_by(toxic, severe_toxic, obscene, threat, insult, identity_hate) %>%
  summarise(total = sum(n))
# total 41 such categories exist.

words_by_cat$category = 1:41
str(words_by_cat)

# we have 41 unique combined categories, Adding this information into trainwords.
final_trainwords <- left_join(
  trainWords,
  words_by_cat,
  by = c(
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
  )
)

#removing stop words from final_trainwords
final_trainwords <- final_trainwords %>%
  anti_join(stop_words)

# now our dataset contains token with its count and to which category it belong.
#We will use  tf-idf statistic to measure how important a word is a document. In our dataset,
# we have multiple documents(comments) belongs to many categories ie. toxic,threat,etc.

#calculating tf-idf for each token
final_trainwords <- final_trainwords %>%
  bind_tf_idf(word, category, n)

#Lets examine some important words.ie words with high tf-idf
final_trainwords %>%
  select(word, category, tf_idf) %>%
  arrange(desc(tf_idf))

#These are some of the important words and categories to which they belong
# word         category tf_idf
#<chr>           <int>  <dbl>
#1 bleachanhero       36  0.820
#2 nl33ers            29  0.744
#3 bunksteve          31  0.508
#4 drink              36  0.363
#5 supertr0ll          6  0.286
#6 criminalwar        28  0.243
#7 nl33ersi           29  0.208
#8 faggot             30  0.201
#9 shomron            33  0.161
#10 fucksex            35  0.153

#Let's visualize these important words per unique cartegories
# Due to limitation of hardware and Rstudio top words of 41 unique categories are ploted and stored
#in a pdf file 'important_words_percat.pdf' .
fillColor = "#F1C40F"
gg1 <- final_trainwords %>%
  arrange(desc(tf_idf)) %>%
  group_by(category) %>%
  top_n(15) %>%
  ungroup() %>%
  ggplot() +
  geom_col(mapping = aes(
    x = word,
    y = tf_idf,
    color = category ,
    fill = tf_idf
  )) +
  labs(x = NULL, y = 'tf-idf') +
  # facet_wrap(~category, ncol = 2, scales = "free")
  coord_flip()

pdf('important_words_percatTF_IDF.pdf')
facet_multiple(
  plot = gg1,
  facets = "category",
  ncol = 1,
  nrow = 1
)
dev.off()

# we can visuaalize top important class relevant to target categories
plot_importantwords <- final_trainwords %>%
  arrange(desc(tf_idf))

#1. top tf-idf important words for Toxic
plot_importantwords %>%
  filter(toxic == 1) %>%
  top_n(20) %>%
  ggplot() +
  geom_col(mapping = aes(x = word, y = tf_idf, fill = tf_idf)) +
  labs(x = 'Top words', y = "TF-IDF", title = "Important words from Toxic") +
  coord_flip()

#2. top tf-idf important words for severe_toxic
plot_importantwords %>%
  filter(severe_toxic == 1) %>%
  top_n(20) %>%
  ggplot() +
  geom_col(mapping = aes(x = word, y = tf_idf, fill = tf_idf)) +
  labs(x = "Top words", y = "TF-IDF", title = "Important words from severe_toxic") +
  coord_flip()

#3. Top tf-idf important words for obscene
plot_importantwords %>%
  filter(obscene == 1) %>%
  top_n(20) %>%
  ggplot() +
  geom_col(mapping = aes(x = word, y = tf_idf, fill = tf_idf)) +
  labs(x = "Top obscene Words", y = "TF-IDF", title = " Top Important words for obscene") +
  coord_flip()

#4. Top tf-idf important words for threat
plot_importantwords %>%
  filter(threat == 1) %>%
  top_n(20) %>%
  ggplot() +
  geom_col(mapping = aes(x = word, y = tf_idf, fill = tf_idf)) +
  labs(x = "Top threat words", y = "TF-IDF", title = " Top important words for threat") +
  coord_flip()

#5. Top tf-idf important words for insult
plot_importantwords %>%
  filter(insult == 1) %>%
  top_n(30) %>%
  ggplot() +
  geom_col(mapping = aes(x = word, y = tf_idf, fill = tf_idf)) +
  labs(x = "Top insult words", y = "TF_IDF", title = "Top important words for insult") +
  coord_flip()

#6. Top tf-idf important words for identity_hate
plot_importantwords %>%
  filter(identity_hate == 1) %>%
  top_n(30) %>%
  ggplot() +
  geom_col(mapping = aes(x = word, y = tf_idf, fill = tf_idf)) +
  labs(x = "Top identity_hate words", y = "TF-IDF", title = "Top important words for identity_hate") +
  coord_flip()