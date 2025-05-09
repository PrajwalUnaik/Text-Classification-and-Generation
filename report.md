
# Reflection on My Modeling Pipeline

## Introduction

As with most projects, the first step was cleaning the transcript data. This involved tokenizing the text, removing stopwords, and lemmatizing words to ensure the models worked with clean, meaningful inputs. Initially, I thought lemmatization and stemming wouldn’t make a significant difference, but after testing, I found that dropping lemmatization actually improved accuracy, which was an interesting insight.

## Approach

Following data cleaning, I applied two approaches: traditional machine learning using Logistic Regression and TF-IDF, and a more advanced Transformer-based model (RoBERTa) to classify the transcripts. As expected, the Transformer model yielded much better results. However, I did notice that with more complex inputs, the model still struggled. I tried fine-tuning the logistic regression model and managed to push its accuracy to around 35%, but eventually, I returned to the Transformer model as it was yielding much higher accuracy.

After some research and watching several YouTube videos on BERT, I decided to experiment with two models: DistilBERT and RoBERTa. Through testing and fine-tuning, RoBERTa consistently outperformed DistilBERT, so I settled with RoBERTa and achieved a satisfying accuracy of 0.65. To double-check, I submitted the model to Codabase and confirmed the results.

I continued tweaking the training process by analyzing the loss plots and decided to settle on 6 epochs and the AdamW optimizer. After reaching a satisfactory accuracy and F1 score, I also plotted the confusion matrix for a better understanding of model performance. Finally, I implemented oversampling and used class weights to handle the class imbalance, which further improved the model's performance.

## Challenges Faced

One of the biggest challenges I ran into was dealing with inconsistencies in the data. The transcripts varied a lot in both style and content, which made it tough to clean everything in a consistent way. On top of that, I was working with a limited amount of labeled data, so I had to be really careful about how I trained the models. Fine-tuning the RoBERTa model definitely helped improve performance, but it was a slow process, especially when working with large datasets and the heavy computational load. Thankfully, having access to the **data lab servers** and using GPUs made a huge difference, though I did spend a fair amount of time troubleshooting CUDA errors along the way.

Another obstacle was the class imbalance in the dataset. To address that, I turned to oversampling to help balance the classes and prevent the model from favoring the larger ones. Still, I know there’s room to improve, and having more data would definitely help the model generalize better in the future if opportunity permits.

## Areas for Improvement

If I had more time and resources, I’d definitely fine-tune the models on a larger dataset. This would likely improve their ability to handle edge cases and make them more robust overall. Another thing that would really help is using cloud computing for faster training, which would allow me to experiment with different ideas and iterate more quickly. Additionally, gaining a deeper understanding of Hugging Face and how to host fine-tuned models would give me more flexibility and control over the process.

That said, I did manage to host the app on the cloud, which was a great step forward. You can find the link to both my GitHub repository and the live app below, as well as at the start of this report.

## Links

You can check out the live app here: [Live App URL](https://text-classification-and-generationgit-g36pp9pnnqscjkfpjb95vx.streamlit.app/)

GitHub: [https://github.com/PrajwalUnaik/Text-Classification-and-Generation](https://github.com/PrajwalUnaik/Text-Classification-and-Generation)

