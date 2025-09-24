from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

def classify_topic(topic, labels):
    """
    Classifies a given topic and returns the top label and its score.
    
    Returns:
        tuple: A tuple containing the label (str) and the score (float).
    """
    result = classifier(topic, labels, multi_label=False)
    
    top_label = result['labels'][0]
    top_score = result['scores'][0]
    
    return top_label, top_score

# --- Example of how to use it ---
if __name__ == '__main__':
    my_topic = "The rules of cricket"
    my_labels = ["Science", "History", "Technology", "Art", "Health", "Finance"]
    
    predicted_category, score = classify_topic(my_topic, my_labels)
    
    print(f"The topic '{my_topic}' was classified as: {predicted_category}")
    print(f"Confidence Score: {score:.2f}")