from transformers import pipeline

# This model is a stable, distilled version for zero-shot classification.
classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli"
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