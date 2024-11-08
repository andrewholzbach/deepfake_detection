from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_model(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    return accuracy, precision, recall
