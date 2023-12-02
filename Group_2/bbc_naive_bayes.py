import math
class Naive_Bayes:
    def __init__(self,laplace_smoothing=1):
        self.ls=laplace_smoothing
        self.prior_probability={}
        self.word_probabilities={}
        
    def fit(self, documents, labels):
        n = len(labels)
        unique_labels = set(labels)

        for label in unique_labels:
            self.prior_probability[label] = (labels.count(label) + self.ls) / n
            self.word_probabilities[label] = {}

        for label in self.prior_probability:
            label_docs = [documents[i] for i in range(n) if labels[i] == label]
            n_docs = len(label_docs)

            for document in label_docs:
                for word, frequency in document.items():
                    self.word_probabilities[label][word] = self.word_probabilities[label].get(word, 0) + frequency

            for word in self.word_probabilities[label]:
                self.word_probabilities[label][word] = (self.word_probabilities[label][word] + self.ls) / (n_docs + self.ls * len(self.word_probabilities[label]))

    def predict(self, documents):
        predictions = [self._predict(document) for document in documents]
        return predictions

    def _predict(self, document):
        label_probabilities = {}
        
        for label in self.prior_probability:
            label_probability = math.log(self.prior_probability[label])
            
            for word, frequency in document.items():
                word_probability = self.word_probabilities[label].get(word, 0)
                if word_probability == 0:
                    word_probability = self.ls / (sum(self.word_probabilities[label].values()) + self.ls * len(self.word_probabilities[label]))
                    
                label_probability += frequency * math.log(word_probability)
            
            label_probabilities[label] = label_probability
        
        return max(label_probabilities, key=label_probabilities.get)
