# ai-classification
ai classification project 2020

## Q1: Perceptron
        
      for iteration in range(self.max_iterations):
          print "Starting iteration ", iteration, "..."
          for i, data in enumerate(trainingData):
              // gán nhãn
              actual = trainingLabels[i]

              // dự đoán dữ liệu
              prediction = self.classify([data])[0]

              //kiểm tra nếu dự đoán khác với nhãn không
              // tăng trọng số cho nhãn 
              // giảm trọng số của dự đoán
              if actual != prediction:
                  self.weights[actual] = self.weights[actual] + data
                  self.weights[prediction] = self.weights[prediction] - data


## Q2: Perceptron Analysis

    def findHighWeightFeatures(self, label):
            """
            Returns a list of the 100 features with the greatest weight for some label
            """
            // tìm ra các feature có trọng số lớn nhất
            return self.weights[label].sortedKeys()[:100]

## Q3: MIRA

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
         
        
        bestWeights = None
        bestCorrect = 0.0
        weights = self.weights.copy()
        for c in Cgrid:
            self.weights = weights.copy()
            for n in range(self.max_iterations):
                for i, data in enumerate(trainingData):
                    actual = trainingLabels[i]
                    prediction = self.classify([data])[0]
                    if actual != prediction:
                        f = data.copy()
                        tau = min(c, ((self.weights[prediction] - self.weights[actual]) * f + 1.0) / (2.0 * (f * f)))
                        f.divideAll(1.0 / tau)
                        self.weights[actual] = self.weights[actual] + f
                        self.weights[prediction] = self.weights[prediction] - f

            correct = 0
            guesses = self.classify(validationData)
            for i, guess in enumerate(guesses):
                correct = correct + (validationLabels[i] == guess and 1.0 or 0.0)

            if correct > bestCorrect:
                bestCorrect = correct
                bestWeights = self.weights

        self.weights = bestWeights


## Q4: Digit Feature Design



## Q5: Behavioral Cloning



## Q6: Pacman Feature Design


