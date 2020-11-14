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

              //kiểm tra nếu dự đoán khác với nhãn:
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
                
                    // tương tự perceptron
                    actual = trainingLabels[i]
                    prediction = self.classify([data])[0]
                    if actual != prediction:
                        f = data.copy()
                        
                        // tương tự perceptron nhưng tính thêm hệ số tau để lượng tăng giảm trọng số đc chính xác hơn
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
Features gồm các thành phần:
- Số lần gấp khúc
- Tỉ lệ dài:rộng của ảnh
- Số pixel không có giá trị 0
- Số pixel không phải có giá trị 0 ở nửa trên tấm hình
- Số pixel không phải có giá trị 0 ở nửa bên phải tấm hình

## Q5: Behavioral Cloning

1. Lặp lại max_iterations lần.
2. Lặp lại số lần bằng độ dài trainingDatamax_iterations.
3. Lấy từng cặp data và nhãn từ trainingData, trainingLabels.
4. Với mỗi nhãn trong phần tử thứ 2 trong data lấy ra từng cặp feature, value.
5. kiểm tra nếu dự đoán khác với nhãn:
   tăng trọng số cho nhãn 
   giảm trọng số của dự đoán.

## Q6: Pacman Feature Design

Features gồm các thành phần:
- STOP: bằng 0 nếu action là dừng không sẽ là 0.01
- nearest_ghost: khoảng cách mahattan đến ghost gần nhất
- các thành phần (ghost, i) với i trong khoảng từ 0 đến số ma hiện có: 5 chia cho tổng 0.1 với khoảng cách mahattan đến ghost
- các thành phần (capsule, i) với i trong khoảng từ 0 đến số capsule hiện có: 15 chia cho tổng 1 với khoảng cách mahattan đến capsule
- các thành phần (food, i) với i trong khoảng từ 0 đến số thức ăn hiện có:
- capsule_count: Số capsules nhân với 10
- win: bằng 1 nếu thua ngược lại sẽ là 0
- lose: bằng 1 nếu thua ngược lại sẽ là 0
- score: số điểm nhân với 10
