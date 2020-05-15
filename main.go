package main

/* TODO
- Batch Normalization and Batch Training or Dropout
- Adjusting Gradient Descenting Method
*/

import (
	"flag"
	"fmt"
	"image"
	"image/png"
	"math/rand"
	"os"
	"time"

	"./aigo"
	"./load"
)

// OneHotVector returns pointer of array
func OneHotVector(o *[]float64, ind int) {
	for i := range *o {
		if i == ind {
			(*o)[i] = 1.0
		} else {
			(*o)[i] = 0.0
		}
	}
}
func main() {
	// Set Random Seed
	rand.Seed(time.Now().UnixNano())

	// Argument Parsing
	_argEpoch := flag.Int("epoch", 4, "Epochs to train")
	_argBatchSize := flag.Int("batchSize", 0, "Batch size to train")
	_argLr := flag.Float64("lr", 0.0001, "Learning Rate") //0.005 works quite not bad, but how about 0.02?
	_argModelName := flag.String("model", "FC1", "Model Conf (FC / Conv)")
	_argLogInterval := flag.Int("logInterval", 3000, "Log interval")
	_argLogSave := flag.Bool("logSave", true, "Log save?")
	flag.Parse()
	argEpoch := *_argEpoch
	argBatchSize := *_argBatchSize
	argLr := float64(*_argLr)
	argModelName := *_argModelName
	argLogInterval := *_argLogInterval
	argLogSave := *_argLogSave

	// Load Data
	fmt.Println("Loading Train Images...")
	trainImageCount, trainImageRow, trainImageCol, trainImageDataRaw := load.ReadImages("train-images.idx3-ubyte")
	trainLabelCount, trainLabelDataRaw := load.ReadLabels("train-labels.idx1-ubyte")
	fmt.Println("Loading Test Images...")
	testImageCount, testImageRow, testImageCol, testImageDataRaw := load.ReadImages("t10k-images.idx3-ubyte")
	testLabelCount, testLabelDataRaw := load.ReadLabels("t10k-labels.idx1-ubyte")
	// Verify Data
	if trainImageCount != trainLabelCount {
		fmt.Println("ERROR: Train image and label size doesn't match.")
		os.Exit(0)
	}
	if testImageCount != testLabelCount {
		fmt.Println("ERROR: Test image and label size doesn't match.")
		os.Exit(0)
	}
	if trainImageRow != testImageRow || trainImageCol != testImageCol {
		fmt.Println("ERROR: Image Dimension Between Test and Train Image doesn't match.")
		os.Exit(0)
	}
	imageDim := [2]int{trainImageRow, trainImageCol}
	fmt.Printf("Image Dimension : ")
	fmt.Println(imageDim)

	// Load Model
	if argModelName != "FC1" && argModelName != "FC2" && argModelName != "conv" {
		fmt.Printf("Unknown Model. Please Specify.")
		os.Exit(0)
	}
	m := &aigo.Network{}
	if argModelName == "FC1" {
		m = aigo.InitModel(imageDim[0]*imageDim[1], 10, argLr)
	} else if argModelName == "FC2" {
		// m = aigo.InitModelFC2(imageDim[0]*imageDim[1], 10, argLr)
	}
	fmt.Println(m.GetNetworkInfo())

	fmt.Println("Normalizing Train Images...")
	// Normalize Image
	testImageData := make([][]float64, testImageCount)
	trainImageData := make([][]float64, trainImageCount)
	for i := range trainImageDataRaw {
		trainImageData[i] = load.Flat(load.Normalize(trainImageDataRaw[i]))
	}
	for i := range testImageDataRaw {
		testImageData[i] = load.Flat(load.Normalize(testImageDataRaw[i]))
	}

	// Adjust Label
	testLabelData := make([]int, testImageCount)
	trainLabelData := make([]int, trainImageCount)
	for ind, val := range trainLabelDataRaw {
		trainLabelData[ind] = int(val)
	}
	for ind, val := range testLabelDataRaw {
		testLabelData[ind] = int(val)
	}

	trainIndex := make([]int, trainImageCount)
	testIndex := make([]int, testImageCount)
	for i := 0; i < len(trainIndex); i++ {
		trainIndex[i] = i
	}
	for i := 0; i < len(testIndex); i++ {
		testIndex[i] = i
	}

	// Log
	var fileLog *os.File
	if argLogSave {
		fileLog, _ = os.Create("./output.txt")
		fileLog.WriteString("Log file Created on " + time.Now().String() + "\n\n")
		fileLog.WriteString(m.GetNetworkInfo() + "\n")
		fileLog.WriteString(fmt.Sprintf("Training MNIST with %s\n", argModelName))
		fileLog.WriteString(fmt.Sprintf("Epoch: %d, Batch Size: %d, lr: %f\n\n", argEpoch, argBatchSize, argLr))
	}

	/// Print Argument
	fmt.Println("------------------------------------------------------")
	fmt.Printf("Training MNIST with %s\n", argModelName)
	fmt.Printf("Epoch: %d, Batch Size: %d, lr: %f\n", argEpoch, argBatchSize, argLr)

	statLoss := make([]float64, trainImageCount/argLogInterval*argEpoch)
	statLossIndex := 0

	fmt.Println("Data Processing Completed. Starting Training")
	// Train
	timeStart := time.Now()
	for epoch := 1; epoch <= argEpoch; epoch++ {
		timeEpoch := time.Now()
		fmt.Printf(" ----- EPOCH %d ----- \n", epoch)
		// Randomize input image sequence
		rand.Shuffle(len(trainIndex), func(i, j int) { trainIndex[i], trainIndex[j] = trainIndex[j], trainIndex[i] })
		// rand.Shuffle(len(testIndex), func(i, j int) { testIndex[i], testIndex[j] = testIndex[j], testIndex[i] })

		// batchCount := 0
		loss := 0.0
		lossCurrent := 0.0
		lossCount := 0
		grad := make([]float64, 10)
		// Do a forward pass until batch size, and get the loss
		for idx := 0; idx < len(trainIndex); idx++ {
			// Forward pass of the image
			_, lossCurrent = m.Forward(&trainImageData[trainIndex[idx]], trainLabelData[trainIndex[idx]])
			// Add loss
			loss += lossCurrent
			lossCount++
			// Get target gradient, and backward propagation of the network
			OneHotVector(&grad, trainLabelData[trainIndex[idx]])
			m.Backward(&grad)

			// Log Interval
			if idx%argLogInterval == argLogInterval-1 {
				elapsedStart := time.Since(timeStart)
				elapsedEpoch := time.Since(timeEpoch)
				loss = loss / float64(lossCount)
				fmt.Printf("[Epoch %2d/%2d %6d/%6d, Loss %f, Time (Epoch)%s (Total)%s]\n", epoch, argEpoch, idx+1, trainImageCount, loss, elapsedEpoch, elapsedStart)
				if argLogSave {
					statLoss[statLossIndex] = loss
					statLossIndex++
				}
				loss = 0.0
				lossCount = 0
			}
		}

		// Test Network
		countCorrect, cM := m.Evaluation(&testImageData, &testLabelData)
		// Print result
		elapsedStart := time.Since(timeStart)
		elapsedEpoch := time.Since(timeEpoch)
		fmt.Printf("[ Test Accuracy %d / %d, Time (Epoch)%s (Total)%s]\n", countCorrect, testImageCount, elapsedEpoch, elapsedStart)
		fmt.Println("Confusion Matrix")
		for i := range cM {
			for j := range cM[i] {
				fmt.Printf("%d\t", cM[i][j])
			}
			fmt.Println()
		}
		// Save log if it is last epoch
		if epoch == argEpoch && argLogSave {
			fileLog.WriteString(fmt.Sprintf("Final Test Accuracy : %d / %d = %f\n\n", countCorrect, testImageCount, float64(countCorrect)/float64(testImageCount)))
			fileLog.WriteString("Confusion Matrix\n")
			for i := range cM {
				for j := range cM[i] {
					fileLog.WriteString(fmt.Sprintf("%d\t", cM[i][j]))
				}
				fileLog.WriteString("\n")
			}
			fileLog.WriteString("\n")
		}
	}

	if argLogSave {
		fmt.Println("Saving Loss table...")
		fileLog.WriteString(fmt.Sprintf("Loss table per %d data\n", argLogInterval))
		for i := range statLoss {
			fileLog.WriteString(fmt.Sprintf("%f\t", statLoss[i]))
		}
		fileLog.WriteString("\n\n")
		// Save top 3 images
		argTopImages := 3
		fmt.Printf("Saving %d top images for class 0...", argTopImages)
		topImages := m.GetTopImages(0, argTopImages, &testImageData, &testLabelData)
		for raw := range topImages {
			s := fmt.Sprintf("./topImage%d.png", raw+1)
			f, _ := os.Create(s)
			img := image.NewGray(image.Rect(0, 0, imageDim[0], imageDim[1]))
			pixels := make([]byte, imageDim[0]*imageDim[1])
			for i := range topImages[raw] {
				pixels[i] = byte(topImages[raw][i] * 256.0)
			}
			img.Pix = pixels
			png.Encode(f, img)
			f.Close()
		}

		elapsedStart := time.Since(timeStart)
		fileLog.WriteString(fmt.Sprintf("Execution Time : %s\n", elapsedStart))
		fileLog.Close()
	}

	fmt.Printf(" ------------------------------ \n Training Completed. Logs saved if argument is true")
}
