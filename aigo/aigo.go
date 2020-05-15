package aigo

import (
	"fmt"
	"math"
	"math/rand"

	"../util"
)

// LeakyReLU Struct N - N
type LeakyReLU struct {
	dimension   []int
	nodes       []float64
	gradients   []float64
	coefficient float64
}

// Forward Function of ReLU (Actually ReLU6)
func (l *LeakyReLU) Forward(i *[]float64) {
	for idx, v := range *i {
		if v < 0 {
			l.nodes[idx] = v * l.coefficient
		} else if v > 6 {
			l.nodes[idx] = 6.0
		} else {
			l.nodes[idx] = v
		}
	}
}

// UpdateGradient of the ReLU
func (l *LeakyReLU) UpdateGradient(weight *[][]float64, gradNode *[]float64) {
	for idx := range l.gradients {
		l.gradients[idx] = 0
		for i2 := range *gradNode {
			l.gradients[idx] += (*gradNode)[i2] * (*weight)[idx][i2]
		}
	}
}

// Init Function of LinearLeakyReLU
func (l *LeakyReLU) Init(dim []int) {
	l.dimension = dim
	l.nodes = make([]float64, l.dimension[0])
	l.gradients = make([]float64, l.dimension[0])
	for idx := range l.nodes {
		l.nodes[idx] = 0.0
		l.gradients[idx] = 0.0
	}
	l.coefficient = 0.1
}

// GetNodesPtr Function of LeakyReLU
func (l *LeakyReLU) GetNodesPtr() *[]float64 {
	return &l.nodes
}

// GetGradientsPtr Function of LeakyReLU
func (l *LeakyReLU) GetGradientsPtr() *[]float64 {
	return &l.gradients
}

// GetNetworkInfo Function of LeakyReLU
func (l *LeakyReLU) GetNetworkInfo() string {
	return fmt.Sprintf("LeakyReLU [%d]", l.dimension[0])
}

// ReLU Struct N - N
type ReLU struct {
	dimension []int
	nodes     []float64
	gradients []float64
}

// Forward Function of ReLU (Actually ReLU6)
func (l *ReLU) Forward(i *[]float64) {
	for idx, v := range *i {
		if v < 0 {
			l.nodes[idx] = 0.0
		} else if v > 6 {
			l.nodes[idx] = 6.0
		} else {
			l.nodes[idx] = v
		}
	}
}

// UpdateGradient of the ReLU
func (l *ReLU) UpdateGradient(weight *[][]float64, gradNode *[]float64) {
	for idx := range l.gradients {
		l.gradients[idx] = 0
		for i2 := range *gradNode {
			l.gradients[idx] += (*gradNode)[i2] * (*weight)[idx][i2]
		}
	}
}

// Init Function of ReLU
func (l *ReLU) Init(dim []int) {
	l.dimension = dim
	l.nodes = make([]float64, l.dimension[0])
	l.gradients = make([]float64, l.dimension[0])
	for idx := range l.nodes {
		l.nodes[idx] = 0.0
		l.gradients[idx] = 0.0
	}
}

// GetNodesPtr Function of ReLU
func (l *ReLU) GetNodesPtr() *[]float64 {
	return &l.nodes
}

// GetGradientsPtr Function of ReLU
func (l *ReLU) GetGradientsPtr() *[]float64 {
	return &l.gradients
}

// GetNetworkInfo Function of ReLU
func (l *ReLU) GetNetworkInfo() string {
	return fmt.Sprintf("ReLU [%d]", l.dimension[0])
}

// Softmax Struct N - N
type Softmax struct {
	dimension []int
	nodes     []float64
	gradients []float64
}

// Forward Function of Softmax
func (l *Softmax) Forward(i *[]float64) {
	m := (*i)[0]
	for _, v := range (*i)[1:] {
		if v > m {
			m = v
		}
	}
	for idx := range l.nodes {
		l.nodes[idx] = (*i)[idx] - m
	}
	o := 0.0
	for i := range l.nodes {
		o += math.Exp(l.nodes[i])
	}
	for i := range l.nodes {
		l.nodes[i] = math.Exp(l.nodes[i]) / o
	}
}

// SetGradient of the Softmax
func (l *Softmax) SetGradient(grad *[]float64) {
	for i := range l.gradients {
		l.gradients[i] = (*grad)[i]
	}
}

// Init Function of Softmax
func (l *Softmax) Init(dim []int) {
	l.dimension = dim
	l.nodes = make([]float64, l.dimension[0])
	l.gradients = make([]float64, l.dimension[0])
	for idx := range l.nodes {
		l.nodes[idx] = 0.0
		l.gradients[idx] = 0.0
	}
}

// GetNodesPtr Function of Softmax
func (l *Softmax) GetNodesPtr() *[]float64 {
	return &l.nodes
}

// GetGradientsPtr Function of Softmax
func (l *Softmax) GetGradientsPtr() *[]float64 {
	return &l.gradients
}

// GetNetworkInfo Function of Softmax
func (l *Softmax) GetNetworkInfo() string {
	return fmt.Sprintf("Softmax [%d]", l.dimension[0])
}

// Linear Struct
type Linear struct {
	dimension  []int
	parameters [][]float64
	nodes      []float64
	gradients  [][]float64
	lr         float64
}

// Forward Function of Linear
func (l *Linear) Forward(i *[]float64) {
	var oo float64 = 0.0
	for i1 := 0; i1 < l.dimension[1]; i1++ {
		oo = 0.0
		for i2 := 0; i2 < l.dimension[0]; i2++ {
			oo += l.parameters[i2][i1] * (*i)[i2]
		}
		l.nodes[i1] = oo
	}
}

// Backward of the Linear
func (l *Linear) Backward() {
	for i1 := 0; i1 < l.dimension[0]; i1++ {
		for i2 := 0; i2 < l.dimension[1]; i2++ {
			l.parameters[i1][i2] -= l.gradients[i1][i2] * l.lr
			l.gradients[i1][i2] = 0.0
		}
	}
}

// AddGradient of the Linear
func (l *Linear) AddGradient(grad *[]float64, prevNode *[]float64) {
	for i1 := 0; i1 < l.dimension[0]; i1++ {
		for i2 := 0; i2 < l.dimension[1]; i2++ {
			l.gradients[i1][i2] += (*grad)[i2] * (*prevNode)[i1]
		}
	}
}

// Init Function of Linear
func (l *Linear) Init(dim []int, lr float64) {
	l.dimension = dim
	l.parameters = make([][]float64, l.dimension[0])
	l.gradients = make([][]float64, l.dimension[0])
	for idx := range l.parameters {
		l.parameters[idx] = make([]float64, l.dimension[1])
		l.gradients[idx] = make([]float64, l.dimension[1])
	}
	for i1 := range l.parameters {
		for i2 := range l.parameters[i1] {
			l.parameters[i1][i2] = rand.Float64()*0.2 - 0.1
			l.gradients[i1][i2] = 0.0
		}
	}
	l.nodes = make([]float64, l.dimension[1])
	for idx := range l.nodes {
		l.nodes[idx] = 0.0
	}
	l.lr = lr
}

// GetNodesPtr Function of Linear
func (l *Linear) GetNodesPtr() *[]float64 {
	return &l.nodes
}

// GetGradientsPtr Function of Linear
func (l *Linear) GetGradientsPtr() *[]float64 {
	return &l.gradients[0] // Fix
}

// GetNetworkInfo Function of Linear
func (l *Linear) GetNetworkInfo() string {
	return fmt.Sprintf("Linear [%dx%d]", l.dimension[0], l.dimension[1])
}

// CrossEntropyLoss value of the layer
func CrossEntropyLoss(cls *[]float64, label int) float64 {
	return -math.Log((*cls)[label])
}

/*
// Network Struct - FC1
type Network struct {
	inp []float64
	l1  Linear
	a1  ReLU
	l2  Linear
	a2  ReLU
	l3  Linear
	a3  ReLU
	l4  Linear
	a4  ReLU
	l5  Linear
	a5  Softmax
}*/

// Network Struct - FC2
type Network struct {
	inp []float64
	l1  Linear
	a1  LeakyReLU
	l2  Linear
	a2  LeakyReLU
	l3  Linear
	a3  LeakyReLU
	l4  Linear
	a4  LeakyReLU
	l5  Linear
	a5  Softmax
}

// Forward Pass for Network
func (n Network) Forward(data *[]float64, label int) ([]float64, float64) {
	for i := range *data {
		n.inp[i] = (*data)[i]
	}
	n.l1.Forward(data)
	n.a1.Forward(&n.l1.nodes)
	n.l2.Forward(&n.a1.nodes)
	n.a2.Forward(&n.l2.nodes)
	n.l3.Forward(&n.a2.nodes)
	n.a3.Forward(&n.l3.nodes)
	n.l4.Forward(&n.a3.nodes)
	n.a4.Forward(&n.l4.nodes)
	n.l5.Forward(&n.a4.nodes)
	n.a5.Forward(&n.l5.nodes)
	return n.a5.nodes, CrossEntropyLoss(&n.a5.nodes, label)
}

// Backward Propagation for Network
func (n Network) Backward(err *[]float64) {
	n.a5.SetGradient(&n.a5.nodes)
	for idx := range *err {
		n.a5.gradients[idx] -= (*err)[idx]
	}
	n.l5.AddGradient(&n.a5.gradients, &n.a4.nodes)
	n.l5.Backward()
	n.a4.UpdateGradient(&n.l5.parameters, &n.a5.gradients)
	n.l4.AddGradient(&n.a4.gradients, &n.a3.nodes)
	n.l4.Backward()
	n.a3.UpdateGradient(&n.l4.parameters, &n.a4.gradients)
	n.l3.AddGradient(&n.a3.gradients, &n.a2.nodes)
	n.l3.Backward()
	n.a2.UpdateGradient(&n.l3.parameters, &n.a3.gradients)
	n.l2.AddGradient(&n.a2.gradients, &n.a1.nodes)
	n.l2.Backward()
	n.a1.UpdateGradient(&n.l2.parameters, &n.a2.gradients)
	n.l1.AddGradient(&n.a1.gradients, &n.inp)
	n.l1.Backward()
}

// GetNetworkInfo : Get information of the layers as string
func (n Network) GetNetworkInfo() string {
	s := "Network Information\n"
	s += fmt.Sprintf(" 0 : %s\n", n.l1.GetNetworkInfo())
	s += fmt.Sprintf(" 1 : %s\n", n.a1.GetNetworkInfo())
	s += fmt.Sprintf(" 2 : %s\n", n.l2.GetNetworkInfo())
	s += fmt.Sprintf(" 3 : %s\n", n.a2.GetNetworkInfo())
	s += fmt.Sprintf(" 4 : %s\n", n.l3.GetNetworkInfo())
	s += fmt.Sprintf(" 5 : %s\n", n.a3.GetNetworkInfo())
	s += fmt.Sprintf(" 6 : %s\n", n.l4.GetNetworkInfo())
	s += fmt.Sprintf(" 7 : %s\n", n.a4.GetNetworkInfo())
	s += fmt.Sprintf(" 8 : %s\n", n.l5.GetNetworkInfo())
	s += fmt.Sprintf(" 9 : %s\n", n.a5.GetNetworkInfo())
	return s
}

// Evaluation of the layer, return correct count and confusion matrix
func (n Network) Evaluation(image *[][]float64, label *[]int) (int, [][]int) {

	confusionMatrix := make([][]int, 10)
	for i := range confusionMatrix {
		confusionMatrix[i] = make([]int, 10)
	}
	ansVal := 0.0
	ansIdx := 0
	result := make([]float64, 10)
	for idx := 0; idx < len(*image); idx++ {
		// Evaluation of the network, change this to Evaluation() if you want to use batch training
		result, _ = n.Forward(&(*image)[idx], (*label)[idx])
		ansVal = 0.0
		ansIdx = 0
		for i, v := range result {
			if v > ansVal {
				ansVal = v
				ansIdx = i
			}
		}
		confusionMatrix[ansIdx][(*label)[idx]]++
	}
	countCorrect := 0
	for i := range confusionMatrix {
		countCorrect += confusionMatrix[i][i]
	}
	return countCorrect, confusionMatrix
}

// GetTopImages return array of top images
func (n Network) GetTopImages(label int, count int, dataImage *[][]float64, dataLabel *[]int) [][]float64 {

	top0ind := make([]int, count)
	top0val := make([]float64, count)
	for i := range top0val {
		top0val[i] = 0.0
		top0ind[i] = 0
	}
	topImages := make([][]float64, count)
	for i := range topImages {
		topImages[i] = make([]float64, 28*28)
	}

	ansVal := 0.0
	ansIdx := 0
	result := make([]float64, 10)

	for idx := 0; idx < len(*dataImage); idx++ {
		// Evaluation of the network, change this to Evaluation() if you want to use batch training
		result, _ = n.Forward(&(*dataImage)[idx], (*dataLabel)[idx])
		ansVal = 0.0
		ansIdx = 0
		ansIdx, ansVal = util.GetMaxValInd(result)
		if ansIdx == label {
			mI, mV := util.GetMinValInd(top0val)
			if mV < ansVal {
				top0ind[mI] = idx
				top0val[mI] = ansVal
			}
		}
	}
	for idx := range top0ind {
		topImages[idx] = (*dataImage)[top0ind[idx]]
	}
	return topImages
}

// InitModel 784-512-512-256-10
func InitModel(inputSize int, outputSize int, lr float64) *Network {
	var n Network
	n.inp = make([]float64, inputSize)
	n.l1.Init([]int{inputSize, 512}, lr)
	n.a1.Init([]int{512})
	n.l2.Init([]int{512, 512}, lr)
	n.a2.Init([]int{512})
	n.l3.Init([]int{512, 256}, lr)
	n.a3.Init([]int{256})
	n.l4.Init([]int{256, 128}, lr)
	n.a4.Init([]int{128})
	n.l5.Init([]int{128, 10}, lr)
	n.a5.Init([]int{10})
	return &n
}
