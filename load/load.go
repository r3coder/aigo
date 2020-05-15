package load

import (
	"fmt"
	"os"
)

// Flat : make image flat
func Flat(d [][]float64) []float64 {
	o := make([]float64, len(d)*len(d[0]))
	ind := 0
	for _, v := range d {
		for _, v1 := range v {
			o[ind] = v1
			ind++
		}
	}
	return o
}

// Normalize : normalize data to 0 to 1
func Normalize(d [][]byte) [][]float64 {
	o := make([][]float64, len(d))
	for i1 := range o {
		o[i1] = make([]float64, len(d[0]))
		for i2 := range o[i1] {
			o[i1][i2] = float64(int(d[i1][i2])) / 256.0
		}
	}
	return o
}

// BytesToInt32 : read []byte and convert front 4 characters into MSB int
func BytesToInt32(data []byte) int {
	return int(data[0])*256*256*256 + int(data[1])*256*256 + int(data[2])*256 + int(data[3])
}

// ReadImages : Read Image Data
func ReadImages(loc string) (int, int, int, [][][]byte) {
	f, err := os.Open(loc)
	f.Seek(0, 0)
	// Checking Magic Number
	t := make([]byte, 4)
	_, err = f.Read(t)
	magicNumber := BytesToInt32(t)
	_, err = f.Read(t)
	dataCount := BytesToInt32(t)
	_, err = f.Read(t)
	rowSize := BytesToInt32(t)
	_, err = f.Read(t)
	colSize := BytesToInt32(t)
	if magicNumber != 2051 {
		fmt.Println("Magic Number Not Matching, There's high possibility to code will not work.")
	}
	// Load Data
	data := make([][][]byte, dataCount)
	for i := range data {
		data[i] = make([][]byte, rowSize)
		for j := range data[i] {
			data[i][j] = make([]byte, colSize)
			_, err = f.Read(data[i][j])
		}
	}
	if err != nil {
		fmt.Println(err)
	}
	f.Close()
	return dataCount, rowSize, colSize, data
}

// ReadLabels : Read Label data
func ReadLabels(loc string) (int, []byte) {
	f, err := os.Open(loc)
	f.Seek(0, 0)
	// Checking Magic Number
	t := make([]byte, 4)
	_, err = f.Read(t)
	magicNumber := BytesToInt32(t)
	_, err = f.Read(t)
	dataCount := BytesToInt32(t)
	if magicNumber != 2049 {
		fmt.Println("Magic Number Not Matching, There's high possibility to code will not work.")
	}
	// Load Data
	data := make([]byte, dataCount)
	_, err = f.Read(data)
	if err != nil {
		fmt.Println(err)
	}
	f.Close()
	return dataCount, data
}
