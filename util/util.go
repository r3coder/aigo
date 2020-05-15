package util

// GetMinValInd Get Min value of the float list
func GetMinValInd(l []float64) (int, float64) {
	i := 0
	v := 0.0
	for ii, vv := range l {
		if ii == 0 || vv < v {
			i = ii
			v = vv
		}
	}
	return i, v
}

// GetMaxValInd Get Min value of the float list
func GetMaxValInd(l []float64) (int, float64) {
	i := 0
	v := 0.0
	for ii, vv := range l {
		if ii == 0 || vv > v {
			i = ii
			v = vv
		}
	}
	return i, v
}
