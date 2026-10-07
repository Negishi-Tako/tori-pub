package evaluation

import (
	"errors"
	"math"
)

// bipartiteMatch は Riesen & Bunke (2009) の二部グラフ近似で、ノードの最適割当を求める。
//
// (n+m)×(n+m) のコスト行列を組む:
//
//	左上 n×m  置換コスト sub[i][j]
//	右上 n×n  対角に削除コスト del[i]、それ以外は禁止
//	左下 m×m  対角に挿入コスト ins[j]、それ以外は禁止
//	右下 m×n  ダミー同士（0）
//
// 戻り値は置換として割り当てられた (i → j) の対応。削除・挿入されたノードは含まない。
// 以前は同じ行列を Python（scipy）のサービスで解いていた。解法を scipy と同じにしてあるので
// （linearSumAssignment を参照）、同点の解き方まで含めて同じ割当が返る。
func bipartiteMatch(sub [][]float64, del, ins []float64) (map[int]int, error) {
	n, m := len(del), len(ins)
	if n == 0 && m == 0 {
		return map[int]int{}, nil
	}
	const forbidden = 1e9
	size := n + m
	cost := make([][]float64, size)
	for i := range cost {
		row := make([]float64, size)
		for j := range row {
			row[j] = forbidden
		}
		cost[i] = row
	}
	for i := range n {
		copy(cost[i][:m], sub[i])
		cost[i][m+i] = del[i]
	}
	for j := range m {
		cost[n+j][j] = ins[j]
		for k := m; k < size; k++ {
			cost[n+j][k] = 0
		}
	}
	col4row, err := linearSumAssignment(cost)
	if err != nil {
		return nil, err
	}
	out := make(map[int]int, min(n, m))
	for i := range n {
		if j := col4row[i]; j < m {
			out[i] = j
		}
	}
	return out, nil
}

// linearSumAssignment は正方コスト行列の最小コスト完全割当を解き、行 i に割り当てた列を返す。
//
// scipy.optimize.linear_sum_assignment（rectangular_lsap.cpp）の移植で、
// Crouse (2016) の最短増加路法。同点時の選び方（残り列を逆順に並べる・未割当の列を
// 優先する）まで scipy と同じにしてある。浮動小数の演算も同じ順序で行うので、
// scipy と同じ行列からは同じ割当が返る。
func linearSumAssignment(cost [][]float64) ([]int, error) {
	n := len(cost)
	for _, row := range cost {
		if len(row) != n {
			return nil, errors.New("evaluation: 割当のコスト行列が正方でない")
		}
		for _, c := range row {
			if math.IsNaN(c) || math.IsInf(c, -1) {
				return nil, errors.New("evaluation: 割当のコスト行列に NaN か -Inf がある")
			}
		}
	}
	u := make([]float64, n)
	v := make([]float64, n)
	shortest := make([]float64, n)
	path := make([]int, n)
	col4row := make([]int, n)
	row4col := make([]int, n)
	sr := make([]bool, n)
	sc := make([]bool, n)
	remaining := make([]int, n)
	for i := range n {
		path[i], col4row[i], row4col[i] = -1, -1, -1
	}

	for cur := range n {
		// ---- 最短増加路を探す ----
		minVal := 0.0
		for it := range n {
			remaining[it] = n - it - 1
		}
		numRemaining := n
		clear(sr)
		clear(sc)
		for j := range shortest {
			shortest[j] = math.Inf(1)
		}
		sink, i := -1, cur
		for sink == -1 {
			index, lowest := -1, math.Inf(1)
			sr[i] = true
			for it := range numRemaining {
				j := remaining[it]
				r := minVal + cost[i][j] - u[i] - v[j]
				if r < shortest[j] {
					path[j] = i
					shortest[j] = r
				}
				if shortest[j] < lowest || (shortest[j] == lowest && row4col[j] == -1) {
					lowest = shortest[j]
					index = it
				}
			}
			minVal = lowest
			if math.IsInf(minVal, 1) {
				return nil, errors.New("evaluation: 割当が存在しない")
			}
			j := remaining[index]
			if row4col[j] == -1 {
				sink = j
			} else {
				i = row4col[j]
			}
			sc[j] = true
			numRemaining--
			remaining[index] = remaining[numRemaining]
		}

		// ---- 双対変数を更新する ----
		u[cur] += minVal
		for i := range n {
			if sr[i] && i != cur {
				u[i] += minVal - shortest[col4row[i]]
			}
		}
		for j := range n {
			if sc[j] {
				v[j] -= minVal - shortest[j]
			}
		}

		// ---- 解を増加路に沿って更新する ----
		for j := sink; ; {
			i := path[j]
			row4col[j] = i
			col4row[i], j = j, col4row[i]
			if i == cur {
				break
			}
		}
	}
	return col4row, nil
}
