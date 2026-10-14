package ged

import (
	"errors"
	"math"
)

// bipartiteMatch finds an optimal node assignment with the bipartite approximation of Riesen & Bunke (2009).
//
// It builds an (n+m)×(n+m) cost matrix:
//
//	top-left     n×m  substitution costs sub[i][j]
//	top-right    n×n  deletion costs del[i] on the diagonal, forbidden elsewhere
//	bottom-left  m×m  insertion costs ins[j] on the diagonal, forbidden elsewhere
//	bottom-right m×n  dummy to dummy (0)
//
// It returns the (i → j) pairs assigned as substitutions. Deleted and inserted nodes are not included.
// The same matrix used to be solved by a Python (scipy) service. The solver matches scipy's
// (see linearSumAssignment), so the same assignment is returned, including how ties are broken.
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

// linearSumAssignment solves the minimum-cost perfect assignment of a square cost matrix and returns the column assigned to each row i.
//
// It is a port of scipy.optimize.linear_sum_assignment (rectangular_lsap.cpp), the
// shortest augmenting path method of Crouse (2016). Tie-breaking (ordering remaining columns in reverse,
// preferring unassigned columns) matches scipy too. Floating-point operations are performed in the same order,
// so the same matrix yields the same assignment as scipy.
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
		// ---- Find the shortest augmenting path ----
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

		// ---- Update the dual variables ----
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

		// ---- Augment the solution along the path ----
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
