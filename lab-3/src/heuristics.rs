use crate::core_types::Queen;

pub trait Heuristic {
    fn score(&self, board: &[Queen], n: usize) -> f32;
}

pub struct H1;

impl Heuristic for H1 {
    fn score(&self, board: &[Queen], n: usize) -> f32 {
        let i = board.len();
        let w_row = |row_i| {
            if (row_i + 1) < n / 2 {
                n - (row_i + 1) + 1
            } else {
                row_i + 1
            }
        };

        if n.saturating_sub(i) == 0 {
            return 0.;
        };

        ((n - i)
            * board
                .iter()
                .map(|row_i| w_row(*row_i as usize))
                .sum::<usize>()) as f32
    }
}

pub struct H2;

impl Heuristic for H2 {
    fn score(&self, board: &[Queen], n: usize) -> f32 {
        let mut infeasible_grid = vec![vec![false; n]; n];
        for (x, y) in board.iter().enumerate() {
            let y = *y as usize;
            for i in 0..n {
                // vertical
                infeasible_grid[i][y] = true;
                //horizontal
                infeasible_grid[x][i] = true;
                //diagonal 4 way (ending on board edges)
                infeasible_grid[usize::min(x + i, n - 1)][usize::min(y + i, n - 1)] = true;
                infeasible_grid[x.saturating_sub(i)][usize::min(y + i, n - 1)] = true;
                infeasible_grid[usize::min(x + i, n - 1)][y.saturating_sub(i)] = true;
                infeasible_grid[x.saturating_sub(i)][y.saturating_sub(i)] = true;
            }
        }

        infeasible_grid
            .iter()
            .flatten()
            .fold(0., |acc, cell| if *cell { acc + 1. } else { acc })
    }
}

pub struct H3;

impl Heuristic for H3 {
    fn score(&self, board: &[Queen], n: usize) -> f32 {
        let s = (n as f32 / 2.) * (n as f32 - 1.);

        let dh = {
            let mut acc = 0;

            for i in 0..board.len() - 1 {
                for j in i + 1..board.len() {
                    if board[i] != board[j] {
                        acc += 1;
                    }
                }
            }
            acc
        };

        s - dh as f32
    }
}

pub struct H4;

impl Heuristic for H4 {
    fn score(&self, board: &[Queen], _n: usize) -> f32 {
        let taxicab_dist = |x1: usize, y1: usize, x2: usize, y2: usize| {
            usize::abs_diff(x1, x2) + usize::abs_diff(y1, y2)
        };

        let mut total_distance = 0;
        for i in 0..board.len() {
            let x1 = i;
            let y1 = board[i];
            let x2 = i + 1;

            let first_right = board.get(x2);

            if let Some(y2) = first_right {
                total_distance += taxicab_dist(x1, y1 as usize, x2, *y2 as usize).abs_diff(3);
            }

            let x2 = i + 2;
            let first_right = board.get(x2);

            if let Some(y2) = first_right {
                total_distance += taxicab_dist(x1, y1 as usize, x2, *y2 as usize).abs_diff(3);
            }
        }
        total_distance as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h1() {
        let heuristic = H1;

        let data = vec![2, 3];
        assert_eq!(heuristic.score(&data, 4), 14.);

        let data = vec![2, 0];
        assert_eq!(heuristic.score(&data, 4), 14.);

        let data = vec![0];
        assert_eq!(heuristic.score(&data, 4), 12.);

        let data = vec![];
        assert_eq!(heuristic.score(&data, 4), 0.);
    }
    #[test]
    fn test_h2() {
        let heuristic = H2;

        let data = vec![];
        assert_eq!(heuristic.score(&data, 4), 0.);

        let data = vec![3];
        assert_eq!(heuristic.score(&data, 4), 10.);

        let data = vec![3, 1];
        assert_eq!(heuristic.score(&data, 4), 15.);
    }

    #[test]
    fn test_h3() {
        let heuristic = H3;

        let data = vec![3];
        assert_eq!(heuristic.score(&data, 4), 6.);

        let data = vec![3, 1];
        assert_eq!(heuristic.score(&data, 4), 5.);

        let data = vec![3, 1, 4, 1];
        assert_eq!(heuristic.score(&data, 4), 1.);

        let data = vec![1, 1, 1, 1];
        assert_eq!(heuristic.score(&data, 4), 6.);
    }

    #[test]
    fn test_h4() {
        let heuristic = H4;

        let data = vec![2, 0, 3, 1];
        assert_eq!(heuristic.score(&data, 4), 1.);

        let data = vec![3, 1];
        assert_eq!(heuristic.score(&data, 4), 0.);

        let data = vec![0, 3];
        assert_eq!(heuristic.score(&data, 4), 1.);

        let data = vec![0, 4];
        assert_eq!(heuristic.score(&data, 5), 2.);

        let data = vec![0, 3, 1];
        assert_eq!(heuristic.score(&data, 4), 1.);

        let data = vec![1, 2, 1];
        assert_eq!(heuristic.score(&data, 4), 3.);
    }
}
