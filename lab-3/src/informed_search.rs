use crate::core_types::Queen;
use crate::heuristics::Heuristic;
use itertools::{chain, Itertools};
use std::cmp::{self, Reverse};
use std::collections::BinaryHeap;
use std::time::Instant;

pub struct SearchResult {
    pub solutions: Vec<Vec<Queen>>,
    pub checks: usize,
    pub generated_children: usize,
    pub checks_to_find_one_solution: usize,
    pub seconds: f64,
}

#[derive(PartialEq, Debug)]
struct State {
    data: Vec<Queen>,
    cost: f32,
}

// Marker for impl Ord
impl Eq for State {}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.cost.partial_cmp(&other.cost)
    }
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.cost.total_cmp(&other.cost)
    }
}

pub(crate) fn can_any_queen_beat(queens: &[Queen]) -> bool {
    let enumerated_queens = queens.iter().enumerate();
    for ((idx1, queen1), (idx2, queen2)) in enumerated_queens
        .clone()
        .cartesian_product(enumerated_queens)
    {
        if idx1 == idx2 {
            continue;
        }

        if queen1 == queen2 {
            return true;
        }

        if idx1.abs_diff(idx2) == queen1.abs_diff(*queen2) as usize {
            return true;
        }
    }
    false
}

pub fn solve_with_heuristic<H: Heuristic + ?Sized>(n: usize, heuristic: &H) -> SearchResult {
    println!();
    println!("For n={}", n);
    let now = Instant::now();

    let mut n_generated_states = 0;
    let mut n_checked = 0;
    let mut n_checked_until_first_solution = 0;

    let mut queue = BinaryHeap::new();
    let mut correct_solutions = Vec::new();

    queue.push(Reverse(State {
        data: vec![],
        cost: 0.,
    }));

    while let Some(state) = queue.pop() {
        let state = state.0;
        let depth = state.data.len();
        if depth > n {
            continue;
        }

        if can_any_queen_beat(&state.data) {
            continue;
        }

        n_checked += 1;

        if correct_solutions.is_empty() {
            n_checked_until_first_solution += 1;
        }

        if depth == n {
            correct_solutions.push(state.data);
            continue;
        }

        for new_queen in get_noncovered(n) {
            let mut next_columns = state.data.clone();
            next_columns.push(new_queen);
            n_generated_states += 1;

            queue.push(Reverse(State {
                cost: state.cost + heuristic.score(&next_columns, n),
                data: next_columns,
            }));
        }
    }

    println!("Number of generated states:   {}", n_generated_states);
    println!("Number of checks: {}", n_checked);
    let elapsed = now.elapsed();
    println!("Elapsed:  {:?}", elapsed);
    SearchResult {
        solutions: correct_solutions,
        generated_children: n_generated_states,
        checks: n_checked,
        seconds: elapsed.as_secs_f64(),
        checks_to_find_one_solution: n_checked_until_first_solution,
    }
}

pub(crate) fn get_noncovered(n: usize) -> impl Iterator<Item = Queen> {
    0..n as u8
}

#[allow(dead_code)]
pub(crate) fn get_noncovered_smart(pos: Queen, n: usize) -> impl Iterator<Item = Queen> {
    let lower_part = 0..pos.saturating_sub(1);
    let higher_part = (pos + 2)..(n as u8);
    chain!(lower_part, higher_part)
}

#[cfg(test)]
mod tests {
    use crate::heuristics::H1;

    use super::*;

    #[test]
    fn test_get_noncovered() {
        let n_queens = 8;
        assert!(Iterator::eq(
            get_noncovered_smart(4, n_queens),
            vec![0, 1, 2, 6, 7]
        ));
        assert!(Iterator::eq(
            get_noncovered_smart(0, n_queens),
            vec![2, 3, 4, 5, 6, 7]
        ));
        assert!(Iterator::eq(
            get_noncovered_smart(7, n_queens),
            vec![0, 1, 2, 3, 4, 5]
        ));
        assert!(Iterator::eq(
            get_noncovered_smart(6, n_queens),
            vec![0, 1, 2, 3, 4]
        ));
    }

    #[test]
    fn solution_correctness_proof() {
        assert!(solve_with_heuristic(8, &H1).solutions.len() == 92);

        let all_solutions = solve_with_heuristic(8, &H1);
        assert!(all_solutions
            .solutions
            .contains(&vec![4, 2, 0, 6, 1, 7, 5, 3]));
    }

    #[test]
    fn min_heap_sorted_by_cost() {
        let mut heap = BinaryHeap::new();
        heap.push(Reverse(State {
            data: vec![],
            cost: 21.,
        }));
        heap.push(Reverse(State {
            data: vec![],
            cost: 3.,
        }));
        heap.push(Reverse(State {
            data: vec![],
            cost: 5.5,
        }));
        heap.push(Reverse(State {
            data: vec![],
            cost: 4.3,
        }));
        heap.push(Reverse(State {
            data: vec![],
            cost: 36.,
        }));

        assert_eq!(heap.pop().unwrap().0.cost, 3.);
    }
}
